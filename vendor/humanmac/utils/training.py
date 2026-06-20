import copy
import math
import os
import time

from torch import optim, nn

from utils.visualization import render_animation
from models.transformer import EMA
from utils import *
from utils.evaluation import compute_stats
from utils.pose_gen import pose_generator


class Trainer:
    def __init__(self,
                 model,
                 diffusion,
                 dataset,
                 cfg,
                 multimodal_dict,
                 logger,
                 tb_logger):
        super().__init__()

        self.generator_val = None
        self.val_losses = None
        self.t_s = None
        self.train_losses = None

        self.criterion = None
        self.lr_scheduler = None
        self.optimizer = None
        self.generator_train = None

        self.model = model
        self.diffusion = diffusion
        self.dataset = dataset
        self.multimodal_dict = multimodal_dict
        self.cfg = cfg
        self.logger = logger
        self.tb_logger = tb_logger

        self.iter = 0

        self.lrs = []

        if self.cfg.ema is True:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)
            self.ema_setup = (self.cfg.ema, self.ema, self.ema_model)
        else:
            self.ema_model = None
            self.ema_setup = None
        self.early_stopping_enabled = bool(getattr(self.cfg, 'early_stopping_enabled', False))
        self.early_stopping_patience = max(1, int(getattr(self.cfg, 'early_stopping_patience', 20)))
        self.early_stopping_min_delta = float(getattr(self.cfg, 'early_stopping_min_delta', 1e-4))
        self.early_stopping_warmup = max(0, int(getattr(self.cfg, 'early_stopping_warmup', 0)))
        self.early_stopping_monitor = str(getattr(self.cfg, 'early_stopping_monitor', 'train_loss')).strip().lower()
        self._early_best = None
        self._early_bad_epochs = 0

    def _save_final_checkpoint(self):
        final_checkpoint_path = str(getattr(self.cfg, 'final_checkpoint_path', '') or '').strip()
        if not final_checkpoint_path:
            return
        os.makedirs(os.path.dirname(final_checkpoint_path), exist_ok=True)
        state_dict = self.ema_model.state_dict() if self.cfg.ema is True else self.model.state_dict()
        torch.save(state_dict, final_checkpoint_path)
        self.logger.info(f"Saved final checkpoint to {final_checkpoint_path}")

    def loop(self):
        self.before_train()
        for self.iter in range(0, self.cfg.num_epoch):
            did_validate = False
            self.before_train_step()
            self.run_train_step()
            self.after_train_step()
            validate_last_only = bool(getattr(self.cfg, 'validate_last_epoch_only', False))
            should_validate = (self.iter == self.cfg.num_epoch - 1) if validate_last_only else True
            if should_validate:
                self.before_val_step()
                self.run_val_step()
                self.after_val_step()
                did_validate = True
            if self._should_stop_early(self.iter + 1, did_validate):
                if not did_validate:
                    self.before_val_step()
                    self.run_val_step()
                    self.after_val_step()
                break
        self._save_final_checkpoint()

    def _should_stop_early(self, epoch_idx: int, did_validate: bool) -> bool:
        if not self.early_stopping_enabled:
            return False
        if epoch_idx <= self.early_stopping_warmup:
            return False

        monitor_val = None
        if self.early_stopping_monitor == 'val_loss':
            if did_validate and self.val_losses is not None:
                monitor_val = float(self.val_losses.avg)
        else:
            if self.train_losses is not None:
                monitor_val = float(self.train_losses.avg)

        if monitor_val is None or not math.isfinite(monitor_val):
            return False

        improved = self._early_best is None or monitor_val < (float(self._early_best) - self.early_stopping_min_delta)
        if improved:
            self._early_best = monitor_val
            self._early_bad_epochs = 0
            return False
        self._early_bad_epochs += 1
        if self._early_bad_epochs >= self.early_stopping_patience:
            self.logger.info(
                f"[EarlyStop] epoch={epoch_idx} monitor={self.early_stopping_monitor} "
                f"best={float(self._early_best):.6f} current={monitor_val:.6f}"
            )
            return True
        return False

    def before_train(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.cfg.milestone,
                                                           gamma=self.cfg.gamma)
        self.criterion = nn.MSELoss()

    def before_train_step(self):
        self.model.train()
        self.generator_train = self.dataset['train'].sampling_generator(num_samples=self.cfg.num_data_sample,
                                                                        batch_size=self.cfg.batch_size)
        self.t_s = time.time()
        self.train_losses = AverageMeter()
        self.logger.info(f"Starting training epoch {self.iter}:")

    def run_train_step(self):

        for traj_np in self.generator_train:
            with torch.no_grad():
                # (N, t_his + t_pre, joints, 3) -> (N, t_his + t_pre, 3 * (joints - 1))
                # discard the root joint and combine xyz coordinate
                traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                traj = tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad, self.cfg.zero_index)
                traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                traj_dct_mod = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)
                if np.random.random() > self.cfg.mod_train:
                    traj_dct_mod = None

            # train
            t = self.diffusion.sample_timesteps(traj.shape[0]).to(self.cfg.device)
            x_t, noise = self.diffusion.noise_motion(traj_dct, t)
            predicted_noise = self.model(x_t, t, mod=traj_dct_mod)
            loss = self.criterion(predicted_noise, noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            args_ema, ema, ema_model = self.ema_setup[0], self.ema_setup[1], self.ema_setup[2]

            if args_ema is True:
                ema.step_ema(ema_model, self.model)

            self.train_losses.update(loss.item())
            self.tb_logger.add_scalar('Loss/train', loss.item(), self.iter)

            del loss, traj, traj_dct, traj_dct_mod, traj_pad, traj_np

    def after_train_step(self):
        self.lr_scheduler.step()
        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        self.logger.info(
            '====> Epoch: {} Time: {:.2f} Train Loss: {} lr: {:.5f}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.train_losses.avg,
                                                                            self.lrs[-1]))
        if self.iter % self.cfg.save_gif_interval == 0:
            pose_gen = pose_generator(self.dataset['train'], self.model, self.diffusion, self.cfg,
                                      mode='gif', nrow=self.cfg.vis_row)
            render_animation(self.dataset['train'].skeleton, pose_gen, ['HumanMAC'], self.cfg.t_his,
                             ncol=self.cfg.vis_col + 2,
                             output=os.path.join(self.cfg.gif_dir, f'training_{self.iter}.gif'))

    def before_val_step(self):
        self.model.eval()
        self.t_s = time.time()
        self.val_losses = AverageMeter()
        self.generator_val = self.dataset['test'].sampling_generator(num_samples=self.cfg.num_val_data_sample,
                                                                     batch_size=self.cfg.batch_size)
        self.logger.info(f"Starting val epoch {self.iter}:")

    def run_val_step(self):
        for traj_np in self.generator_val:
            with torch.no_grad():
                # (N, t_his + t_pre, joints, 3) -> (N, t_his + t_pre, 3 * (joints - 1))
                # discard the root joint and combine xyz coordinate
                traj_np = traj_np[..., 1:, :].reshape([traj_np.shape[0], self.cfg.t_his + self.cfg.t_pred, -1])
                traj = tensor(traj_np, device=self.cfg.device, dtype=self.cfg.dtype)
                traj_pad = padding_traj(traj, self.cfg.padding, self.cfg.idx_pad,
                                        self.cfg.zero_index)  #
                # [n_pre × (t_his + t_pre)] matmul [(t_his + t_pre) × 3 * (joints - 1)]

                traj_dct = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj)
                traj_dct_mod = torch.matmul(self.cfg.dct_m_all[:self.cfg.n_pre], traj_pad)
                if np.random.random() > self.cfg.mod_train:
                    traj_dct_mod = None

                t = self.diffusion.sample_timesteps(traj.shape[0]).to(self.cfg.device)
                x_t, noise = self.diffusion.noise_motion(traj_dct, t)
                predicted_noise = self.model(x_t, t, mod=traj_dct_mod)
                loss = self.criterion(predicted_noise, noise)

                self.val_losses.update(loss.item())
                self.tb_logger.add_scalar('Loss/val', loss.item(), self.iter)

            del loss, traj, traj_dct, traj_dct_mod, traj_pad, traj_np

    def after_val_step(self):
        self.logger.info('====> Epoch: {} Time: {:.2f} Val Loss: {}'.format(self.iter,
                                                                            time.time() - self.t_s,
                                                                            self.val_losses.avg))

        if self.iter % self.cfg.save_gif_interval == 0:
            if self.cfg.ema is True:
                pose_gen = pose_generator(self.dataset['test'], self.ema_model, self.diffusion, self.cfg,
                                          mode='gif', nrow=self.cfg.vis_row)
            else:
                pose_gen = pose_generator(self.dataset['test'], self.model, self.diffusion, self.cfg,
                                          mode='gif', nrow=self.cfg.vis_row)
            render_animation(self.dataset['test'].skeleton, pose_gen, ['HumanMAC'], self.cfg.t_his,
                             ncol=self.cfg.vis_col + 2,
                             output=os.path.join(self.cfg.gif_dir, f'val_{self.iter}.gif'))

        should_save_metrics = (
            self.cfg.save_metrics_interval > 0
            and (
                (self.iter % self.cfg.save_metrics_interval == 0 and self.iter != 0)
                or (self.iter == self.cfg.num_epoch - 1)
            )
        )
        if should_save_metrics:
            if self.cfg.ema is True:
                compute_stats(self.diffusion, self.multimodal_dict, self.ema_model, self.logger, self.cfg)
            else:
                compute_stats(self.diffusion, self.multimodal_dict, self.model, self.logger, self.cfg)
        if self.cfg.save_model_interval > 0 and (self.iter + 1) % self.cfg.save_model_interval == 0:
            if self.cfg.ema is True:
                torch.save(self.ema_model.state_dict(),
                           os.path.join(self.cfg.model_path, f"ckpt_ema_{self.iter + 1}.pt"))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.cfg.model_path, f"ckpt_{self.iter + 1}.pt"))
