import os
import sys
import csv
import math
import time
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from einops import rearrange
from torchvision import transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader
from scipy.spatial.distance import pdist, squareform

sys.path.append(os.getcwd())
from utils import *
from models.load_models import get_model
from models.GaussianDiffusion import GaussianDiffusion
from utils.metrics import APD, APDE, ADE, FDE, MMADE, MMFDE, SPLINEEQNET_DIFFUSION_BATCH_EVAL
from data_utils.transforms import calculate_stats, load_stats
from data_utils.dataset_assembly import build_assembly_train_val
from data_utils.transforms import DataAugmentation
from common.evaluation import save_eval_samples_npz


def generate_assembly_loss_weights(t_length, scale=10):
    """
    Assembly uses a 21-joint hand (63 dims).
    Keep a fixed weighting shape without importing external dataset dependencies.
    """
    assert scale > 0 and scale != 1
    weights = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 18, 13, 15, 19, 16, 20, 17, 21]
    weights = np.repeat(weights, 3)
    chain = [[0],
             [128.13, 38.05, 41.24, 13.91],
             [128.68, 38.06, 40.78, 14.27],
             [112.41, 114.3, 5.85, 21.91, 9.84],
             [14.06, 11.88, 25.8, 25.78],
             [14.51, 11.43, 25.87, 26.24]]
    new_chain = []
    for x in chain:
        s = sum(x)
        if s == 0:
            new_chain.append([0])
            continue
        new_x = []
        for i in range(len(x)):
            new_x.append((i + 1) / (len(x)) * math.log(sum(x[:i]) + scale))
        new_chain.append(new_x)
    new_chain = [item for sublist in new_chain for item in sublist]
    s_weight = [new_chain[weights[i]] for i in range(len(weights))]
    s_weight = np.asarray(s_weight)
    t_weight = np.ones(t_length)
    ret = s_weight[None, :] * t_weight[:, None]
    ret = ret / np.sum(ret) * 63 * t_length
    return torch.from_numpy(ret)


def generate_loss_weight(cfg):
    if cfg.dataset == 'assembly' and bool(getattr(cfg, 'remove_model_internals', False)):
        return torch.ones((cfg.t_his + cfg.t_pred, cfg.node_n), dtype=torch.float32)
    if cfg.dataset in {'assembly', 'h2o', 'bighands', 'fpha'}:
        gen_weight = generate_assembly_loss_weights
    else:
        raise ValueError(f"Unsupported dataset in this pruned build: {cfg.dataset}")
    # history recon weight
    in_weights = gen_weight(cfg.t_his, scale=cfg.loss_weight_scale)
    # future prediction weight
    out_weights = gen_weight(cfg.t_pred, scale=cfg.loss_weight_scale)
    loss_weights = torch.cat((in_weights, out_weights), dim=0)
    return loss_weights


class Trainer(object):
    def __init__(
        self,
        dataset,
        diffusion_model,
        cfg,
        train_batch_size=16,
        train_lr=1e-4,
        weight_decay=0,
        actions='all',
    ):
        super().__init__()

        self.model = diffusion_model
        self.device = next(self.model.parameters()).device

        self.cfg = cfg

        self.batch_size = train_batch_size
        self.input_n = cfg.t_his
        self.output_n = cfg.t_pred
        self.dtype = torch.float32 if self.cfg.dtype == 'float32' else torch.float64
        self.is_assembly = self.cfg.dataset in {'assembly', 'h2o', 'bighands', 'fpha'}
        self.drop_root_joint = not self.is_assembly
        
        # dataset and dataloader initialization
        if self.is_assembly:
            print('Preparing Assembly datasets with SplineEqNet pipeline...')
            self.train_dataset, self.eval_dataset, self.train_dataloader, self.eval_dataloader = build_assembly_train_val(
                cfg,
                self.input_n,
                self.output_n,
                self.batch_size,
                seed=getattr(cfg, 'seed', 0),
            )
            self.mean, self.std, self.minv, self.maxv = None, None, None, None
            self.mean_torch, self.std_torch = None, None
            self.minv_torch, self.maxv_torch = None, None
            self.multimodal_traj = None
            self.mmapd = None
        else:
            transform = transforms.Compose([DataAugmentation(cfg.rota_prob)])
            stat_dataset = dataset('train', self.input_n, self.output_n, augmentation=cfg.augmentation, stride=cfg.stride, transform=transform, dtype=cfg.dtype)
           
            # info saving
            stats_folder = os.path.join('auxiliar/datasets/', cfg.dataset, "stats", stat_dataset.stat_id)
            mmapd_path = os.path.join('auxiliar/datasets/', cfg.dataset, 'mmapd_GT.csv')

            if not os.path.exists(stats_folder) or len(os.listdir(stats_folder)) == 0:
                print('Calculating stats...')
                calculate_stats(stat_dataset)
            else:
                print('Stats precomputed.')
            print('Loading stats...')
            
            self.mean, self.std, self.minv, self.maxv = load_stats(stat_dataset)
            self.mean_torch = torch.from_numpy(self.mean[1:,:]).to(self.device).to(self.dtype)
            self.std_torch = torch.from_numpy(self.std[1:,:]).to(self.device).to(self.dtype)
            self.minv_torch = torch.from_numpy(self.minv[1:,:]).to(self.device).to(self.dtype)
            self.maxv_torch = torch.from_numpy(self.maxv[1:,:]).to(self.device).to(self.dtype)

            print('Preparing datasets...')
            self.train_dataset = dataset('train', self.input_n, self.output_n, augmentation=cfg.augmentation, stride=cfg.stride, transform=transform, dtype=cfg.dtype)
            self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)

            # NOTE: test partition: can be seen only once
            self.eval_dataset = dataset('test', self.input_n, self.output_n, augmentation=0, stride=1, transform=None, dtype=cfg.dtype)
            self.eval_dataloader = DataLoader(dataset=self.eval_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
            
            # multimodal GT
            print('Calculating mmGT...')
            self.multimodal_traj = self.get_multimodal_gt()

            # APDE computation loading
            self.mmapd = self.get_mmapd(mmapd_path)

        print('No FID classifier available...')
        self.classifier_for_fid = None

        # optimizer
        self.opt = AdamW(self.model.parameters(), lr=train_lr, betas=(0.9, 0.99), weight_decay=weight_decay)    # weight_decay is 0, same as Adam Pytorch Implementation
        self.scheduler = get_scheduler(self.opt, policy=cfg.sched_policy, nepoch_fix=cfg.num_epoch_fix_lr, nepoch=cfg.train_epoch)

        # epoch counter state
        self.epoch = 0
        self.train_loss_list = []
        print('Trainer initialization done.')

    def _flatten_motion(self, traj):
        """
        traj: [B, T, N, 3] -> [B, T, NC]
        """
        if self.drop_root_joint:
            traj = traj[..., 1:, :]
        return traj.reshape(traj.shape[0], traj.shape[1], -1)

    def _batch_to_traj_and_extra(self, batch):
        """
        Unified loader interface.
        Returns:
            traj_np: [B, T_total, N, 3]
            extra: dict with keys at least {'act', 'norm_factor'}
        """
        if self.is_assembly:
            if not isinstance(batch, (list, tuple)) or len(batch) < 3:
                raise ValueError("Assembly loader must return (inp, out, norm_factor[, ...]).")
            inp, out, norm_factor = batch[:3]
            traj_np = torch.cat((inp[:, :, :, 4:], out[:, :, :, 4:]), dim=1)
            extra = {
                'act': torch.zeros(inp.shape[0], dtype=torch.int64),
                'norm_factor': norm_factor,
            }
            return traj_np, extra

        if not isinstance(batch, (list, tuple)) or len(batch) != 2:
            raise ValueError("Expected CoMusion datasets to return (traj, extra).")
        traj_np, extra = batch
        if 'norm_factor' not in extra:
            extra['norm_factor'] = torch.ones(traj_np.shape[0], dtype=traj_np.dtype)
        return traj_np, extra

    def _mpjpe_candidates_from_flat(self, pred, gt):
        """
        pred: [B, S, T, NC], gt: [B, T, NC]
        returns per-sample-candidate mean joint distance: [B, S]
        """
        if pred.shape[-1] % 3 != 0:
            raise ValueError(f"Expected flattened coord dim divisible by 3, got {pred.shape[-1]}")
        pred_j = pred.reshape(pred.shape[0], pred.shape[1], pred.shape[2], -1, 3)
        gt_j = gt.reshape(gt.shape[0], gt.shape[1], -1, 3).unsqueeze(1)
        return torch.linalg.norm(pred_j - gt_j, dim=-1).mean(dim=(-1, -2))

    def train(self):
        self.model.train()
        t_s = time.time()
        epoch_loss = 0.
        epoch_iter = 0
        epoch_loss_info = {}
        for batch in self.train_dataloader:
            traj_np, extra = self._batch_to_traj_and_extra(batch)
            traj = self._flatten_motion(traj_np).to(self.device).to(self.dtype)
            div_k = 1 if (self.is_assembly and bool(getattr(self.cfg, 'remove_model_internals', False))) else self.cfg.div_k
            loss, loss_info = self.model(traj, None, div_k=div_k, uncond=True, mmgt=None)
            for key, value in loss_info.items():
                if key not in epoch_loss_info:
                    epoch_loss_info[key] = value
                else:
                    epoch_loss_info[key] += value
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
            epoch_iter += 1

        self.scheduler.step()
        self.epoch += 1
        epoch_loss /= epoch_iter
        for key, value in epoch_loss_info.items():
            epoch_loss_info[key] /= epoch_iter
        lr = self.opt.param_groups[0]['lr']
        dt = time.time() - t_s
        self.train_loss_list.append(epoch_loss)
        return lr, epoch_loss, epoch_loss_info, dt 

    def get_multimodal_gt(self):
        """
        return list of tensors of shape [[num_similar, t_pred, NC]]
        """
        all_data = []
        for i, batch in enumerate(self.eval_dataloader):
            data, _ = self._batch_to_traj_and_extra(batch)
            data = self._flatten_motion(data)
            all_data.append(data)
        all_data = torch.cat(all_data, dim=0).cpu().numpy()
        all_context = all_data[:, :self.input_n, :]
        pd = squareform(pdist(all_context.reshape(all_context.shape[0], -1)))
        traj_gt_arr = []
        for i in range(pd.shape[0]):
            ind = np.nonzero(pd[i] < self.cfg.multimodal_threshold)
            traj_gt_arr.append(torch.from_numpy(all_data[ind][:, self.input_n:, :]).to(self.dtype))
        return traj_gt_arr

    def get_mmapd(self, mmapd_path):
        df = pd.read_csv(mmapd_path)
        mmapds = torch.as_tensor(list(df["gt_APD"]))
        return mmapds

    def get_prediction(self, data, act, sample_num, uncond, use_ema=True, concat_hist=False):
        """
        data: [batch_size, total_len, num_joints=17, 3]
        act:  [batch_size]
        sample_num: how many samples to generate for one data entry
        """
        traj = self._flatten_motion(data).to(self.device).to(self.dtype)

        # process x_0_history: [b*sample_num, t_pred, nc]
        x_0_history = torch.repeat_interleave(traj[:,:-self.output_n,:], sample_num, dim=0)
        total_sample_num = x_0_history.shape[0]
        Y = self.model.sample(x_0_history, None, batch_size=total_sample_num, clip_denoised=False, uncond=uncond)       # [b*sample_num, t_pred, nc]

        if concat_hist:
            Y = torch.cat((x_0_history, Y), dim=1)
        Y = Y.contiguous()

        return Y

    @torch.no_grad()
    def compute_stats(self):
        """
        return: dic [stat_name, stat_val] NOTE: val.avg is standard
        """
        self.model.eval()

        if self.is_assembly:
            stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE', 'CMD', 'FID']
            stats_meter = {x: AverageMeterTorch() for x in stats_names}
            hm_pred_batches = []
            hm_gt_batches = []
            hm_context_batches = []
            saved_obs = []
            saved_tgt = []
            saved_pred = []
            saved_pred_all = []
            for batch_idx, batch in enumerate(self.eval_dataloader):
                data, extra = self._batch_to_traj_and_extra(batch)
                data_flat = self._flatten_motion(data)
                gt = data_flat[:, self.input_n:, :].to(self.device).to(self.dtype)
                k = max(1, int(self.cfg.eval_sample_num))
                pred_tries = []
                for sample_idx in range(k):
                    sample_seed = int(batch_idx * 1000003 + sample_idx)
                    set_global_seed(sample_seed)
                    pred_try = self.get_prediction(
                        data,
                        extra['act'],
                        sample_num=1,
                        uncond=True,
                        concat_hist=False,
                    ).detach()
                    pred_tries.append(pred_try)

                pred = torch.stack(pred_tries, dim=1)  # [B, K, T, NC]
                norm_factor = extra.get('norm_factor', torch.ones(gt.shape[0]))
                norm_factor = norm_factor.to(self.device).to(self.dtype).reshape(-1)
                if not torch.isfinite(norm_factor).all():
                    raise RuntimeError("CoMusion eval received non-finite norm_factor values.")
                if (norm_factor <= 0).any():
                    raise RuntimeError("CoMusion eval received non-positive norm_factor values.")
                if batch_idx == 0:
                    print(
                        "[CoMusion eval] norm_factor stats: "
                        f"min={float(norm_factor.min().item()):.6f}, "
                        f"max={float(norm_factor.max().item()):.6f}, "
                        f"mean={float(norm_factor.mean().item()):.6f}"
                    )

                conditioning_context = data_flat[:, :self.input_n, :].reshape(gt.shape[0], self.input_n, -1, 3)

                batch_eval = SPLINEEQNET_DIFFUSION_BATCH_EVAL(
                    pred.permute(1, 0, 2, 3).contiguous(),  # [K, B, T, NC]
                    gt,
                    conditioning_context.to(self.device).to(self.dtype),
                    norm_factor,
                    threshold=float(self.cfg.humanmac_multimodal_threshold),
                )
                hm_pred_batches.append(pred.permute(1, 0, 2, 3).detach().cpu())
                hm_gt_batches.append(gt.detach().cpu())
                hm_context_batches.append(conditioning_context.detach().cpu())
                pred_first_all = pred[0].reshape(k, self.cfg.t_pred, -1, 3)
                gt_first = gt[0].reshape(self.cfg.t_pred, -1, 3)
                obs_first = conditioning_context[0]
                best_idx = torch.norm(pred_first_all - gt_first.unsqueeze(0), dim=-1).mean(dim=(1, 2)).argmin()
                saved_obs.append(obs_first.detach().cpu())
                saved_tgt.append(gt_first.detach().cpu())
                saved_pred.append(pred_first_all[best_idx].detach().cpu())
                saved_pred_all.append(pred_first_all.detach().cpu())

                print('-' * 80)
                print(f"Samples processed: {sum(batch.shape[0] for batch in hm_gt_batches)}")

            hm_metrics = SPLINEEQNET_DIFFUSION_BATCH_EVAL(
                torch.cat(hm_pred_batches, dim=1),
                torch.cat(hm_gt_batches, dim=0),
                torch.cat(hm_context_batches, dim=0),
                torch.ones((torch.cat(hm_gt_batches, dim=0).shape[0],), dtype=torch.float32),
                threshold=float(self.cfg.humanmac_multimodal_threshold),
            )['humanmac']
            eval_samples_path = getattr(self.cfg, 'eval_samples_path', '')
            if eval_samples_path and saved_obs:
                save_eval_samples_npz(
                    Path(str(eval_samples_path)),
                    obs=torch.stack(saved_obs, dim=0),
                    target=torch.stack(saved_tgt, dim=0),
                    pred=torch.stack(saved_pred, dim=0),
                    pred_all=torch.stack(saved_pred_all, dim=0),
                    metadata={
                        'model': 'comusion',
                        'dataset': str(getattr(self.cfg, 'dataset', '')),
                        'action_filter': str(getattr(self.cfg, 'action_filter', '')),
                        'num_candidates': int(k),
                    },
                )
            for key in ('APD', 'ADE', 'FDE', 'MMADE', 'MMFDE', 'CMD', 'FID'):
                stats_meter[key].direct_set_avg(hm_metrics[key])
            return stats_meter

        raise RuntimeError("Only assembly evaluation is supported in this pruned build.")

    def evaluation(self):
        """NOTE: can be only called once"""
        stats_dic = self.compute_stats()
        return {x: y.avg for x, y in stats_dic.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    """setup"""
    cfg = Config(args.cfg, test=False)
    cfg.seed = args.seed
    set_global_seed(args.seed)
    dtype = torch.float32 if cfg.dtype == 'float32' else torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)

    """parameter"""
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    node_n = cfg.node_n

    """data"""
    if cfg.dataset in {'assembly', 'h2o', 'bighands', 'fpha'}:
        dataset_cls = None
    else:
        raise ValueError(f"Unsupported dataset in this pruned build: {cfg.dataset}")
    action = 'all'

    """loss weight"""
    loss_weights = generate_loss_weight(cfg)    # [t_all/r_pred, NxC]

    """model"""
    model = get_model(cfg).to(dtype).to(device)
    diffuser = GaussianDiffusion(
        model=model,
        cfg=cfg,
        future_motion_size=(t_pred, node_n), # [T_pred, N*C=num_nodes]
        timesteps=cfg.diffuse_steps,
        loss_type=cfg.loss_type,
        objective=cfg.objective,
        beta_schedule=cfg.beta_schedule,
        history_weight=1 if (cfg.dataset == 'assembly' and bool(getattr(cfg, 'remove_model_internals', False))) else cfg.history_weight,
        future_weight=1 if (cfg.dataset == 'assembly' and bool(getattr(cfg, 'remove_model_internals', False))) else cfg.future_weight,
        st_loss_weight=loss_weights,
    ).to(dtype).to(device)

    """trainer"""
    trainer = Trainer(
        dataset=dataset_cls,
        diffusion_model=diffuser,
        train_batch_size=cfg.batch_size,
        train_lr=cfg.train_lr,
        weight_decay=cfg.weight_decay,
        actions=action,
        cfg=cfg,
    )

    print(">>> model on:", device)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    es_enabled = bool(getattr(cfg, 'early_stopping_enabled', False))
    es_patience = max(1, int(getattr(cfg, 'early_stopping_patience', 20)))
    es_min_delta = float(getattr(cfg, 'early_stopping_min_delta', 1e-4))
    es_warmup = max(0, int(getattr(cfg, 'early_stopping_warmup', 0)))
    es_best = None
    es_bad_epochs = 0

    # Train from scratch for this run only (no checkpoint save/load).
    for epoch in range(0, cfg.train_epoch):
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        lr, epoch_loss, epoch_loss_info, dt = trainer.train()

        print(">>> epoch: ", epoch)

        ret_log = np.append(ret_log, [lr, dt, epoch_loss])
        head = np.append(head, ['lr', 'dt', 't_l'])

        for key, value in epoch_loss_info.items():
            head = np.append(head, key)
            ret_log = np.append(ret_log, value)

        save_csv_log(cfg, head, ret_log, epoch == 0, file_name=cfg.id + '_log')

        if es_enabled and (epoch + 1) > es_warmup:
                monitored = float(epoch_loss)
                if np.isfinite(monitored):
                    improved = es_best is None or monitored < (float(es_best) - es_min_delta)
                    if improved:
                        es_best = monitored
                        es_bad_epochs = 0
                    else:
                        es_bad_epochs += 1
                        if es_bad_epochs >= es_patience:
                            print(
                                f"[CoMusion][EarlyStop] epoch={epoch + 1} "
                                f"best={float(es_best):.6f} current={monitored:.6f}"
                            )
                            break


    print('Compute final stats...')
    stats = trainer.evaluation()
    print(stats)

    final_checkpoint_path = str(getattr(cfg, 'final_checkpoint_path', '') or '').strip()
    if final_checkpoint_path:
        os.makedirs(os.path.dirname(final_checkpoint_path), exist_ok=True)
        with open(final_checkpoint_path, 'wb') as f:
            pickle.dump({'model_dict': model.state_dict()}, f)
        print(f"Saved final checkpoint to {final_checkpoint_path}")


    with open('%s/eval_stats.csv' % (cfg.result_dir), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, stats.keys())
        writer.writeheader()
        writer.writerow(stats)
    print('Done.')


if __name__ == '__main__':
    main()
