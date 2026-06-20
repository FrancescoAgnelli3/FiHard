import os
import sys
import math
import argparse
import time
import csv
import pickle
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_factory import get_dataset_cls
from models.motion_pred import *
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from common.evaluation import save_eval_samples_npz
from common.metrics import humanmac_metrics, splineeqnet_diffusion_batch_eval


def loss_function(X, Y_r, Y, mu, logvar):
    MSE = (Y_r - Y).pow(2).sum() / Y.shape[1]
    MSE_v = (X[-1] - Y_r[0]).pow(2).sum() / Y.shape[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / Y.shape[1]
    loss_r = MSE + cfg.lambda_v * MSE_v + cfg.beta * KLD
    return loss_r, np.array([loss_r.item(), MSE.item(), MSE_v.item(), KLD.item()])


def train(epoch):
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    loss_names = ['TOTAL', 'MSE', 'MSE_v', 'KLD']
    generator = dataset.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size)
    for traj_np in generator:
        traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
        traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        X = traj[:t_his]
        Y = traj[t_his:]
        Y_r, mu, logvar = model(X, Y)
        loss, losses = loss_function(X, Y_r, Y, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses += losses
        total_num_sample += 1

    scheduler.step()
    dt = time.time() - t_s
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, dt, losses_str, lr))
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalar('vae_' + name, loss, epoch)
    return float(train_losses[0])


@torch.no_grad()
def evaluate_assembly_vae(eval_sample_num: int, multimodal_threshold: float):
    model.eval()
    stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE', 'CMD', 'FID']
    stats_meter = {x: AverageMeter() for x in stats_names}
    hm_pred_batches = []
    hm_gt_batches = []
    hm_context_batches = []
    saved_obs = []
    saved_tgt = []
    saved_pred = []
    saved_pred_all = []

    data_gen = eval_dataset.iter_generator_with_scale()
    num_samples = 0
    for data, norm_factor in data_gen:
        num_samples += 1
        traj_np = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
        traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()

        X = traj[:t_his]
        X_rep = X.repeat((1, eval_sample_num, 1))
        if eval_sample_num == 1:
            z = torch.zeros((X_rep.shape[1], nz), device=device, dtype=dtype)
            Y = model.sample_prior(X_rep, z=z)
        else:
            Y = model.sample_prior(X_rep)

        pred = Y.permute(1, 0, 2).contiguous().view(data.shape[0], eval_sample_num, t_pred, -1)
        gt = traj[t_his:].permute(1, 0, 2).contiguous().view(data.shape[0], t_pred, -1)
        conditioning_context = traj[:t_his].permute(1, 0, 2).contiguous().view(data.shape[0], t_his, -1, 3)
        norm_factor_t = torch.as_tensor(norm_factor, device=device, dtype=torch.float32).view(-1)

        batch_eval = splineeqnet_diffusion_batch_eval(
            pred_candidates=pred.permute(1, 0, 2, 3).contiguous().to(dtype=torch.float32),
            gt_future=gt.to(dtype=torch.float32),
            conditioning_context=conditioning_context.to(dtype=torch.float32),
            norm_factor=norm_factor_t,
            threshold=float(multimodal_threshold),
        )
        hm_pred_batches.append(pred.permute(1, 0, 2, 3).contiguous().to(dtype=torch.float32).detach().cpu())
        hm_gt_batches.append(gt.to(dtype=torch.float32).detach().cpu())
        hm_context_batches.append(conditioning_context.to(dtype=torch.float32).detach().cpu())
        pred_first_all = pred[0].reshape(eval_sample_num, t_pred, -1, 3)
        gt_first = gt[0].reshape(t_pred, -1, 3)
        obs_first = conditioning_context[0]
        best_idx = torch.norm(pred_first_all - gt_first.unsqueeze(0), dim=-1).mean(dim=(1, 2)).argmin()
        saved_obs.append(obs_first.detach().cpu())
        saved_tgt.append(gt_first.detach().cpu())
        saved_pred.append(pred_first_all[best_idx].detach().cpu())
        saved_pred_all.append(pred_first_all.detach().cpu())

    if num_samples == 0:
        raise RuntimeError("Assembly evaluation produced zero samples.")

    hm = humanmac_metrics(
        pred_candidates=torch.cat(hm_pred_batches, dim=1),
        gt_future=torch.cat(hm_gt_batches, dim=0),
        conditioning_context=torch.cat(hm_context_batches, dim=0),
        threshold=float(multimodal_threshold),
    )
    for k in ('APD', 'ADE', 'FDE', 'MMADE', 'MMFDE', 'CMD', 'FID'):
        stats_meter[k].update(float(hm[k]))

    out_csv = os.path.join(cfg.result_dir, 'stats_1.csv')
    with open(out_csv, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric', 'vae'])
        writer.writeheader()
        for stats in stats_names:
            writer.writerow({'Metric': stats, 'vae': stats_meter[stats].avg})
    eval_samples_path = getattr(cfg, 'eval_samples_path', '')
    if eval_samples_path and saved_obs:
        save_eval_samples_npz(
            Path(str(eval_samples_path)),
            obs=torch.stack(saved_obs, dim=0),
            target=torch.stack(saved_tgt, dim=0),
            pred=torch.stack(saved_pred, dim=0),
            pred_all=torch.stack(saved_pred_all, dim=0),
            metadata={
                'model': 'dlow_cvae',
                'dataset': str(getattr(cfg, 'dataset', '')),
                'action_filter': str(getattr(cfg, 'action_filter', '')),
                'num_candidates': int(eval_sample_num),
            },
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--eval_after_train', action='store_true', default=False)
    parser.add_argument('--eval_sample_num', type=int, default=10)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    cfg = Config(args.cfg, test=args.test)
    tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    """parameter"""
    mode = args.mode
    nz = cfg.nz
    t_his = cfg.t_his
    t_pred = cfg.t_pred

    """data"""
    dataset_cls = get_dataset_cls(cfg.dataset)
    dataset = dataset_cls(
        'train',
        t_his,
        t_pred,
        actions='all',
        dataset_name=cfg.dataset,
        use_vel=cfg.use_vel,
        data_dir=cfg.data_dir,
        action_filter=cfg.action_filter,
        stride=cfg.stride,
        seed=cfg.seed,
        time_interp=cfg.time_interp,
        window_norm=cfg.window_norm,
        splineeqnet_root=getattr(cfg, 'splineeqnet_root', ''),
    )
    if cfg.normalize_data:
        dataset.normalize_data()
    eval_mode = 'val' if cfg.dataset == 'assembly' else 'test'
    eval_dataset = dataset_cls(
        eval_mode,
        t_his,
        t_pred,
        actions='all',
        dataset_name=cfg.dataset,
        use_vel=cfg.use_vel,
        data_dir=cfg.data_dir,
        action_filter=cfg.action_filter,
        stride=cfg.stride,
        seed=cfg.seed,
        time_interp=cfg.time_interp,
        window_norm=cfg.window_norm,
        splineeqnet_root=getattr(cfg, 'splineeqnet_root', ''),
    )
    if cfg.normalize_data:
        eval_dataset.normalize_data(dataset.mean, dataset.std)

    """model"""
    model = get_vae_model(cfg, dataset.traj_dim)
    optimizer = optim.Adam(model.parameters(), lr=cfg.vae_lr)
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_vae_epoch_fix, nepoch=cfg.num_vae_epoch)

    if mode == 'train':
        model.to(device)
        model.train()
        es_enabled = bool(getattr(cfg, 'early_stopping_enabled', False))
        es_patience = max(1, int(getattr(cfg, 'early_stopping_patience', 20)))
        es_min_delta = float(getattr(cfg, 'early_stopping_min_delta', 1e-4))
        es_warmup = max(0, int(getattr(cfg, 'early_stopping_warmup', 0)))
        es_best = None
        es_bad_epochs = 0

        for i in range(0, cfg.num_vae_epoch):
            total_loss = train(i)
            if es_enabled and (i + 1) > es_warmup and np.isfinite(total_loss):
                improved = es_best is None or total_loss < (float(es_best) - es_min_delta)
                if improved:
                    es_best = total_loss
                    es_bad_epochs = 0
                else:
                    es_bad_epochs += 1
                    if es_bad_epochs >= es_patience:
                        print(
                            f"[DLow][EarlyStop] epoch={i + 1} "
                            f"best={float(es_best):.6f} current={float(total_loss):.6f}"
                        )
                        break
        final_checkpoint_path = str(getattr(cfg, 'final_checkpoint_path', '') or '').strip()
        if final_checkpoint_path:
            os.makedirs(os.path.dirname(final_checkpoint_path), exist_ok=True)
            with open(final_checkpoint_path, 'wb') as f:
                pickle.dump({'model_dict': model.state_dict()}, f)
            print(f"Saved final checkpoint to {final_checkpoint_path}")
        if args.eval_after_train:
            evaluate_assembly_vae(
                eval_sample_num=max(1, int(args.eval_sample_num)),
                multimodal_threshold=float(args.multimodal_threshold),
            )
