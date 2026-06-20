import csv
import os
import sys
from pathlib import Path

import pandas as pd
from utils.metrics import *
from tqdm import tqdm
from utils import *
from utils.script import sample_preprocessing

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from common.metrics import humanmac_metrics, splineeqnet_diffusion_batch_eval  # type: ignore
from common.evaluation import save_eval_samples_npz  # type: ignore

tensor = torch.tensor
DoubleTensor = torch.DoubleTensor
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
ones = torch.ones
zeros = torch.zeros

EVAL_SAMPLES_PER_SEQUENCE = 10


def compute_stats(diffusion, multimodal_dict, model, logger, cfg):
    """
    The GPU is strictly needed because we need to give predictions for multiple samples in parallel and repeat for
    several (K=50) times.
    """

    # TODO reduce computation complexity
    def _predict_once(data, model_select):
        traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
        traj = tensor(traj_np, device=cfg.device, dtype=torch.float32)
        traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)
        # traj.shape: [*, t_his + t_pre, 3 * joints_num]

        mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(traj, cfg, mode='metrics')
        sampled_motion = diffusion.sample_ddim(model_select,
                                               traj_dct,
                                               traj_dct_cond,
                                               mode_dict)

        traj_est = torch.matmul(cfg.idct_m_all[:, :cfg.n_pre], sampled_motion)
        # traj_est.shape (K, 125, 48)
        traj_est = traj_est.cpu().numpy()
        return traj_est

    def get_prediction(data, model_select):
        eval_bs = max(1, int(getattr(cfg, 'batch_size', 64)))
        preds = []
        for st in range(0, data.shape[0], eval_bs):
            ed = min(data.shape[0], st + eval_bs)
            preds.append(_predict_once(data[st:ed], model_select))
        pred = np.concatenate(preds, axis=0)
        return pred[None, ...]

    data_group = multimodal_dict['data_group']
    num_samples = int(multimodal_dict['num_samples'])

    stats_names = ['APD', 'ADE', 'FDE', 'MMADE', 'MMFDE', 'CMD', 'FID']
    stats_meter = {x: {y: AverageMeter() for y in ['HumanMAC']} for x in stats_names}

    K = EVAL_SAMPLES_PER_SEQUENCE
    pred = []
    for i in tqdm(range(0, K), position=0):
        # It generates a prediction for all samples in the test set
        # So we need loop for K times
        pred_i_nd = get_prediction(data_group, model)
        pred.append(pred_i_nd)
        if i == K - 1:  # in last iteration, concatenate all candidate pred
            pred = np.concatenate(pred, axis=0)
            pred = torch.from_numpy(pred[:, :, cfg.t_his:, :]).to(cfg.device, dtype=torch.float32)
            all_data = torch.from_numpy(data_group[..., 1:, :]).to(cfg.device, dtype=torch.float32)
            gt_group = all_data[:, cfg.t_his:, :, :].reshape(all_data.shape[0], all_data.shape[1] - cfg.t_his, -1)
            conditioning_context = all_data[:, :cfg.t_his, :, :]

            metrics = humanmac_metrics(
                pred_candidates=pred,
                gt_future=gt_group,
                conditioning_context=conditioning_context,
                threshold=float(cfg.multimodal_threshold),
            )
            for stats in stats_names:
                stats_meter[stats]['HumanMAC'].update(float(metrics[stats]), n=num_samples)

            for stats in stats_names:
                str_stats = f'{stats}: ' + ' '.join(
                    [f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()]
                )
                logger.info(str_stats)
            pred = []

    # save stats in csv
    file_latest = '%s/stats_latest.csv'
    file_stat = '%s/stats.csv'
    with open(file_latest % cfg.result_dir, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + ['HumanMAC'])
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            val = new_meter['HumanMAC']
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    val = float(val.detach().cpu().item())
                else:
                    val = float(val.detach().cpu().mean().item())
            else:
                val = float(val)
            new_meter['HumanMAC'] = val
            new_meter['Metric'] = stats
            writer.writerow(new_meter)
    df1 = pd.read_csv(file_latest % cfg.result_dir)

    if os.path.exists(file_stat % cfg.result_dir) is False:
        df1.to_csv(file_stat % cfg.result_dir, index=False)
    else:
        df2 = pd.read_csv(file_stat % cfg.result_dir)
        df = pd.concat([df2, df1['HumanMAC']], axis=1, ignore_index=True)
        df.to_csv(file_stat % cfg.result_dir, index=False)

