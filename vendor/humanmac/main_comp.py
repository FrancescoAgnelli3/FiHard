import argparse
import json
import os
import sys

import torch
from tensorboardX import SummaryWriter

from config import Config, update_config
from utils import create_logger, seed_set
from utils.evaluation import compute_stats
from utils.script import create_model_and_diffusion, dataset_split, display_exp_setting, get_multimodal_gt_full
from utils.training import Trainer

sys.path.append(os.getcwd())
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from common.preprocessing import split_train_val_test  # type: ignore


def _str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='assembly', help='assembly')
    parser.add_argument('--mode', default='train', help='train / eval / pred / switch / control / zero_shot')
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_seed', type=int, default=None, help='Seed used for SplineEqNet-compatible splits.')
    parser.add_argument('--eval_split', type=str, default='test', choices=['test', 'val'],
                        help='Which SplineEqNet split to evaluate on.')
    parser.add_argument('--card_eval_best_of_k', type=int, default=None,
                        help='SplineEqNet-style best-of-k for stochastic eval. Mapped to HumanMAC mpjpe_best_of_k.')
    parser.add_argument('--num_candidates', type=int, default=None,
                        help='Canonical number of stochastic candidates used throughout the project.')
    parser.add_argument('--card_ddim_steps', type=int, default=None,
                        help='SplineEqNet-style DDIM steps. Mapped to HumanMAC ddim_timesteps.')
    parser.add_argument('--device', type=str,
                        default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--multimodal_th_high', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size (use same value as SplineEqNet for comparability).')
    parser.add_argument('--num_epoch', type=int, default=None,
                        help='Override total number of training epochs.')
    parser.add_argument('--milestone', type=list, default=[75, 150, 225, 275, 350, 450])
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--save_model_interval', type=int, default=20)
    parser.add_argument('--save_gif_interval', type=int, default=20)
    parser.add_argument('--save_metrics_interval', type=int, default=100)
    parser.add_argument('--ckpt', type=str, default='./checkpoints/assembly_ckpt.pt')
    parser.add_argument('--ema', type=_str2bool, default=True)
    parser.add_argument('--vis_switch_num', type=int, default=10)
    parser.add_argument('--vis_col', type=int, default=5)
    parser.add_argument('--vis_row', type=int, default=3)
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Optional dataset directory override (used by assembly).')
    parser.add_argument('--action_filter', type=str, default=None,
                        help='Optional filename filter for dataset files (used by assembly).')
    parser.add_argument('--eval_samples_path', type=str, default=None,
                        help='Optional path where the first sample from each evaluation batch is saved as a compressed bundle.')
    parser.add_argument('--final_checkpoint_path', type=str, default=None,
                        help='Optional path where the final trained model checkpoint is saved.')
    parser.add_argument('--mpjpe_best_of_k', type=int, default=None,
                        help='Legacy alias for the stochastic evaluation sample count.')
    parser.add_argument('--validate_last_epoch_only', type=_str2bool, default=True,
                        help='If True, run validation only at the final epoch.')
    parser.add_argument('--early_stopping_enabled', type=_str2bool, default=None,
                        help='Enable early stopping.')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help='Early stopping patience in epochs.')
    parser.add_argument('--early_stopping_min_delta', type=float, default=None,
                        help='Minimum improvement for early stopping.')
    parser.add_argument('--early_stopping_warmup', type=int, default=None,
                        help='Warmup epochs before enabling early stopping checks.')
    parser.add_argument('--early_stopping_monitor', type=str, default=None,
                        help='Monitor for early stopping: train_loss or val_loss.')
    args = parser.parse_args()

    seed_set(args.seed)
    cfg = Config(f'{args.cfg}', test=(args.mode != 'train'))
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    cfg = update_config(cfg, args_dict)
    if args.num_candidates is not None:
        cfg.mpjpe_best_of_k = max(1, int(args.num_candidates))
    if args.card_eval_best_of_k is not None and args.mpjpe_best_of_k is None:
        cfg.mpjpe_best_of_k = max(1, int(args.card_eval_best_of_k))
    if args.card_ddim_steps is not None:
        cfg.ddim_timesteps = int(args.card_ddim_steps)

    if cfg.dataset in {'assembly', 'h2o', 'bighands', 'fpha'}:
        split_seed = args.seed if args.split_seed is None else args.split_seed
        train_files, val_files, test_files = split_train_val_test(
            data_dir=cfg.data_dir,
            action_filter=cfg.action_filter or '',
            seed=split_seed,
        )
        eval_files = val_files if args.eval_split == 'val' else test_files
        cfg.train_subset_files = train_files
        cfg.test_subset_files = eval_files

        split_manifest = {
            'dataset': cfg.dataset,
            'split_seed': int(split_seed),
            'action_filter': cfg.action_filter or '',
            'data_dir': cfg.data_dir,
            'n_train': len(train_files),
            'n_val': len(val_files),
            'n_test': len(test_files),
            'eval_split': args.eval_split,
            'train_files': train_files,
            'val_files': val_files,
            'test_files': test_files,
        }
        with open(os.path.join(cfg.cfg_dir, 'splineeqnet_compatible_split.json'), 'w') as f:
            json.dump(split_manifest, f, indent=2)

    dataset, dataset_multi_test = dataset_split(cfg)

    tb_logger = SummaryWriter(cfg.tb_dir)
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))
    display_exp_setting(logger, cfg)
    if cfg.dataset in {'assembly', 'h2o', 'bighands', 'fpha'}:
        logger.info(
            "SplineEqNet-compatible split -> "
            f"train={len(cfg.train_subset_files)}, eval={len(cfg.test_subset_files)}, "
            f"eval_split={args.eval_split}, split_seed={args.seed if args.split_seed is None else args.split_seed}, "
            f"batch_size={cfg.batch_size}, ddim_timesteps={cfg.ddim_timesteps}, mpjpe_best_of_k={cfg.mpjpe_best_of_k}"
        )
    model, diffusion = create_model_and_diffusion(cfg)
    logger.info(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.mode == 'train':
        multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        trainer = Trainer(
            model=model,
            diffusion=diffusion,
            dataset=dataset,
            cfg=cfg,
            multimodal_dict=multimodal_dict,
            logger=logger,
            tb_logger=tb_logger,
        )
        trainer.loop()
    elif args.mode == 'eval':
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        multimodal_dict = get_multimodal_gt_full(logger, dataset_multi_test, args, cfg)
        compute_stats(diffusion, multimodal_dict, model, logger, cfg)
    else:
        from utils.demo_visualize import demo_visualize
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt)
        demo_visualize(args.mode, cfg, model, diffusion, dataset)
