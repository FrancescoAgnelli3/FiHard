import os
import sys


def build_assembly_train_val(cfg, t_his, t_pred, train_batch_size, seed):
    """
    Build Assembly train/val datasets using the exact SplineEqNet data pipeline.
    """
    spline_root = cfg.splineeqnet_root
    if not os.path.isdir(spline_root):
        raise FileNotFoundError(
            f"SplineEqNet root not found: {spline_root}. "
            "Set data_specs.splineeqnet_root in the config."
        )
    if spline_root not in sys.path:
        sys.path.insert(0, spline_root)

    from config import DatasetCfg
    from data import build_datasets, get_dataset_metadata, make_loaders

    dataset_name = str(getattr(cfg, "dataset", "assembly")).lower()
    metadata = get_dataset_metadata(dataset_name)
    data_dir = cfg.data_dir or metadata.get("default_dir", "")
    action_filter = (
        metadata.get("default_action_filter", "")
        if getattr(cfg, "action_filter", None) is None
        else str(cfg.action_filter)
    )
    wrist_indices = tuple(int(idx) for idx in metadata.get("default_wrist_indices", (5, 26)))

    ds_cfg = DatasetCfg(
        data_dir=data_dir,
        action_filter=action_filter,
        input_n=int(t_his),
        output_n=int(t_pred),
        stride=int(cfg.stride),
        time_interp=getattr(cfg, "time_interp", None),
        window_norm=getattr(cfg, "window_norm", None),
        batch_size=int(train_batch_size),
        eval_batch_mult=int(cfg.eval_batch_mult),
        seed=int(seed),
        wrist_indices=wrist_indices,
        dataset=dataset_name,
        node_count=int(metadata.get("node_count", 21)),
        edge_index=tuple(metadata.get("edge_index", ())),
        adjacency=tuple(metadata.get("adjacency", ())),
    )

    train_dataset, val_dataset, test_dataset = build_datasets(ds_cfg)
    train_loader, _, test_loader = make_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        ds_cfg.batch_size,
        ds_cfg.seed,
        ds_cfg.eval_batch_mult,
    )
    # Keep helper name for backward compatibility; evaluation uses test split for comparability.
    return train_dataset, test_dataset, train_loader, test_loader
