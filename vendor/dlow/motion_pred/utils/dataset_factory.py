def get_dataset_cls(dataset_name):
    if dataset_name in {"assembly", "h2o", "bighands", "fpha"}:
        from motion_pred.utils.dataset_assembly import DatasetAssembly
        return DatasetAssembly
    raise ValueError(f"Unsupported dataset '{dataset_name}'.")
