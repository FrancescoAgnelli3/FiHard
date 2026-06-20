# AssemblyHands + MAMP (Steps 1-7)

This setup keeps MAMP's masking strategy (`mask_ratio` + `motion_aware_tau`) for pretraining.

## 1) Representation alignment
- Use SplineEqNet-preprocessed coordinates (wrist-centered/canonicalized).
- Export only `coords` channels (`[..., 4:]`) to MAMP inputs, so MAMP sees `(T, V, 3)`.

## 2) Assembly feeder
- Feeder: `feeder.feeder_assembly.Feeder`
- Supports MAMP NPZ keys (`x_train`, `y_train`, `x_test`, `y_test`) and returns `(C,T,V,M)`.

## 3) Export windows for MAMP
```bash
cd /home/agnelli/projects/MAMP
python3 tools/export_assembly_for_mamp.py \
  --dataset assembly \
  --data-dir /mnt/turing-datasets/AssemblyHands/assembly101-download-scripts/data_our/ \
  --action-filter pick_up_screwd \
  --input-n 70 \
  --window-stride 5 \
  --seed 0 \
  --out data/assembly/assembly_mamp_t70.npz
```

## 4) Pretrain config
- Config file: `config/assembly_joint/pretrain_mamp_t70_layer8+5_mask90.yaml`
- Key settings:
  - `num_frames: 70`
  - `num_joints: 21`
  - `mask_ratio: 0.9`
  - `motion_aware_tau: 0.80`

## 5) Launch pretraining
```bash
cd /home/agnelli/projects/MAMP
bash script_pretrain_mamp_assembly.sh
```

## 6) Extract MAMP features
This exports a global pooled feature per window.

```bash
cd /home/agnelli/projects/MAMP
python3 tools/extract_mamp_features.py \
  --checkpoint output_dir/assembly_joint/pretrain_mamp_t70_layer8+5_mask90_tau0.80_ep400/checkpoint-399.pth \
  --data-path data/assembly/assembly_mamp_t70.npz \
  --split train \
  --config config/assembly_joint/pretrain_mamp_t70_layer8+5_mask90.yaml \
  --mask-ratio 0.0 \
  --motion-aware-tau 0.80 \
  --out data/assembly/assembly_mamp_t70_train_feats.npz
```

If you want masked stochastic extraction too, set `--mask-ratio 0.9`.

## 7) Conditioning granularity
- Current output is one global vector `(D,)` per history window (recommended first integration target).
