# FiHaRD: Frequency-Anisotropic Residual Diffusion for 3D Hand Motion Forecasting

Anonymous codebase for the ACVR workshop submission:

**Forecasting 3D Hand Motion with Spatial and Frequency Anisotropic Residual Diffusion**

This repository contains the training, evaluation, and ablation workspace for **FiHaRD**, our hand-motion forecasting method built around:

- a coarse future predictor in the DCT / frequency domain
- residual diffusion over future hand motion
- spatial anisotropy from the hand graph
- frequency anisotropy over future temporal modes

## Naming Note

The repository still uses `FiHard` as the internal implementation key for FiHaRD:

- main FiHaRD config: `configs/models/fihard.yaml`
- FiHaRD launcher entry: `models.FiHard` inside experiment configs
- FiHaRD implementation: `vendor/splineeqnet/models/fihard.py`

When editing configs or reading outputs, treat `FiHard` as the code name for **FiHaRD**.

## Repository Layout

- `configs/experiment.yaml`: main multi-model experiment config
- `configs/models/`: per-model configuration files
- `configs/ablations/fihard_ablation.yaml`: FiHaRD ablation suite
- `scripts/run_all_models.sh`: main launcher
- `scripts/run_fihard_ablation.sh`: FiHaRD ablation launcher
- `tools/run_all_models.py`: orchestration script for training and evaluation
- `common/`: shared preprocessing, metrics, and evaluation helpers
- `results/`: aggregated experiment tables and analysis outputs
- `vendor/`: external or adapted baseline/model code used by the runner
- `paper-template-Latest/`: anonymized paper and supplementary material sources

## Environment Setup

Create a virtual environment and install dependencies:

```bash
cd /path/to/diffusion_hands_freq
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer `uv`, the same setup works with:

```bash
cd /path/to/diffusion_hands_freq
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

The launcher uses `DIFFUSION_HANDS_PYTHON` when provided, and otherwise falls back to `.venv/bin/python` or `python3`.

## Data Configuration

Dataset roots are resolved in `tools/run_all_models.py`. Current defaults are:

- `assembly`: `/mnt/pve/Turing-Storage2/AssemblyHands/assembly101-download-scripts/data_our/`
- `h2o`: `/mnt/pve/Turing-Storage2/h2o/`
- `bighands`: `/mnt/pve/Turing-Storage2/BigHands/BigHand2.2M/data/`
- `fpha`: `/mnt/pve/Turing-Storage2/FPHA/data/`

You can override these paths in your experiment YAML via `data_roots`.

Shared windowing and preprocessing are defined under `preprocessing` in `configs/experiment.yaml`:

- `input_n`
- `output_n`
- `stride`
- `time_interp`
- `window_norm`
- `eval_batch_mult`

## Running FiHaRD

Run the default experiment config:

```bash
cd /path/to/diffusion_hands_freq
bash scripts/run_all_models.sh
```

Run a specific config:

```bash
bash scripts/run_all_models.sh configs/experiment.yaml
```

Run the FiHaRD ablation suite:

```bash
bash scripts/run_fihard_ablation.sh
```

## Experiment Configuration

The main experiment file is `configs/experiment.yaml`.

Important top-level fields:

- `seed`
- `gpu_index`
- `num_candidates`
- `humanmac_multimodal_threshold`
- `save_model`
- `datasets`
- `action_filter`
- `models`

### Enabling FiHaRD

FiHaRD is enabled through the `FiHard` entry:

```yaml
models:
  FiHard:
    enabled: true
```

Its default hyperparameters live in `configs/models/fihard.yaml`.

### Dataset and Action Handling

- `datasets` may contain one or more datasets and they are processed sequentially.
- `action_filter` is only applied for `assembly`.
- for non-Assembly datasets, the runner ignores `action_filter`.
- if `assembly` is used with no explicit action filter, the code falls back to its default Assembly action selection logic.

## FiHaRD Ablations

The main ablation file is `configs/ablations/fihard_ablation.yaml`.

It includes variants for:

- removing frequency anisotropy
- removing spatial anisotropy
- fully isotropic diffusion
- removing coarse conditioning
- diffusion-only forecasting
- varying the frequency anisotropy strength `q`
- varying the coarse low-frequency DCT budget

Aggregated ablation results are written to:

- `results/fihard_ablations.csv`

## Outputs

The default aggregated metrics file is:

- `results/all_models_metrics_long.csv`

FiHaRD artifacts are typically stored under:

- `out/diffusion_hands_runs/FiHard/<run_id>/`
- `vendor/splineeqnet/out/diffusion_hands_runs/FiHard/<run_id>/`

Depending on the backend, some baseline-specific intermediate outputs may also appear inside their respective `vendor/<model>/` directories.

## Additional Analysis

This repository also includes utilities for downstream analysis, including:

- `tools/analyze_action_temporal_priors.py`
- `tools/bayesian_signed_rank_test.py`
- `results/action_temporal_prior_analysis.md`

These are used to inspect action-dependent temporal priors and compare FiHaRD against baselines.

## Paper Sources

The anonymized ACVR / ECCV workshop paper sources are included in:

- `paper-template-Latest/main.tex`
- `paper-template-Latest/supplementary.tex`

The paper currently defines the method macro as:

```tex
\newcommand{\modelname}{\textsc{FiHaRD}}
```

## Citation

If this work is accepted, the README should be updated with the final workshop citation and any public release links.
