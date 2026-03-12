# Unified PSD Compression Library

This repository now organizes PSD compression work under one official model set:

- `KL/PCA` (canonical notebook in `notebooks/core/implementation.ipynb`)
- `VAE` (production scripts preserved under `VAE_implementation/scripts/` with subfolders)
- `FWHT` (deterministic transform codec in `psd_compression/fwht/`)

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Unified CLI

Standard entrypoint:

```powershell
python -m psd_compression.cli --help
```

### Task Matrix

| Model | Command Group | Core Commands |
|---|---|---|
| FWHT | `fwht` | `encode`, `decode`, `evaluate` |
| VAE | `vae` | `preprocess`, `train`, `eval`, `export`, `entropy`, `benchmark` |
| KL/PCA | `kl-pca` | `fit`, `encode`, `decode`, `evaluate` |

### Canonical Commands

FWHT evaluation:

```powershell
python -m psd_compression.cli fwht evaluate --config configs/fwht_default.yaml --max-frames 64
```

VAE training wrapper (delegates to legacy script):

```powershell
python -m psd_compression.cli vae train -- --config VAE_implementation/configs/vae_default.yaml
```

KL/PCA fit:

```powershell
python -m psd_compression.cli kl-pca fit --config configs/kl_pca_default.yaml
```

KL/PCA evaluation:

```powershell
python -m psd_compression.cli kl-pca evaluate --config configs/kl_pca_default.yaml --max-frames 64
```

## Backward Compatibility

Legacy commands remain valid. For example:

```powershell
python VAE_implementation/scripts/training/01_preprocess.py --config VAE_implementation/configs/vae_default.yaml
python VAE_implementation/scripts/training/02_train.py --config VAE_implementation/configs/vae_default.yaml
```

## Project Layout

- `psd_compression/`: canonical runtime package for CLI, FWHT, KL/PCA, and shared utilities.
- `api/`: notebook-facing compatibility package for the remote measurement API client.
- `VAE_implementation/`: legacy VAE workflow kept stable because its scripts still depend on fixed filesystem paths.
- `configs/`: hand-edited runtime configuration files.
- `data/`: datasets and campaign exports only.
- `notebooks/`: all notebooks grouped by purpose (`core/`, `examples/`, `experiments/`, `data/`, `vae/`).
- `docs/`: LaTeX sources plus published PDFs.
- `deploy/`: deployment bundles and device-specific payloads.
- `tests/`: repository verification.

## Key Paths

- Unified package: `psd_compression/`
- API client package: `api/` with canonical implementation in `psd_compression/api/`
- New configs: `configs/fwht_default.yaml`, `configs/kl_pca_default.yaml`
- KL/PCA module: `psd_compression/kl_pca/`
- VAE scripts: `VAE_implementation/scripts/{training,analysis,codec,prod}/`
- Report source/PDF: `docs/main.tex`, `docs/main.pdf`
- Notebook hub: `notebooks/`
