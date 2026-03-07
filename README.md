# Unified PSD Compression Library

This repository now organizes PSD compression work under one official model set:

- `KL/PCA` (canonical implementation in `implementation.ipynb`)
- `VAE` (legacy production scripts preserved under `VAE_implementation/scripts/`)
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
python VAE_implementation/scripts/01_preprocess.py --config VAE_implementation/configs/vae_default.yaml
python VAE_implementation/scripts/02_train.py --config VAE_implementation/configs/vae_default.yaml
```

## Key Paths

- Unified package: `psd_compression/`
- New configs: `configs/fwht_default.yaml`, `configs/kl_pca_default.yaml`
- KL/PCA module: `psd_compression/kl_pca/`
- VAE scripts: `VAE_implementation/scripts/`
- Report source/PDF: `docs/main.tex`, `docs/main.pdf`
- Exploratory notebooks: `experiments/`
