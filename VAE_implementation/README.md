# VAE_implementation

Quick reference for the stable VAE PSD pipeline.

Full documentation:
- `VAE_implementation/TECHNICAL_DOCUMENTATION.md`
- `VAE_implementation/REAL_DEPLOYMENT.md`

## Requirements

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pyzmq
```

Notes:
- On Raspberry Pi, `tflite-runtime` is preferred; `tensorflow` is the fallback.
- On Windows with TensorFlow, use `tf.lite.Interpreter`.

## Script layout

- `VAE_implementation/scripts/training/` training and export
- `VAE_implementation/scripts/analysis/` analysis and benchmarks
- `VAE_implementation/scripts/codec/` protocol and pack/unpack helpers
- `VAE_implementation/scripts/prod/` production edge/server workflows

## Data layout

- `data/raw/` raw acquisitions from external pipelines
- `data/processed/` processed datasets plus split indices
- `data/external/` optional drop area for external datasets

## Base pipeline

```powershell
.\VAE_implementation\scripts\training\00_get_data.ps1
python VAE_implementation/scripts/training/01_preprocess.py --config VAE_implementation/configs/vae_default.yaml
python VAE_implementation/scripts/training/02_train.py --config VAE_implementation/configs/vae_default.yaml
python VAE_implementation/scripts/training/03_eval.py --config VAE_implementation/configs/vae_default.yaml --use_global_best
python VAE_implementation/scripts/training/04_export_tflite.py --config VAE_implementation/configs/vae_default.yaml --use_global_best
```

## External dataset acquisition

Dataset acquisition happens outside this repository. This project assumes an
external pipeline already produces PSD data in real time or in batch mode.

If `data/raw/DataBase-RF-FM-88MHz-108MHz-Bogota-Funza` is unavailable, the
preprocess step first reuses the existing processed dataset and, if that
dataset is also missing, can fall back to the campaign export in
`data/campaigns/MeasurementCalibration` for local smoke tests.

## Local test

Receiver:

```powershell
python VAE_implementation/scripts/prod/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --save_every_packets 1 --idle_stop_s 3 --invert_norm_to_original
```

Sender:

```powershell
python VAE_implementation/scripts/prod/10_udp_sender_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --source dataset --split test --block_len 30 --n_blocks 3 --zlib_level 1
```

## SDR production

Stable edge bridge:

```bash
python VAE_implementation/scripts/prod/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --use_global_best \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --dest_ip <SERVER_IP> --port 5005 \
  --block_len 30 --zlib_level 1
```

## External pipeline integration

This repository consumes PSD frames through ZMQ/JSON. The PSD producer must
publish to `ipc:///tmp/ane_psd.ipc` using a compatible key such as `psd_dbm`.

Integration options:
- `12_acq_sensor_to_zmq.py` if you need to adapt another repository or process that emits PSD frames
- `13_rf_engine_controller_to_zmq.py` if you use `rf_engine` and cannot modify its orchestrator
