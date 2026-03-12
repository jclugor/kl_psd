# VAE_implementation

Guia rapida del pipeline VAE PSD (version estable).

Documentacion completa:
- `VAE_implementation/DOCUMENTACION_TECNICA.md`
- `VAE_implementation/IMPLEMENTACION_REAL.md`

## Requerimientos

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pyzmq
```

Notas:
- En Raspberry: ideal `tflite-runtime`; fallback `tensorflow`.
- En Windows/TensorFlow usar `tf.lite.Interpreter`.

## Estructura de scripts

- `VAE_implementation/scripts/training/` entrenamiento y export
- `VAE_implementation/scripts/analysis/` analisis y benchmarks
- `VAE_implementation/scripts/codec/` protocolo y pack/unpack
- `VAE_implementation/scripts/prod/` produccion (edge/servidor)

## Estructura de datos

- `data/raw/` datos crudos (adquisicion externa)
- `data/processed/` datasets procesados + splits
- `data/external/` drop area opcional para datasets externos

## Pipeline base

```powershell
.\VAE_implementation\scripts\00_get_data.ps1
python VAE_implementation/scripts/training/01_preprocess.py --config VAE_implementation/configs/vae_default.yaml
python VAE_implementation/scripts/training/02_train.py --config VAE_implementation/configs/vae_default.yaml
python VAE_implementation/scripts/training/03_eval.py --config VAE_implementation/configs/vae_default.yaml --use_global_best
python VAE_implementation/scripts/training/04_export_tflite.py --config VAE_implementation/configs/vae_default.yaml --use_global_best
```

## Adquisicion de dataset (externa)

La adquisicion de dataset se realiza fuera de este repositorio. Este repo asume que ya existe
un pipeline externo que produce PSD en tiempo real o por lotes.

## Test local 

Receiver:
```powershell
python VAE_implementation/scripts/prod/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --save_every_packets 1 --idle_stop_s 3 --invert_norm_to_original
```

Sender:
```powershell
python VAE_implementation/scripts/prod/10_udp_sender_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --source dataset --split test --block_len 30 --n_blocks 3 --zlib_level 1
```

## Produccion SDR

Bridge edge (estable):
```bash
python VAE_implementation/scripts/prod/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --use_global_best \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --dest_ip <IP_SERVIDOR> --port 5005 \
  --block_len 30 --zlib_level 1
```

## Integracion con un pipeline externo

Este repo consume PSD via ZMQ/JSON. El productor de PSD debe publicar a:
`ipc:///tmp/ane_psd.ipc` con una llave compatible (por ejemplo `psd_dbm`).

Opciones de integracion:
- `12_acq_sensor_to_zmq.py` si necesitas adaptar otro repo o un proceso que emite PSD.
- `13_rf_engine_controller_to_zmq.py` si usas `rf_engine` y no puedes modificar su orchestrator.
