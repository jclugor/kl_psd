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

## Pipeline base

```powershell
.\VAE_implementation\scripts\00_get_data.ps1
python VAE_implementation/scripts/01_preprocess.py --config VAE_implementation/configs/vae_default.yaml
python VAE_implementation/scripts/02_train.py --config VAE_implementation/configs/vae_default.yaml
python VAE_implementation/scripts/03_eval.py --config VAE_implementation/configs/vae_default.yaml --use_global_best
python VAE_implementation/scripts/04_export_tflite.py --config VAE_implementation/configs/vae_default.yaml --use_global_best
```

## Test local 

Receiver:
```powershell
python VAE_implementation/scripts/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --save_every_packets 1 --idle_stop_s 3 --invert_norm_to_original
```

Sender:
```powershell
python VAE_implementation/scripts/10_udp_sender_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --source dataset --split test --block_len 30 --n_blocks 3 --zlib_level 1
```

## Produccion SDR

Bridge edge (estable):
```bash
python VAE_implementation/scripts/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --use_global_best \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --dest_ip <IP_SERVIDOR> --port 5005 \
  --block_len 30 --zlib_level 1
```

## Integracion con SDR-SpectrumMonitoring-Sensor

Publica PSD del repo externo hacia IPC con:

```bash
python VAE_implementation/scripts/12_acq_sensor_to_zmq.py \
  --mode callable \
  --sensor_repo_path /home/pi/SDR-SpectrumMonitoring-Sensor \
  --callable "mi_modulo.mi_fuente:get_next_psd" \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --out_key psd_dbm
```

Alternativa por stdout JSON:

```bash
python VAE_implementation/scripts/12_acq_sensor_to_zmq.py \
  --mode script \
  --script_cmd "python3 /home/pi/SDR-SpectrumMonitoring-Sensor/run_sensor.py --json" \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --out_key psd_dbm
```
