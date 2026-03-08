# IMPLEMENTACION_REAL.md

Guia operativa para despliegue real.

## 1. Objetivo

1. Edge recibe PSD del pipeline SDR.
2. Edge codifica latente y envia UDP.
3. Servidor reconstruye PSD y guarda metricas/plots.

## 2. Scripts principales

- `VAE_implementation/scripts/10_udp_sender_prod.py`
- `VAE_implementation/scripts/10_udp_receiver_prod.py`
- `VAE_implementation/scripts/11_edge_hackrf_psd_zmq_to_udp.py`
- `VAE_implementation/scripts/12_acq_sensor_to_zmq.py` (adaptador para repo externo)

## 3. Prerrequisitos

```powershell
pip install -r requirements.txt
pip install pyzmq
```

Artefactos minimos en `VAE_implementation/models/GLOBAL_BEST/`:
- `encoder_mu_int8.tflite`
- `dec_best.weights.h5`

## 4. Test local rapido

Terminal A (receiver):
```powershell
python VAE_implementation/scripts/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --save_every_packets 1 --idle_stop_s 3 --invert_norm_to_original --compare_split test --plot_every_packets 1
```

Terminal B (sender):
```powershell
python VAE_implementation/scripts/10_udp_sender_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --source dataset --split test --block_len 30 --n_blocks 10 --zlib_level 1
```

Salida:
- `VAE_implementation/models/GLOBAL_BEST/udp_prod/`

## 5. Despliegue real con SDR

Servidor:
```powershell
python VAE_implementation/scripts/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 0.0.0.0 --port 5005 --save_every_packets 10 --idle_stop_s 10 --invert_norm_to_original
```

Edge bridge:
```bash
python VAE_implementation/scripts/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --use_global_best \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --dest_ip <IP_SERVIDOR> --port 5005 \
  --block_len 30 --zlib_level 1
```

## 6. Integracion con repo de adquisicion externo

Para conectar `SDR-SpectrumMonitoring-Sensor` al IPC del bridge:

Modo callable:
```bash
python VAE_implementation/scripts/12_acq_sensor_to_zmq.py \
  --mode callable \
  --sensor_repo_path /home/pi/SDR-SpectrumMonitoring-Sensor \
  --callable "mi_modulo.mi_fuente:get_next_psd" \
  --callable_kwargs_json "{}" \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --out_key psd_dbm
```

Modo script (stdout JSON):
```bash
python VAE_implementation/scripts/12_acq_sensor_to_zmq.py \
  --mode script \
  --script_cmd "python3 /home/pi/SDR-SpectrumMonitoring-Sensor/run_sensor.py --json" \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --out_key psd_dbm
```

## 7. Formato de PSD por ZMQ

JSON ejemplo:
```json
{"psd_dbm": [-91.2, -90.8, -89.7]}
```

Llaves auto-detectadas:
- `psd_dbm`, `psd`, `p_out`, `power_dbm`, `p_dbm`, `spectrum_dbm`, `bins_dbm`, `pxx`

Si usa otra llave:
```bash
--psd_key <llave>
```

## 8. Validacion

Revisar:
- `udp_prod_metrics.json`
- `waterfall_recon.png`
- `waterfall_orig.png` (si compare mode)
- `recon_last_psd.npy` y `orig_last_psd.npy` (si compare mode)

## 9. Troubleshooting

- `Missing encoder_mu_int8.tflite`: ejecutar `04_export_tflite.py`.
- `ImportError tensorflow.lite.Interpreter`: actualizar entorno o usar `tflite-runtime`.
- No llegan paquetes: revisar IP, puerto y firewall.
- Bridge sin PSD: verificar endpoint ZMQ y `--psd_key`.
