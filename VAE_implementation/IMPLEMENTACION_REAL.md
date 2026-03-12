# IMPLEMENTACION_REAL.md

Guia operativa para despliegue real.

## 1. Objetivo

1. Edge recibe PSD del pipeline SDR.
2. Edge codifica latente y envia UDP.
3. Servidor reconstruye PSD y guarda metricas/plots.

## 2. Scripts principales

- `VAE_implementation/scripts/prod/10_udp_sender_prod.py`
- `VAE_implementation/scripts/prod/10_udp_receiver_prod.py`
- `VAE_implementation/scripts/prod/11_edge_hackrf_psd_zmq_to_udp.py`
- `VAE_implementation/scripts/prod/12_acq_sensor_to_zmq.py` (adaptador para repo externo)
- `VAE_implementation/scripts/prod/13_rf_engine_controller_to_zmq.py` (controller propio para rf_engine)

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
python VAE_implementation/scripts/prod/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --save_every_packets 1 --idle_stop_s 3 --invert_norm_to_original --compare_split test --plot_every_packets 1
```

Terminal B (sender):
```powershell
python VAE_implementation/scripts/prod/10_udp_sender_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --source dataset --split test --block_len 30 --n_blocks 10 --zlib_level 1
```

Salida:
- `VAE_implementation/models/GLOBAL_BEST/udp_prod/`

## 5. Despliegue real con SDR

Servidor:
```powershell
python VAE_implementation/scripts/prod/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 0.0.0.0 --port 5005 --save_every_packets 10 --idle_stop_s 10 --invert_norm_to_original
```

Edge bridge:
```bash
python VAE_implementation/scripts/prod/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --use_global_best \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --dest_ip <IP_SERVIDOR> --port 5005 \
  --block_len 30 --zlib_level 1
```

## 6. Adquisicion externa

La adquisicion de PSD se realiza fuera de este repositorio. El pipeline externo debe
publicar PSD en ZMQ/JSON al IPC `ipc:///tmp/ane_psd.ipc` para que el bridge pueda consumirlo.

### 6.1 Adaptador por callable/script

```bash
python VAE_implementation/scripts/prod/12_acq_sensor_to_zmq.py \
  --mode callable \
  --sensor_repo_path /home/pi/SDR-SpectrumMonitoring-Sensor \
  --callable "mi_modulo.mi_fuente:get_next_psd" \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --out_key psd_dbm
```

### 6.2 Controller propio para rf_engine

```bash
python VAE_implementation/scripts/prod/13_rf_engine_controller_to_zmq.py \
  --rf_ipc "ipc:///tmp/rf_engine" \
  --out_ipc "ipc:///tmp/ane_psd.ipc" \
  --in_key Pxx \
  --cmd_json '{"center_freq_hz":100100000,"sample_rate_hz":2000000,"rbw_hz":10000,"window":"hann","overlap":0.5,"lna_gain":16,"vga_gain":20,"antenna_amp":false,"antenna_port":0}' \
  --send_cmd_every_s 1.0
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

## 10. Troubleshooting

- `Missing encoder_mu_int8.tflite`: ejecutar `04_export_tflite.py`.
- `ImportError tensorflow.lite.Interpreter`: actualizar entorno o usar `tflite-runtime`.
- No llegan paquetes: revisar IP, puerto y firewall.
- Bridge sin PSD: verificar endpoint ZMQ y `--psd_key`.
