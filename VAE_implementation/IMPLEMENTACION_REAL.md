# IMPLEMENTACION_REAL.md

Guia paso a paso para desplegar la version actual en entorno real (sin Kalman).

## 1. Objetivo

Desplegar un flujo estable:

1. Edge recibe PSD de tu pipeline SDR.
2. Edge codifica con encoder TFLite INT8 y envia UDP.
3. Servidor reconstruye PSD con decoder y guarda metricas/plots.

## 2. Ruta oficial (MAINLINE)

- Receiver servidor: `VAE_implementation/scripts/10_udp_receiver_prod.py`
- Sender edge simulado/dataset: `VAE_implementation/scripts/10_udp_sender_prod.py`
- Bridge SDR real: `VAE_implementation/scripts/11_edge_hackrf_psd_zmq_to_udp.py`

## 3. Prerrequisitos

## 3.1 Servidor (PC)

- Python 3.10+
- `pip install -r requirements.txt`
- Artefactos en `VAE_implementation/models/GLOBAL_BEST/`:
  - `encoder_mu_int8.tflite`
  - `dec_best.weights.h5`

## 3.2 Edge (Raspberry/PC)

- Python 3.10+
- `pip install -r requirements.txt`
- `pip install pyzmq`
- Proceso SDR que publique PSD por ZMQ PAIR

## 3.3 Red

- UDP edge -> servidor habilitado
- Puerto abierto (ej. `5005`)

## 4. Test local rapido

Terminal A (receiver):

```powershell
python VAE_implementation/scripts/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --save_every_packets 1 --idle_stop_s 3 --invert_norm_to_original
```

Terminal B (sender dataset):

```powershell
python VAE_implementation/scripts/10_udp_sender_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --source dataset --split test --block_len 30 --n_blocks 3 --zlib_level 1
```

Modo comparacion (guardar original + reconstruccion):

```powershell
python VAE_implementation/scripts/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --save_every_packets 1 --idle_stop_s 3 --invert_norm_to_original --compare_split test --plot_every_packets 1
```

Archivos extra en modo comparacion:

- `waterfall_orig.png`
- `waterfall_recon.png`
- `overlay_pktXXXXXX.png`
- `orig_last_psd.npy`
- `recon_last_psd.npy`

Salidas en:

- `VAE_implementation/models/GLOBAL_BEST/udp_prod/`

## 5. Despliegue con SDR real

## 5.1 Servidor

```powershell
python VAE_implementation/scripts/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 0.0.0.0 --port 5005 --save_every_packets 10 --idle_stop_s 10 --invert_norm_to_original
```

## 5.2 Edge

Asegura que tu proceso SDR publique JSON PSD por ZMQ en `ipc:///tmp/ane_psd.ipc`.

Luego ejecuta:

```bash
python VAE_implementation/scripts/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --use_global_best \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --dest_ip <IP_SERVIDOR> --port 5005 \
  --block_len 30 --zlib_level 1
```

Si tu JSON usa una llave específica para PSD:

```bash
--psd_key p_out
```

Si la PSD ya viene normalizada [0,1]:

```bash
--already_normalized
```

## 6. Formato esperado de entrada ZMQ

Ejemplo de mensaje JSON valido:

```json
{"psd_dbm": [-91.2, -90.8, -89.7, ...]}
```

Llaves aceptadas automaticamente:

- `psd_dbm`, `psd`, `p_out`, `power_dbm`, `p_dbm`, `spectrum_dbm`, `bins_dbm`, `pxx`

## 7. Verificacion operativa

- Revisar `udp_prod_metrics.json`:
  - `packets`, `frames`, `avg_bytes_per_frame`, `frames_per_second`, `seq_gaps`, `out_of_order`
- Revisar plots:
  - `waterfall_recon.png`
  - `psd_last.png`

Checklist si el modulo PSD ya existe:

- Endpoint ZMQ coincide (`ipc:///tmp/ane_psd.ipc`).
- JSON contiene PSD en una llave soportada o se usa `--psd_key`.
- Escala de PSD correcta (si ya normalizada, usar `--already_normalized`).
- UDP llega al puerto del servidor (sin bloqueo de firewall).

## 8. Troubleshooting

- `Missing encoder_mu_int8.tflite`: ejecutar `04_export_tflite.py`.
- Sin paquetes en servidor: revisar IP, puerto y firewall.
- Bridge no detecta PSD: usar `--psd_key`.
- Error de normalizacion en bridge: verificar `gmin/gmax` en `dataset_psd_1024_norm.npz`.

