# VAE_implementation

Documentacion tecnica de la version estable sin Kalman.

## 1. Objetivo

Comprimir PSD de 1024 bins con VAE para transporte eficiente por UDP.

## 2. Arquitectura

- Encoder TFLite INT8 en edge
- Decoder Keras en servidor
- Transporte UDP con formato `KLP1`

## 3. Scripts clave

- `00_get_data.ps1`: descarga dataset
- `01_preprocess.py`: genera dataset normalizado + splits
- `02_train.py`: entrenamiento VAE
- `03_eval.py`: evaluacion y plots
- `04_export_tflite.py`: export encoder INT8
- `05_entropy_stats.py`: analisis entropia latente
- `06_codec_benchmark.py`: benchmark de codec
- `07_pack_unpack.py`: wire format y pack/unpack
- `08_udp_*`: pruebas basicas
- `09_udp_*`: pruebas con comparacion contra original
- `10_udp_sender_prod.py`: sender produccion
- `10_udp_receiver_prod.py`: receiver produccion
- `11_edge_hackrf_psd_zmq_to_udp.py`: bridge SDR (ZMQ PSD -> UDP)

## 4. Pipeline

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```powershell
.\VAE_implementation\scripts\00_get_data.ps1
python VAE_implementation/scripts/01_preprocess.py --config VAE_implementation/configs/vae_default.yaml
python VAE_implementation/scripts/02_train.py --config VAE_implementation/configs/vae_default.yaml
python VAE_implementation/scripts/04_export_tflite.py --config VAE_implementation/configs/vae_default.yaml --use_global_best
```

## 5. Produccion

## 5.1 Simulacion local

Receiver:

```powershell
python VAE_implementation/scripts/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --save_every_packets 1 --idle_stop_s 3 --invert_norm_to_original
```

Sender:

```powershell
python VAE_implementation/scripts/10_udp_sender_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --source dataset --split test --block_len 30 --n_blocks 3 --zlib_level 1
```

Comparacion (original vs reconstruccion):

```powershell
python VAE_implementation/scripts/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --save_every_packets 1 --idle_stop_s 3 --invert_norm_to_original --compare_split test --plot_every_packets 1
```

## 5.2 SDR real

Servidor:

```powershell
python VAE_implementation/scripts/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 0.0.0.0 --port 5005 --save_every_packets 10 --idle_stop_s 10 --invert_norm_to_original
```

Edge:

```bash
python VAE_implementation/scripts/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --use_global_best \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --dest_ip <IP_SERVIDOR> --port 5005 \
  --block_len 30 --zlib_level 1
```

## 6. Datos y normalizacion

- El bridge `11` espera PSD en escala original (dB/dBm-like).
- Normaliza usando `gmin/gmax` del dataset procesado.
- Si ya llega normalizado, usar `--already_normalized`.

## 7. Wire protocol

- Header `KLP1` + payload zlib
- `07_pack_unpack.py` define parseo y validaciones

## 8. Troubleshooting

- Falta TFLite: correr `04_export_tflite.py`
- Sin paquetes: revisar IP/puerto/firewall
- Bridge sin PSD: revisar JSON o usar `--psd_key`
- Si ya existe modulo PSD, validar:
  - endpoint ZMQ exacto (`ipc:///tmp/ane_psd.ipc`)
  - llave PSD del JSON (o `--psd_key`)
  - escala de PSD (original o `--already_normalized`)

