# VAE_implementation

Documentacion tecnica de la implementacion VAE PSD (version estable).

## 1. Objetivo

Comprimir y transportar PSD (1024 bins) por UDP minimizando ancho de banda y manteniendo reconstruccion usable.

## 2. Arquitectura

- Encoder TFLite INT8 en edge.
- Decoder Keras en servidor.
- Protocolo UDP binario (`KLP1`) definido en `07_pack_unpack.py`.
- Flujo temporal: cada bloque transmite `mu_int8` (mu0+deltas).

## 3. Scripts clave

Base:
- `01_preprocess.py`, `02_train.py`, `03_eval.py`, `04_export_tflite.py`
- `07_pack_unpack.py`

Produccion:
- `10_udp_sender_prod.py`
- `10_udp_receiver_prod.py`
- `11_edge_hackrf_psd_zmq_to_udp.py`
- `12_acq_sensor_to_zmq.py` (adaptador de adquisicion externa)

## 4. Dependencias

- `numpy`, `tensorflow/keras`, `pyyaml`, `matplotlib`
- `pyzmq` (bridge SDR)
- `tflite-runtime` opcional en edge

## 5. Flujo de ejecucion

1. Exportar TFLite:
```powershell
python VAE_implementation/scripts/04_export_tflite.py --config VAE_implementation/configs/vae_default.yaml --use_global_best
```

2. Test local:
```powershell
python VAE_implementation/scripts/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --save_every_packets 1 --idle_stop_s 3 --invert_norm_to_original --compare_split test --plot_every_packets 1
python VAE_implementation/scripts/10_udp_sender_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --source dataset --split test --block_len 30 --n_blocks 10 --zlib_level 1
```

3. Produccion SDR:
```bash
python VAE_implementation/scripts/12_acq_sensor_to_zmq.py \
  --mode callable \
  --sensor_repo_path /home/pi/SDR-SpectrumMonitoring-Sensor \
  --callable "mi_modulo.mi_fuente:get_next_psd" \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --out_key psd_dbm

python VAE_implementation/scripts/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --use_global_best \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --dest_ip <IP_SERVIDOR> --port 5005 \
  --block_len 30 --zlib_level 1
```

## 6. Consideraciones tecnicas

- Normalizacion esperada: `global_minmax` con `gmin/gmax` del dataset procesado.
- Si PSD ya viene normalizada [0,1], usar `--already_normalized` en script `11`.
- Ajustar `block_len` y `zlib_level` para balance latencia/compresion.

## 7. Salidas y metricas

- `models/GLOBAL_BEST/udp_prod/`

Archivos comunes:
- `udp_prod_metrics.json`
- `waterfall_recon.png`
- `psd_last.png`

Modo comparacion:
- `waterfall_orig.png`
- `overlay_pktXXXXXX.png`
- `orig_last_psd.npy`
- `recon_last_psd.npy`

## 8. Troubleshooting

- Sin trafico UDP: revisar IP/puerto/firewall.
- Sin PSD en bridge: validar endpoint ZMQ y llave JSON (`--psd_key`).
- Error TFLite en Windows: usar `tf.lite.Interpreter` fallback.
