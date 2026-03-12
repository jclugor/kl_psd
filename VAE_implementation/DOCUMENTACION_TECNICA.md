# VAE_implementation

Documentacion tecnica de la implementacion VAE PSD (version estable).

## 1. Objetivo

Comprimir y transportar PSD (1024 bins) por UDP minimizando ancho de banda y manteniendo reconstruccion usable.

## 2. Arquitectura

- Encoder TFLite INT8 en edge.
- Decoder Keras en servidor.
- Protocolo UDP binario (`KLP1`) definido en `scripts/codec/07_pack_unpack.py`.
- Flujo temporal: cada bloque transmite `mu_int8` (mu0+deltas).

## 3. Scripts clave

Base:
- `scripts/training/01_preprocess.py`, `scripts/training/02_train.py`, `scripts/training/03_eval.py`, `scripts/training/04_export_tflite.py`
- `scripts/codec/07_pack_unpack.py`

Produccion:
- `scripts/prod/10_udp_sender_prod.py`
- `scripts/prod/10_udp_receiver_prod.py`
- `scripts/prod/11_edge_hackrf_psd_zmq_to_udp.py`
- `scripts/prod/12_acq_sensor_to_zmq.py` (adaptador de adquisicion externa)
- `scripts/prod/13_rf_engine_controller_to_zmq.py` (controller propio para rf_engine)

## 4. Dependencias

- `numpy`, `tensorflow/keras`, `pyyaml`, `matplotlib`
- `pyzmq` (bridge SDR)
- `tflite-runtime` opcional en edge

## 5. Flujo de ejecucion

1. Exportar TFLite:
```powershell
python VAE_implementation/scripts/training/04_export_tflite.py --config VAE_implementation/configs/vae_default.yaml --use_global_best
```

2. Test local:
```powershell
python VAE_implementation/scripts/prod/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --save_every_packets 1 --idle_stop_s 3 --invert_norm_to_original --compare_split test --plot_every_packets 1
python VAE_implementation/scripts/prod/10_udp_sender_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --source dataset --split test --block_len 30 --n_blocks 10 --zlib_level 1
```

3. Produccion SDR:
```bash
python VAE_implementation/scripts/prod/12_acq_sensor_to_zmq.py \
  --mode callable \
  --sensor_repo_path /home/pi/SDR-SpectrumMonitoring-Sensor \
  --callable "mi_modulo.mi_fuente:get_next_psd" \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --out_key psd_dbm

python VAE_implementation/scripts/prod/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --use_global_best \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --dest_ip <IP_SERVIDOR> --port 5005 \
  --block_len 30 --zlib_level 1
```

Alternativa sin modificar repo SDR:
```bash
python VAE_implementation/scripts/prod/13_rf_engine_controller_to_zmq.py \
  --rf_ipc "ipc:///tmp/rf_engine" \
  --out_ipc "ipc:///tmp/ane_psd.ipc" \
  --in_key Pxx \
  --cmd_json '{"center_freq_hz":100100000,"sample_rate_hz":2000000,"rbw_hz":10000,"window":"hann","overlap":0.5,"lna_gain":16,"vga_gain":20,"antenna_amp":false,"antenna_port":0}' \
  --send_cmd_every_s 1.0
```

## 6. Consideraciones tecnicas

- Normalizacion esperada: `global_minmax` con `gmin/gmax` del dataset procesado.
- Si PSD ya viene normalizada [0,1], usar `--already_normalized` en script `11`.
- Ajustar `block_len` y `zlib_level` para balance latencia/compresion.
- La adquisicion de dataset se realiza fuera de este repositorio.

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
