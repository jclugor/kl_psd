# Raspberry Realtime Bundle

Bundle para:
- streaming VAE en tiempo real

## Archivos incluidos

- `VAE_implementation/scripts/codec/07_pack_unpack.py`
- `VAE_implementation/scripts/prod/11_edge_hackrf_psd_zmq_to_udp.py`
- `VAE_implementation/scripts/prod/12_acq_sensor_to_zmq.py`
- `VAE_implementation/scripts/prod/13_rf_engine_controller_to_zmq.py`
- `VAE_implementation/configs/vae_default.yaml`
- `VAE_implementation/models/GLOBAL_BEST/encoder_mu_int8.tflite`
- `data/processed/psd_1024/dataset_psd_1024_norm.npz`
  
Nota: La adquisicion de dataset es externa a este repositorio.

## Instalacion en Raspberry

```bash
python3 -m venv .venv_edge
source .venv_edge/bin/activate
pip install --upgrade pip
pip install numpy pyzmq pyyaml
```

Si `tflite-runtime` no existe para tu plataforma, usa `tensorflow`.

## Streaming VAE al servidor

Si ademas quieres enviar al PC por UDP:

### Terminal A

```bash
python VAE_implementation/scripts/prod/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --use_global_best \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --dest_ip 192.168.0.112 --port 5005 \
  --block_len 30 --zlib_level 1 --log_every_packets 1
```

### Terminal B

El productor de PSD es externo. Opciones:

1) Adaptador callable/script:
```bash
python VAE_implementation/scripts/prod/12_acq_sensor_to_zmq.py \
  --mode callable \
  --sensor_repo_path /home/pi/SDR-SpectrumMonitoring-Sensor \
  --callable "mi_modulo.mi_fuente:get_next_psd" \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --out_key psd_dbm
```

2) Controller para `rf_engine`:
```bash
python VAE_implementation/scripts/prod/13_rf_engine_controller_to_zmq.py \
  --rf_ipc "ipc:///tmp/rf_engine" \
  --out_ipc "ipc:///tmp/ane_psd.ipc" \
  --in_key Pxx \
  --cmd_json '{"center_freq_hz":100100000,"sample_rate_hz":2000000,"rbw_hz":10000,"window":"hann","overlap":0.5,"lna_gain":16,"vga_gain":20,"antenna_amp":false,"antenna_port":0}' \
  --send_cmd_every_s 1.0
```
