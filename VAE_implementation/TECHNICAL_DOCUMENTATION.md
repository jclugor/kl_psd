# VAE_implementation

Technical documentation for the stable VAE PSD implementation.

## 1. Goal

Compress and transport PSD frames (1024 bins) over UDP while minimizing
bandwidth and keeping the reconstruction useful.

## 2. Architecture

- INT8 TFLite encoder on the edge device
- Keras decoder on the server
- Binary UDP protocol (`KLP1`) defined in `scripts/codec/07_pack_unpack.py`
- Temporal stream format in which each block transmits `mu_int8` as `mu0 + deltas`

## 3. Key scripts

Base workflow:
- `scripts/training/01_preprocess.py`
- `scripts/training/02_train.py`
- `scripts/training/03_eval.py`
- `scripts/training/04_export_tflite.py`
- `scripts/codec/07_pack_unpack.py`

Production workflow:
- `scripts/prod/10_udp_sender_prod.py`
- `scripts/prod/10_udp_receiver_prod.py`
- `scripts/prod/11_edge_hackrf_psd_zmq_to_udp.py`
- `scripts/prod/12_acq_sensor_to_zmq.py` for external-acquisition adaptation
- `scripts/prod/13_rf_engine_controller_to_zmq.py` as a custom controller for `rf_engine`

## 4. Dependencies

- `numpy`, `tensorflow/keras`, `pyyaml`, `matplotlib`
- `pyzmq` for the SDR bridge
- optional `tflite-runtime` on the edge device

## 5. Execution flow

1. Export TFLite artifacts:

```powershell
python VAE_implementation/scripts/training/04_export_tflite.py --config VAE_implementation/configs/vae_default.yaml --use_global_best
```

2. Run a local test:

```powershell
python VAE_implementation/scripts/prod/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --save_every_packets 1 --idle_stop_s 3 --invert_norm_to_original --compare_split test --plot_every_packets 1
python VAE_implementation/scripts/prod/10_udp_sender_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --source dataset --split test --block_len 30 --n_blocks 10 --zlib_level 1
```

3. Run the SDR production path:

```bash
python VAE_implementation/scripts/prod/12_acq_sensor_to_zmq.py \
  --mode callable \
  --sensor_repo_path /home/pi/SDR-SpectrumMonitoring-Sensor \
  --callable "my_module.my_source:get_next_psd" \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --out_key psd_dbm

python VAE_implementation/scripts/prod/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --use_global_best \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --dest_ip <SERVER_IP> --port 5005 \
  --block_len 30 --zlib_level 1
```

Alternative without modifying the SDR repository:

```bash
python VAE_implementation/scripts/prod/13_rf_engine_controller_to_zmq.py \
  --rf_ipc "ipc:///tmp/rf_engine" \
  --out_ipc "ipc:///tmp/ane_psd.ipc" \
  --in_key Pxx \
  --cmd_json '{"center_freq_hz":100100000,"sample_rate_hz":2000000,"rbw_hz":10000,"window":"hann","overlap":0.5,"lna_gain":16,"vga_gain":20,"antenna_amp":false,"antenna_port":0}' \
  --send_cmd_every_s 1.0
```

## 6. Technical considerations

- Expected normalization: `global_minmax` using the `gmin/gmax` values from the processed dataset
- If the PSD stream is already normalized to `[0, 1]`, use `--already_normalized` with script `11`
- Tune `block_len` and `zlib_level` to balance latency and compression ratio
- Dataset acquisition happens outside this repository

## 7. Outputs and metrics

- `models/GLOBAL_BEST/udp_prod/`

Common files:
- `udp_prod_metrics.json`
- `waterfall_recon.png`
- `psd_last.png`

Comparison mode:
- `waterfall_orig.png`
- `overlay_pktXXXXXX.png`
- `orig_last_psd.npy`
- `recon_last_psd.npy`

## 8. Troubleshooting

- No UDP traffic: check the IP, port, and firewall
- No PSD frames in the bridge: validate the ZMQ endpoint and JSON key (`--psd_key`)
- TFLite error on Windows: use the `tf.lite.Interpreter` fallback
