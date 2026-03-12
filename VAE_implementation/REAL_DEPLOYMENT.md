# REAL_DEPLOYMENT.md

Operational guide for real deployment.

## 1. Goal

1. The edge device receives PSD frames from the SDR pipeline.
2. The edge device encodes the latent representation and sends UDP packets.
3. The server reconstructs the PSD frames and stores metrics and plots.

## 2. Main scripts

- `VAE_implementation/scripts/prod/10_udp_sender_prod.py`
- `VAE_implementation/scripts/prod/10_udp_receiver_prod.py`
- `VAE_implementation/scripts/prod/11_edge_hackrf_psd_zmq_to_udp.py`
- `VAE_implementation/scripts/prod/12_acq_sensor_to_zmq.py` as an adapter for an external repository
- `VAE_implementation/scripts/prod/13_rf_engine_controller_to_zmq.py` as a custom controller for `rf_engine`

## 3. Prerequisites

```powershell
pip install -r requirements.txt
pip install pyzmq
```

Minimum artifacts in `VAE_implementation/models/GLOBAL_BEST/`:
- `encoder_mu_int8.tflite`
- `dec_best.weights.h5`

## 4. Quick local test

Terminal A (receiver):

```powershell
python VAE_implementation/scripts/prod/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 127.0.0.1 --port 5005 --save_every_packets 1 --idle_stop_s 3 --invert_norm_to_original --compare_split test --plot_every_packets 1
```

Terminal B (sender):

```powershell
python VAE_implementation/scripts/prod/10_udp_sender_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --source dataset --split test --block_len 30 --n_blocks 10 --zlib_level 1
```

Output directory:
- `VAE_implementation/models/GLOBAL_BEST/udp_prod/`

## 5. Real deployment with SDR

Server:

```powershell
python VAE_implementation/scripts/prod/10_udp_receiver_prod.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --bind_ip 0.0.0.0 --port 5005 --save_every_packets 10 --idle_stop_s 10 --invert_norm_to_original
```

Edge bridge:

```bash
python VAE_implementation/scripts/prod/11_edge_hackrf_psd_zmq_to_udp.py \
  --config VAE_implementation/configs/vae_default.yaml \
  --use_global_best \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --dest_ip <SERVER_IP> --port 5005 \
  --block_len 30 --zlib_level 1
```

## 6. External acquisition

PSD acquisition is external to this repository. The external pipeline must
publish PSD frames over ZMQ/JSON to `ipc:///tmp/ane_psd.ipc` so the bridge can
consume them.

### 6.1 Callable/script adapter

```bash
python VAE_implementation/scripts/prod/12_acq_sensor_to_zmq.py \
  --mode callable \
  --sensor_repo_path /home/pi/SDR-SpectrumMonitoring-Sensor \
  --callable "my_module.my_source:get_next_psd" \
  --ipc "ipc:///tmp/ane_psd.ipc" \
  --out_key psd_dbm
```

### 6.2 Custom controller for `rf_engine`

```bash
python VAE_implementation/scripts/prod/13_rf_engine_controller_to_zmq.py \
  --rf_ipc "ipc:///tmp/rf_engine" \
  --out_ipc "ipc:///tmp/ane_psd.ipc" \
  --in_key Pxx \
  --cmd_json '{"center_freq_hz":100100000,"sample_rate_hz":2000000,"rbw_hz":10000,"window":"hann","overlap":0.5,"lna_gain":16,"vga_gain":20,"antenna_amp":false,"antenna_port":0}' \
  --send_cmd_every_s 1.0
```

## 7. ZMQ PSD format

Example JSON:

```json
{"psd_dbm": [-91.2, -90.8, -89.7]}
```

Auto-detected keys:
- `psd_dbm`, `psd`, `p_out`, `power_dbm`, `p_dbm`, `spectrum_dbm`, `bins_dbm`, `pxx`

If you use a different key:

```bash
--psd_key <key>
```

## 8. Validation

Check:
- `udp_prod_metrics.json`
- `waterfall_recon.png`
- `waterfall_orig.png` in compare mode
- `recon_last_psd.npy` and `orig_last_psd.npy` in compare mode

## 9. Troubleshooting

- `Missing encoder_mu_int8.tflite`: run `04_export_tflite.py`
- `ImportError tensorflow.lite.Interpreter`: update the environment or use `tflite-runtime`
- Packets do not arrive: check the IP, port, and firewall
- Bridge without PSD frames: verify the ZMQ endpoint and `--psd_key`
