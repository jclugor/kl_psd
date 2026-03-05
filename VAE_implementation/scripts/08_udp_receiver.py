#!/usr/bin/env python3
"""
08_udp_receiver.py — UDP receiver (server side).

Receives packets produced by 08_udp_sender.py / 07_pack_unpack.py:
- UDP packet = header + zlib(payload)
- payload decodes to mu_block int8 (L,32)

Optionally reconstructs PSD using decoder weights on the server.

Run (localhost test):
  Terminal 1:
    python VAE_implementation/scripts/08_udp_receiver.py --bind_ip 127.0.0.1 --port 5005 --use_global_best
  Terminal 2:
    python VAE_implementation/scripts/08_udp_sender.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --split test --block_len 30 --n_blocks 50 --zlib_level 1
"""

import argparse
import importlib.util
import socket
import time
from pathlib import Path
import json

import numpy as np
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_pack_module() -> object:
    """Load 07_pack_unpack.py via importlib (filename starts with a digit)."""
    root = repo_root()
    mod_path = root / "VAE_implementation" / "scripts" / "07_pack_unpack.py"
    spec = importlib.util.spec_from_file_location("packmod", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _has_conv1d_transpose() -> bool:
    return hasattr(layers, "Conv1DTranspose")


def build_decoder(latent_dim: int = 32) -> keras.Model:
    z_in = keras.Input(shape=(latent_dim,), name="z_in")
    x = layers.Dense(256 * 32, name="dec_dense")(z_in)
    x = layers.Reshape((256, 32), name="dec_reshape")(x)

    if _has_conv1d_transpose():
        Conv1DTranspose = layers.Conv1DTranspose
        x = Conv1DTranspose(32, kernel_size=3, strides=2, padding="same", name="dec_deconv1")(x)
        x = layers.LeakyReLU(alpha=0.2, name="dec_lrelu1")(x)
        x = Conv1DTranspose(16, kernel_size=5, strides=2, padding="same", name="dec_deconv2")(x)
        x = layers.LeakyReLU(alpha=0.2, name="dec_lrelu2")(x)
    else:
        x = layers.UpSampling1D(size=2, name="dec_ups1")(x)
        x = layers.Conv1D(32, kernel_size=3, padding="same", name="dec_conv1")(x)
        x = layers.LeakyReLU(alpha=0.2, name="dec_lrelu1")(x)
        x = layers.UpSampling1D(size=2, name="dec_ups2")(x)
        x = layers.Conv1D(16, kernel_size=5, padding="same", name="dec_conv2")(x)
        x = layers.LeakyReLU(alpha=0.2, name="dec_lrelu2")(x)

    x_hat = layers.Conv1D(1, kernel_size=1, activation="sigmoid", name="x_hat")(x)
    return keras.Model(z_in, x_hat, name="decoder")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="VAE_implementation/configs/vae_default.yaml")
    ap.add_argument("--bind_ip", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--use_global_best", action="store_true")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--reconstruct", action="store_true", help="If set, reconstruct PSD with decoder on server.")
    ap.add_argument("--save_last", action="store_true", help="Save last reconstructed PSD to .npy")
    ap.add_argument("--max_packets", type=int, default=0, help="Stop after N packets (0=run forever).")
    args = ap.parse_args()

    root = repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())

    models_dir = root / cfg["paths"]["models_dir"]
    if args.use_global_best:
        model_dir = models_dir / "GLOBAL_BEST"
        tag = "GLOBAL_BEST"
    else:
        run = args.run_name or cfg.get("train", {}).get("run_name", "run_current")
        model_dir = models_dir / "runs" / run
        tag = f"RUN:{run}"

    # Load pack/unpack module
    packmod = load_pack_module()

    # Optional decoder
    decoder = None
    if args.reconstruct:
        dec_w = model_dir / "dec_best.weights.h5"
        if not dec_w.exists():
            raise FileNotFoundError(f"Decoder weights not found: {dec_w}")
        decoder = build_decoder(latent_dim=32)
        _ = decoder(tf.zeros((1, 32), dtype=tf.float32), training=False)
        decoder.load_weights(dec_w)
        print(f"[RECV] Loaded decoder weights: {dec_w}")

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, args.port))
    sock.settimeout(1.0)

    print(f"[RECV] Listening on udp://{args.bind_ip}:{args.port} | {tag}")
    print("[RECV] Waiting for packets...")

    pkt_count = 0
    frame_count = 0
    byte_count = 0
    t_start = time.perf_counter()
    t_last = t_start
    last_xhat = None

    while True:
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            now = time.perf_counter()
            if now - t_last >= 2.0 and pkt_count > 0:
                dt = now - t_start
                print(f"[RECV] pkts={pkt_count} frames={frame_count} bytes={byte_count} "
                      f"| pkts/s={pkt_count/dt:.1f} frames/s={frame_count/dt:.1f} "
                      f"| KB/s={byte_count/dt/1024:.1f}")
                t_last = now
            continue

        pkt_count += 1
        byte_count += len(data)

        hdr, mu_block = packmod.decode_packet_to_mu(data)  # mu_block (L,32) int8
        L = int(mu_block.shape[0])
        frame_count += L

        if decoder is not None:
            # Placeholder dequant (Stage 9 will fix using exact tflite out_scale/out_zp)
            mu_f = mu_block.astype(np.float32) / 128.0
            xhat = decoder(mu_f, training=False).numpy()  # (L,1024,1)
            last_xhat = xhat

        if args.max_packets and pkt_count >= args.max_packets:
            break

    dt = time.perf_counter() - t_start
    print(f"[RECV] DONE pkts={pkt_count} frames={frame_count} bytes={byte_count} "
          f"| frames/s={frame_count/dt:.1f} | avg bytes/frame={byte_count/max(1,frame_count):.3f}")

    if args.save_last and last_xhat is not None:
        out_path = model_dir / "last_recon_psd.npy"
        np.save(out_path, last_xhat)
        print("[RECV] Saved last recon to:", out_path)


if __name__ == "__main__":
    main()