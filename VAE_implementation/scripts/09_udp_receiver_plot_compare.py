#!/usr/bin/env python3
"""
09_udp_receiver_plot_compare.py (improved) — UDP receiver with:

- Correct mu dequantization using encoder_mu_int8.tflite output quant params (scale/zp)
- Reconstruction using decoder weights
- Comparison vs original PSD (needs sender v9 indexed meta)
- Plots: overlay per packet + waterfalls (ring buffer, no RAM explosion)
- Robust networking: no crash on timeout, idle-stop optional
- Seq gap logging (packet loss / out-of-order)

Outputs:
  <model_dir>/udp_eval/
    overlay_seqXXXXXX.png
    waterfall_orig.png
    waterfall_recon.png
    udp_metrics.json
"""

import argparse
import importlib.util
import socket
import time
from pathlib import Path
import json

import numpy as np
import yaml
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_pack_module() -> object:
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


def load_npz(processed_dir: Path) -> np.ndarray:
    npz_path = processed_dir / "dataset_psd_1024_norm.npz"
    if not npz_path.exists():
        candidates = sorted(processed_dir.glob("*.npz"), key=lambda p: p.stat().st_size, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No .npz found in {processed_dir}")
        npz_path = candidates[0]
    d = np.load(npz_path, allow_pickle=True)
    return d["X"].astype(np.float32)


def recon_loss_sum_mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.mean(np.sum((x - x_hat) ** 2, axis=1)))


def topk_peak_mse(x: np.ndarray, x_hat: np.ndarray, k: int = 32) -> float:
    idx = np.argpartition(x, -k, axis=1)[:, -k:]
    rows = np.arange(x.shape[0])[:, None]
    return float(np.mean((x[rows, idx] - x_hat[rows, idx]) ** 2))


def peak_bias(x: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.mean(np.max(x_hat, axis=1) - np.max(x, axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bind_ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--use_global_best", action="store_true")
    ap.add_argument("--run_name", default=None)

    ap.add_argument("--max_packets", type=int, default=0, help="0 = run forever until idle-stop or Ctrl+C")
    ap.add_argument("--idle_stop_s", type=float, default=0.0, help="Stop if no packets arrive for this many seconds (0=never).")
    ap.add_argument("--socket_timeout_s", type=float, default=1.0)

    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--plot_every", type=int, default=1, help="save overlay every N packets (1=all)")
    ap.add_argument("--waterfall_max_frames", type=int, default=300)
    args = ap.parse_args()

    root = repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())

    processed_dir = root / cfg["paths"]["processed_dir"]
    models_dir = root / cfg["paths"]["models_dir"]

    if args.use_global_best:
        model_dir = models_dir / "GLOBAL_BEST"
        tag = "GLOBAL_BEST"
    else:
        run = args.run_name or cfg.get("train", {}).get("run_name", "run_current")
        model_dir = models_dir / "runs" / run
        tag = f"RUN:{run}"

    out_dir = model_dir / "udp_eval"
    ensure_dir(out_dir)

    packmod = load_pack_module()
    X = load_npz(processed_dir)

    # Decoder
    dec_w = model_dir / "dec_best.weights.h5"
    if not dec_w.exists():
        raise FileNotFoundError(f"Decoder weights not found: {dec_w}")

    decoder = build_decoder(latent_dim=32)
    _ = decoder(tf.zeros((1, 32), dtype=tf.float32), training=False)
    decoder.load_weights(dec_w)

    # Dequant params from encoder tflite (correct!)
    tflite_path = model_dir / "encoder_mu_int8.tflite"
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite encoder not found: {tflite_path}")

    tfl = tf.lite.Interpreter(model_path=str(tflite_path))
    tfl.allocate_tensors()
    out_det = tfl.get_output_details()[0]
    out_scale, out_zp = out_det["quantization"]
    print("[RECV9] Dequant params:", "scale=", out_scale, "zp=", out_zp)

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind_ip, args.port))
    sock.settimeout(float(args.socket_timeout_s))

    print(f"[RECV9] Listening udp://{args.bind_ip}:{args.port} | {tag}")
    print("[RECV9] Waiting for packets...")

    packets = 0
    frames = 0
    bytes_total = 0

    # seq tracking
    last_seq = None
    seq_gaps = 0
    out_of_order = 0

    # metrics accumulation
    mse_sum = 0.0
    topk_sum = 0.0
    bias_sum = 0.0
    blocks = []

    # ring buffers for waterfalls
    wf_orig = []
    wf_recon = []

    t0 = time.perf_counter()
    last_recv_time = time.perf_counter()

    try:
        while True:
            # stop conditions
            if args.max_packets > 0 and packets >= args.max_packets:
                break
            if args.idle_stop_s > 0 and (time.perf_counter() - last_recv_time) > args.idle_stop_s and packets > 0:
                print("[RECV9] Idle stop triggered.")
                break

            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue

            last_recv_time = time.perf_counter()
            packets += 1
            bytes_total += len(data)

            hdr, payload_comp = packmod.unpack_packet(data)
            raw = zlib.decompress(payload_comp)

            # meta v1: [ver][L][indices u32*L] + raw_block
            meta_ver = raw[0]
            if meta_ver != 1:
                print(f"[RECV9] WARN unsupported meta version: {meta_ver}, skipping packet.")
                continue

            L = raw[1]
            if L <= 0:
                continue

            idx_bytes = raw[2:2 + 4 * L]
            indices = np.frombuffer(idx_bytes, dtype=">u4").astype(np.int64)
            raw_block = raw[2 + 4 * L:]

            # seq checks
            if last_seq is not None:
                if hdr.seq < last_seq:
                    out_of_order += 1
                    print(f"[RECV9] WARN out-of-order: got {hdr.seq} after {last_seq}")
                elif hdr.seq > last_seq + 1:
                    seq_gaps += int(hdr.seq - last_seq - 1)
                    print(f"[RECV9] WARN seq gap: expected {last_seq+1} got {hdr.seq} (missed {hdr.seq-last_seq-1})")
            last_seq = hdr.seq

            # decode mu
            mu_block = packmod.decode_mu_block(raw_block)
            if mu_block.shape[0] != L:
                print("[RECV9] WARN L mismatch, skipping packet.")
                continue

            # bounds check indices
            if np.any(indices < 0) or np.any(indices >= X.shape[0]):
                print("[RECV9] WARN indices out of bounds, skipping packet.")
                continue

            # dequant
            mu_f = (mu_block.astype(np.float32) - float(out_zp)) * float(out_scale)

            # reconstruct
            xhat = decoder(mu_f, training=False).numpy()[:, :, 0]  # (L,1024)
            xorig = X[indices]                                     # (L,1024)

            frames += L

            # metrics
            mse_b = recon_loss_sum_mse(xorig, xhat)
            topk_b = topk_peak_mse(xorig, xhat, k=args.topk)
            bias_b = peak_bias(xorig, xhat)

            mse_sum += mse_b
            topk_sum += topk_b
            bias_sum += bias_b

            blocks.append({
                "seq": int(hdr.seq),
                "L": int(L),
                "bytes": int(len(data)),
                "mse_sum": float(mse_b),
                "topk_mse": float(topk_b),
                "peak_bias": float(bias_b),
            })

            # ring buffer for waterfall
            wf_orig.append(xorig)
            wf_recon.append(xhat)
            # trim
            while sum(w.shape[0] for w in wf_orig) > args.waterfall_max_frames:
                wf_orig.pop(0)
                wf_recon.pop(0)

            # overlay plot
            if args.plot_every > 0 and (packets % args.plot_every == 0):
                f0o = xorig[0]
                f0r = xhat[0]
                plt.figure(figsize=(10, 3))
                plt.plot(f0o, label="orig")
                plt.plot(f0r, label="recon")
                plt.title(f"Seq {hdr.seq} — overlay (frame 0)")
                plt.xlabel("bin")
                plt.ylabel("value")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / f"overlay_seq{int(hdr.seq):06d}.png", dpi=180)
                plt.close()

    except KeyboardInterrupt:
        print("[RECV9] Ctrl+C received, stopping...")

    dt = time.perf_counter() - t0
    if frames == 0:
        raise RuntimeError("No frames received. Run sender in another terminal with the same IP/port.")

    # build waterfalls from ring buffers
    W0 = np.vstack(wf_orig)
    W1 = np.vstack(wf_recon)

    plt.figure(figsize=(10, 5))
    plt.imshow(W0, aspect="auto", origin="lower", interpolation="nearest")
    plt.title("Waterfall ORIG (recent frames)")
    plt.xlabel("bin")
    plt.ylabel("frame")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / "waterfall_orig.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.imshow(W1, aspect="auto", origin="lower", interpolation="nearest")
    plt.title("Waterfall RECON (recent frames)")
    plt.xlabel("bin")
    plt.ylabel("frame")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / "waterfall_recon.png", dpi=180)
    plt.close()

    summary = {
        "tag": tag,
        "packets": int(packets),
        "frames": int(frames),
        "bytes_total": int(bytes_total),
        "avg_bytes_per_frame": float(bytes_total / frames),
        "frames_per_second": float(frames / max(1e-9, dt)),
        "seconds": float(dt),
        "seq_gaps": int(seq_gaps),
        "out_of_order": int(out_of_order),
        "metrics_avg_per_packet": {
            "mse_sum": float(mse_sum / max(1, packets)),
            "topk_mse": float(topk_sum / max(1, packets)),
            "peak_bias": float(bias_sum / max(1, packets)),
            "topk": int(args.topk),
        },
        "dequant": {"out_scale": float(out_scale), "out_zero_point": int(out_zp)},
        "plots_dir": str(out_dir),
        "blocks": blocks,
    }

    (out_dir / "udp_metrics.json").write_text(json.dumps(summary, indent=2))
    print("[RECV9] DONE. Saved plots + metrics in:", out_dir)
    print("[RECV9] avg bytes/frame:", summary["avg_bytes_per_frame"], "| frames/s:", summary["frames_per_second"])
    print("[RECV9] seq_gaps:", seq_gaps, "| out_of_order:", out_of_order)
    print("[RECV9] avg packet mse_sum:", summary["metrics_avg_per_packet"]["mse_sum"])


if __name__ == "__main__":
    main()