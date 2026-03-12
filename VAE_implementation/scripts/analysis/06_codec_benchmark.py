#!/usr/bin/env python3
"""
06_codec_benchmark.py ??? End-to-end benchmark with timing (encode -> packetize -> compress -> decompress -> unpack -> reconstruct)

Uses:
- TFLite INT8 encoder_mu model to produce mu_int8 (32 bytes/frame)
- Delta coding with periodic keyframes: blocks contain [L][mu0][deltas]
- Real compression: zlib / lzma / bz2 (baseline)
- Optional reconstruction using Keras decoder weights (server-side)
- Measures time for each stage + end-to-end time per codec

Outputs:
- codec_report.json saved into the selected model directory (GLOBAL_BEST or runs/<run_name>)

Run examples:
  python VAE_implementation/scripts/analysis/06_codec_benchmark.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --split train --keyframe_every 30
  python VAE_implementation/scripts/analysis/06_codec_benchmark.py --config ... --run_name run_current --split test --keyframe_every 30
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import zlib
import lzma
import bz2


# -----------------------------
# Repo helpers
# -----------------------------
def repo_root() -> Path:
    """Return the project root for repository-relative config paths."""

    return Path(__file__).resolve().parents[3]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))


# -----------------------------
# Data loading
# -----------------------------
def load_npz(processed_dir: Path) -> np.ndarray:
    npz_path = processed_dir / "dataset_psd_1024_norm.npz"
    if not npz_path.exists():
        candidates = sorted(processed_dir.glob("*.npz"), key=lambda p: p.stat().st_size, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No .npz found in {processed_dir}")
        npz_path = candidates[0]
    data = np.load(npz_path, allow_pickle=True)
    if "X" not in data:
        raise KeyError(f"NPZ keys: {list(data.keys())} ??? expected 'X'.")
    return data["X"].astype(np.float32)


def load_splits(processed_dir: Path):
    sdir = processed_dir / "splits"
    tr = np.load(sdir / "train_idx.npy")
    va = np.load(sdir / "val_idx.npy")
    te = np.load(sdir / "test_idx.npy")
    return tr, va, te


def load_metadata(processed_dir: Path) -> pd.DataFrame:
    meta_path = processed_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {processed_dir}")
    return pd.read_csv(meta_path)


# -----------------------------
# Decoder architecture (must match training)
# -----------------------------
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


# -----------------------------
# TFLite inference (INT8 mu)
# -----------------------------
def load_tflite(tflite_path: Path):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    return interpreter, in_det, out_det


def infer_mu_int8_batched(interpreter, in_det, out_det, X_sorted: np.ndarray, batch: int) -> Tuple[np.ndarray, Dict]:
    """
    X_sorted: (N,1024) float32 in [0,1]
    Returns:
      mu_q: (N,32) int8
      stats with timing
    """
    N = X_sorted.shape[0]
    B = max(1, int(batch))

    # Prepare fixed input tensor with batch size B to avoid re-alloc every batch.
    # We'll pad last batch with zeros.
    input_index = in_det["index"]
    input_dtype = in_det["dtype"]
    in_scale, in_zp = in_det["quantization"]

    output_index = out_det["index"]
    out_dtype = out_det["dtype"]
    out_scale, out_zp = out_det["quantization"]

    # Resize input tensor to (B,1024,1) once
    interpreter.resize_tensor_input(input_index, [B, 1024, 1], strict=False)
    interpreter.allocate_tensors()

    mu_list = []
    t0 = time.perf_counter()

    for s in range(0, N, B):
        n = min(B, N - s)
        x = X_sorted[s:s+n]  # (n,1024)
        xb = np.zeros((B, 1024, 1), dtype=np.float32)
        xb[:n, :, 0] = x

        # Quantize input batch
        if input_dtype == np.int8:
            xq = np.round(xb / in_scale + in_zp).astype(np.int8)
        elif input_dtype == np.uint8:
            xq = np.round(xb / in_scale + in_zp).astype(np.uint8)
        else:
            xq = xb.astype(input_dtype)

        interpreter.set_tensor(input_index, xq)
        interpreter.invoke()
        yq = interpreter.get_tensor(output_index)  # (B,32) quantized
        yq = yq[:n]

        mu_list.append(yq.astype(np.int8, copy=False))

    t1 = time.perf_counter()
    mu_q = np.concatenate(mu_list, axis=0)

    stats = {
        "encode_seconds": float(t1 - t0),
        "encode_ms_per_frame": float(1000.0 * (t1 - t0) / max(1, N)),
        "input_dtype": str(input_dtype),
        "output_dtype": str(out_dtype),
        "input_quant": {"scale": float(in_scale), "zero_point": int(in_zp)},
        "output_quant": {"scale": float(out_scale), "zero_point": int(out_zp)},
        "batch": int(B),
    }
    return mu_q, stats


# -----------------------------
# Packet building / unpacking
# -----------------------------
def build_blocks(mu_all: np.ndarray, source_file: np.ndarray, keyframe_every: int) -> Tuple[List[bytes], Dict]:
    """
    Payload format:
      [L (1 byte)] + mu0 (32 bytes) + deltas ((L-1)*32 bytes, int8 clipped)
    """
    if keyframe_every is None or keyframe_every <= 1:
        keyframe_every = 1

    blocks: List[bytes] = []
    total_frames = int(mu_all.shape[0])
    total_delta_frames = 0

    start = 0
    while start < total_frames:
        end = start
        while end < total_frames and source_file[end] == source_file[start]:
            end += 1

        group = mu_all[start:end].astype(np.int16)  # (T,32)
        T = group.shape[0]

        for t0 in range(0, T, keyframe_every):
            seg = group[t0:t0 + keyframe_every]
            L = seg.shape[0]
            if L <= 0:
                continue

            mu0 = seg[0].astype(np.int8)
            if L == 1:
                blocks.append(bytes([1]) + mu0.tobytes())
                continue

            d = seg[1:] - seg[:-1]                   # (L-1,32) int16
            d8 = np.clip(d, -128, 127).astype(np.int8)
            total_delta_frames += (L - 1)

            blocks.append(bytes([L]) + mu0.tobytes() + d8.tobytes())

        start = end

    stats = {
        "n_frames": total_frames,
        "n_blocks": len(blocks),
        "keyframe_every": int(keyframe_every),
        "delta_frames": int(total_delta_frames),
    }
    return blocks, stats


def unpack_blocks(payloads: List[bytes]) -> np.ndarray:
    """
    Inverse of build_blocks(): returns mu_int8 stream (N,32).
    """
    out = []
    for p in payloads:
        if len(p) < 1 + 32:
            continue
        L = p[0]
        mu0 = np.frombuffer(p[1:33], dtype=np.int8).astype(np.int16)
        out.append(mu0.copy())

        if L > 1:
            dbytes = p[33:]
            expected = (L - 1) * 32
            if len(dbytes) < expected:
                # truncated block, skip
                continue
            d = np.frombuffer(dbytes[:expected], dtype=np.int8).reshape((L - 1, 32)).astype(np.int16)

            prev = mu0
            for i in range(L - 1):
                nxt = prev + d[i]
                nxt = np.clip(nxt, -128, 127)
                out.append(nxt.copy())
                prev = nxt

    mu = np.stack(out, axis=0).astype(np.int8)
    return mu


# -----------------------------
# Reconstruction metrics
# -----------------------------
def recon_loss_sum_mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.mean(np.sum((x - x_hat) ** 2, axis=(1, 2))))


def topk_peak_mse(x: np.ndarray, x_hat: np.ndarray, k: int = 32) -> float:
    x0 = x[:, :, 0]
    y0 = x_hat[:, :, 0]
    idx = np.argpartition(x0, -k, axis=1)[:, -k:]
    rows = np.arange(x0.shape[0])[:, None]
    xpk = x0[rows, idx]
    ypk = y0[rows, idx]
    return float(np.mean((xpk - ypk) ** 2))


def peak_bias(x: np.ndarray, x_hat: np.ndarray) -> float:
    mx = np.max(x[:, :, 0], axis=1)
    my = np.max(x_hat[:, :, 0], axis=1)
    return float(np.mean(my - mx))


# -----------------------------
# Codec benchmarking
# -----------------------------
def compress_block(b: bytes, codec: str, level: int) -> bytes:
    if codec == "none":
        return b
    if codec == "zlib":
        return zlib.compress(b, level=level)
    if codec == "lzma":
        return lzma.compress(b, preset=level)
    if codec == "bz2":
        return bz2.compress(b, compresslevel=level)
    raise ValueError(codec)


def decompress_block(b: bytes, codec: str) -> bytes:
    if codec == "none":
        return b
    if codec == "zlib":
        return zlib.decompress(b)
    if codec == "lzma":
        return lzma.decompress(b)
    if codec == "bz2":
        return bz2.decompress(b)
    raise ValueError(codec)


def bench_codec_roundtrip(blocks: List[bytes], codec: str, level: int) -> Tuple[List[bytes], Dict]:
    """
    Compress and decompress all blocks; returns decompressed blocks and timing stats.
    """
    t0 = time.perf_counter()
    comp = [compress_block(b, codec, level) for b in blocks]
    t1 = time.perf_counter()
    decomp = [decompress_block(b, codec) for b in comp]
    t2 = time.perf_counter()

    stats = {
        "codec": codec,
        "level": int(level),
        "compressed_bytes_total": int(sum(len(c) for c in comp)),
        "compressed_bytes_per_frame": None,  # filled later
        "compress_seconds": float(t1 - t0),
        "decompress_seconds": float(t2 - t1),
    }
    return decomp, stats


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--use_global_best", action="store_true")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--keyframe_every", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_frames", type=int, default=0, help="limit frames after sorting (faster)")
    ap.add_argument("--topk", type=int, default=32)
    args = ap.parse_args()

    root = repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())

    processed_dir = root / cfg["paths"]["processed_dir"]
    models_dir = root / cfg["paths"]["models_dir"]

    # Select model directory
    if args.use_global_best:
        out_dir = models_dir / "GLOBAL_BEST"
        tag = "GLOBAL_BEST"
        dec_w = out_dir / "dec_best.weights.h5"
    else:
        run_name = args.run_name or cfg.get("train", {}).get("run_name", "run_current")
        out_dir = models_dir / "runs" / run_name
        tag = f"RUN:{run_name}"
        dec_w = out_dir / "dec_best.weights.h5"

    ensure_dir(out_dir)

    tflite_path = out_dir / "encoder_mu_int8.tflite"
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite not found: {tflite_path} (run 04_export_tflite.py first).")

    # Load dataset + metadata
    X = load_npz(processed_dir)
    tr, va, te = load_splits(processed_dir)
    meta = load_metadata(processed_dir)

    if args.split == "train":
        idx = tr
    elif args.split == "val":
        idx = va
    else:
        idx = te

    X_sub = X[idx].copy()
    meta_sub = meta.iloc[idx].copy()

    if "timestamp" not in meta_sub.columns or meta_sub["timestamp"].isna().all():
        raise ValueError("metadata.csv has no valid 'timestamp' for time ordering.")

    meta_sub["__i"] = np.arange(len(meta_sub))
    meta_sub.sort_values(["source_file", "timestamp"], inplace=True, kind="mergesort")

    order = meta_sub["__i"].to_numpy()
    sf = meta_sub["source_file"].to_numpy()

    if args.max_frames and args.max_frames > 0:
        order = order[:args.max_frames]
        sf = sf[:args.max_frames]

    # Sorted frames (stream order)
    X_sorted = X_sub[order]  # (N,1024)
    N = int(X_sorted.shape[0])

    # -----------------------------
    # Stage A: ENCODING (TFLite)
    # -----------------------------
    interpreter, in_det, out_det = load_tflite(tflite_path)
    mu_all, enc_stats = infer_mu_int8_batched(interpreter, in_det, out_det, X_sorted, batch=args.batch)

    # -----------------------------
    # Stage B: PACKETIZE (delta)
    # -----------------------------
    t0 = time.perf_counter()
    blocks, blk_stats = build_blocks(mu_all, sf, keyframe_every=args.keyframe_every)
    t1 = time.perf_counter()
    packetize_stats = {
        "packetize_seconds": float(t1 - t0),
        "packetize_ms_per_frame": float(1000.0 * (t1 - t0) / max(1, N)),
        **blk_stats
    }

    # Raw sizes
    raw_mu_bytes_total = int(N * 32)
    raw_mu_bytes_per_frame = float(raw_mu_bytes_total / max(1, N))
    raw_block_bytes_total = int(sum(len(b) for b in blocks))
    raw_block_bytes_per_frame = float(raw_block_bytes_total / max(1, N))

    # -----------------------------
    # Stage C/D: CODEC BENCH (compress+decompress) + UNPACK timing
    # We'll also do UNPACK once from "none" (equivalent payloads), and RECON once.
    # -----------------------------
    codecs = [
        ("none", 0),
        ("zlib", 1), ("zlib", 6), ("zlib", 9),
        ("lzma", 1), ("lzma", 6), ("lzma", 9),
        ("bz2", 1), ("bz2", 9),
    ]

    bench = []
    unpack_reference = None
    unpack_stats_reference = None

    for codec, level in codecs:
        decomp_blocks, st = bench_codec_roundtrip(blocks, codec, level)
        st["compressed_bytes_per_frame"] = float(st["compressed_bytes_total"] / max(1, N))

        # Time unpack (receiver-side) for this codec
        tU0 = time.perf_counter()
        mu_rec = unpack_blocks(decomp_blocks)
        tU1 = time.perf_counter()
        st["unpack_seconds"] = float(tU1 - tU0)
        st["unpack_ms_per_frame"] = float(1000.0 * (tU1 - tU0) / max(1, mu_rec.shape[0]))
        st["mu_rec_frames"] = int(mu_rec.shape[0])

        # Keep reference stream for reconstruction (first codec == none)
        if unpack_reference is None:
            unpack_reference = mu_rec
            unpack_stats_reference = {"unpack_seconds": st["unpack_seconds"], "mu_rec_frames": st["mu_rec_frames"]}

        bench.append(st)

    # -----------------------------
    # Stage E: RECONSTRUCTION (decoder) once, using unpack_reference
    # -----------------------------
    recon_section = {"performed": False}
    if dec_w.exists() and unpack_reference is not None:
        # Dequantize mu_int8 -> float using TFLite output quant params
        out_scale, out_zp = out_det["quantization"]
        mu_float = (unpack_reference.astype(np.float32) - float(out_zp)) * float(out_scale)  # (N,32)

        decoder = build_decoder(latent_dim=32)
        _ = decoder(tf.zeros((1, 32), dtype=tf.float32), training=False)  # build
        decoder.load_weights(dec_w)

        # Batch decode for speed
        Bdec = 256
        tR0 = time.perf_counter()
        xhats = []
        for s in range(0, mu_float.shape[0], Bdec):
            mb = mu_float[s:s+Bdec]
            xh = decoder(mb, training=False).numpy()  # (b,1024,1)
            xhats.append(xh)
        X_hat = np.concatenate(xhats, axis=0)
        tR1 = time.perf_counter()

        # Align original X for the reconstructed frames count
        M = min(X_hat.shape[0], X_sorted.shape[0])
        X0 = X_sorted[:M][:, :, None].astype(np.float32)

        recon = recon_loss_sum_mse(X0, X_hat[:M])
        pk_mse = topk_peak_mse(X0, X_hat[:M], k=args.topk)
        pk_bias = peak_bias(X0, X_hat[:M])

        recon_section = {
            "performed": True,
            "decoder_weights": str(dec_w),
            "recon_seconds": float(tR1 - tR0),
            "recon_ms_per_frame": float(1000.0 * (tR1 - tR0) / max(1, M)),
            "frames_reconstructed": int(M),
            "metrics": {
                "recon_loss_sum_mse": float(recon),
                "topk_peak_mse": float(pk_mse),
                "peak_bias_mean_recon_minus_orig": float(pk_bias),
                "topk": int(args.topk),
            },
        }
    else:
        recon_section = {
            "performed": False,
            "reason": "dec_best.weights.h5 not found in selected directory, or unpack failed.",
            "expected_decoder_path": str(dec_w),
        }

    # -----------------------------
    # End-to-end time per codec (includes encode + packetize + compress+decompress + unpack + recon)
    # We include recon time as the same constant across codecs (since mu stream after decompress should match).
    # -----------------------------
    recon_time = float(recon_section.get("recon_seconds", 0.0))
    for st in bench:
        st["end_to_end_seconds"] = float(
            enc_stats["encode_seconds"]
            + packetize_stats["packetize_seconds"]
            + st["compress_seconds"]
            + st["decompress_seconds"]
            + st["unpack_seconds"]
            + recon_time
        )
        st["end_to_end_ms_per_frame"] = float(1000.0 * st["end_to_end_seconds"] / max(1, N))

    report = {
        "tag": tag,
        "split": args.split,
        "n_frames": int(N),
        "keyframe_every": int(args.keyframe_every),
        "tflite": str(tflite_path),
        "encoding": enc_stats,
        "packetize": packetize_stats,
        "baselines": {
            "raw_mu_bytes_total": raw_mu_bytes_total,
            "raw_mu_bytes_per_frame": raw_mu_bytes_per_frame,
            "raw_block_bytes_total": raw_block_bytes_total,
            "raw_block_bytes_per_frame": raw_block_bytes_per_frame,
            "note": "raw_mu is 32 int8/frame. raw_block is keyframe+delta payload with 1-byte header per block."
        },
        "reconstruction": recon_section,
        "benchmarks": bench,
        "notes": [
            "Compression codecs are baselines (zlib/lzma/bz2). For production use ANS/Huffman tailored to symbol stats.",
            "End-to-end time includes encoder inference + packetize + codec + unpack + (optional) decoder reconstruction.",
            "Timings depend heavily on CPU and batch sizes."
        ]
    }

    out_path = out_dir / "codec_report.json"
    safe_write_json(out_path, report)

    print("[CODEC] Saved:", out_path)
    print("[CODEC] Encode ms/frame:", enc_stats["encode_ms_per_frame"])
    print("[CODEC] Packetize ms/frame:", packetize_stats["packetize_ms_per_frame"])
    if recon_section.get("performed"):
        print("[CODEC] Recon ms/frame:", recon_section["recon_ms_per_frame"])
        print("[CODEC] Recon loss:", recon_section["metrics"]["recon_loss_sum_mse"])
    else:
        print("[CODEC] Recon skipped:", recon_section.get("reason"))

    print("[CODEC] Results (bytes/frame, end-to-end ms/frame):")
    for r in bench:
        print(f"  - {r['codec']:>4} lvl={r['level']:>2} -> {r['compressed_bytes_per_frame']:.3f} B/f | {r['end_to_end_ms_per_frame']:.3f} ms/f")


if __name__ == "__main__":
    main()
