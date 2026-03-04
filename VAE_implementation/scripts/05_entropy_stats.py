#!/usr/bin/env python3
"""
05_entropy_stats.py — Estimate bitrate/entropy of the INT8 latent (mu) and its temporal delta.

Inputs:
- Preprocessed dataset (.npz) and metadata.csv (timestamps)
- TFLite model: encoder_mu_int8.tflite (INT8 in/out)
- Splits: uses val or test indices

Computes:
- Symbol histogram for mu_int8 (all 32 dims, all frames)
- Symbol histogram for delta(mu_int8) using timestamp order within each source_file
- Entropy H(mu) and H(delta) in bits/symbol
- Approx bytes/frame for raw mu and delta-coded mu (entropy lower bound)

Usage:
  python VAE_implementation/scripts/05_entropy_stats.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --split test
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import tensorflow as tf


# -----------------------------
# Repo helpers
# -----------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
        raise KeyError(f"NPZ keys: {list(data.keys())} — expected 'X'.")
    return data["X"].astype(np.float32)


def load_splits(processed_dir: Path):
    splits_dir = processed_dir / "splits"
    tr = np.load(splits_dir / "train_idx.npy")
    va = np.load(splits_dir / "val_idx.npy")
    te = np.load(splits_dir / "test_idx.npy")
    return tr, va, te


def load_metadata(processed_dir: Path) -> pd.DataFrame:
    meta_path = processed_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {processed_dir} (needed for time ordering).")
    return pd.read_csv(meta_path)


# -----------------------------
# TFLite inference (INT8 mu)
# -----------------------------
def load_tflite(tflite_path: Path):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    return interpreter, in_det, out_det


def infer_mu_int8(interpreter, in_det, out_det, X_batch: np.ndarray) -> np.ndarray:
    """
    X_batch: (B,1024) float32 normalized in [0,1]
    Returns mu_int8: (B,32) int8 (raw quantized output tensor)
    """
    in_scale, in_zp = in_det["quantization"]

    mus = []
    for i in range(X_batch.shape[0]):
        x = X_batch[i:i+1, :, None].astype(np.float32)  # (1,1024,1)
        # quantize input
        if in_det["dtype"] == np.int8:
            xq = np.round(x / in_scale + in_zp).astype(np.int8)
        elif in_det["dtype"] == np.uint8:
            xq = np.round(x / in_scale + in_zp).astype(np.uint8)
        else:
            xq = x.astype(in_det["dtype"])

        interpreter.set_tensor(in_det["index"], xq)
        interpreter.invoke()
        yq = interpreter.get_tensor(out_det["index"])  # (1,32) int8
        mus.append(yq)

    mu_q = np.concatenate(mus, axis=0)
    # Ensure int8 type (some runtimes may return int32)
    mu_q = mu_q.astype(np.int8, copy=False)
    return mu_q


# -----------------------------
# Entropy helpers
# -----------------------------
def entropy_from_hist(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def hist_int8(values: np.ndarray) -> np.ndarray:
    """
    values: int8 array
    returns counts[256] for symbols -128..127
    """
    v = values.astype(np.int16) + 128
    counts = np.bincount(v.ravel(), minlength=256)
    return counts


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--use_global_best", action="store_true")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--keyframe_every", type=int, default=0,
                    help="If >0, compute delta within blocks separated by keyframes (simulates periodic absolute frames).")
    args = ap.parse_args()

    root = repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())

    processed_dir = root / cfg["paths"]["processed_dir"]
    models_dir = root / cfg["paths"]["models_dir"]

    # Choose tflite
    if args.use_global_best:
        out_dir = models_dir / "GLOBAL_BEST"
        tag = "GLOBAL_BEST"
    else:
        run_name = args.run_name or cfg.get("train", {}).get("run_name", "run_current")
        out_dir = models_dir / "runs" / run_name
        tag = f"RUN:{run_name}"

    tflite_path = out_dir / "encoder_mu_int8.tflite"
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}. Run 04_export_tflite.py first.")

    X = load_npz(processed_dir)
    tr, va, te = load_splits(processed_dir)
    meta = load_metadata(processed_dir)

    if args.split == "train":
        idx = tr
    elif args.split == "val":
        idx = va
    else:
        idx = te

    # Use the subset of metadata aligned to idx (same ordering used when building dataset)
    meta_sub = meta.iloc[idx].copy()
    X_sub = X[idx].copy()

    # Order by (source_file, timestamp) for delta coding
    if "timestamp" not in meta_sub.columns or meta_sub["timestamp"].isna().all():
        raise ValueError("metadata.csv has no valid 'timestamp' for time ordering.")

    meta_sub["__i"] = np.arange(len(meta_sub))
    meta_sub.sort_values(["source_file", "timestamp"], inplace=True, kind="mergesort")

    # TFLite interpreter
    interpreter, in_det, out_det = load_tflite(tflite_path)

    # Inference in batches (following sorted order)
    order = meta_sub["__i"].to_numpy()
    mu_all = []

    B = args.batch
    for s in range(0, len(order), B):
        sl = order[s:s+B]
        mu_q = infer_mu_int8(interpreter, in_det, out_det, X_sub[sl])
        mu_all.append(mu_q)

    mu_all = np.concatenate(mu_all, axis=0)  # (N,32) int8, aligned to meta_sub sorted order

    # Entropy for raw mu symbols
    counts_mu = hist_int8(mu_all)
    H_mu = entropy_from_hist(counts_mu)  # bits/symbol
    bytes_per_frame_mu = 32 * H_mu / 8.0

    # Delta coding (within each source_file)
    # We compute delta along time for each source_file group.
    deltas = []
    sf = meta_sub["source_file"].to_numpy()
    # segment boundaries
    start = 0
    while start < len(sf):
        end = start
        while end < len(sf) and sf[end] == sf[start]:
            end += 1
        block = mu_all[start:end].astype(np.int16)  # (T,32)

        if args.keyframe_every and args.keyframe_every > 0:
            k = int(args.keyframe_every)
            for t0 in range(0, block.shape[0], k):
                seg = block[t0:t0+k]
                if seg.shape[0] <= 1:
                    continue
                d = seg[1:] - seg[:-1]
                deltas.append(d)
        else:
            if block.shape[0] > 1:
                d = block[1:] - block[:-1]
                deltas.append(d)

        start = end

    if len(deltas) == 0:
        delta_all = np.zeros((0, 32), dtype=np.int16)
    else:
        delta_all = np.vstack(deltas)

    # Clip to int8 range for "stored as int8 deltas" interpretation (common)
    delta_int8 = np.clip(delta_all, -128, 127).astype(np.int8)

    counts_d = hist_int8(delta_int8)
    H_d = entropy_from_hist(counts_d)
    bytes_per_frame_delta = 32 * H_d / 8.0

    report = {
        "tag": tag,
        "split": args.split,
        "n_frames": int(mu_all.shape[0]),
        "tflite": str(tflite_path),
        "entropy": {
            "H_mu_bits_per_symbol": H_mu,
            "H_delta_bits_per_symbol": H_d,
            "bytes_per_frame_mu_lower_bound": bytes_per_frame_mu,
            "bytes_per_frame_delta_lower_bound": bytes_per_frame_delta,
            "keyframe_every": int(args.keyframe_every),
            "delta_samples": int(delta_int8.shape[0]),
        },
        "notes": [
            "Entropy is a lower bound (ideal coder). Real Huffman/ANS will be close but not identical.",
            "Delta uses int8 clipping; if you store int16 deltas, re-run without clipping and compute a different histogram."
        ]
    }

    out_path = out_dir / "entropy_report.json"
    safe_write_json(out_path, report)

    print("[ENTROPY] Saved:", out_path)
    print("[ENTROPY] H(mu)   =", H_mu, "bits/symbol ->", bytes_per_frame_mu, "bytes/frame (lower bound)")
    print("[ENTROPY] H(delta)=", H_d,  "bits/symbol ->", bytes_per_frame_delta, "bytes/frame (lower bound)")
    print("[ENTROPY] delta samples:", delta_int8.shape[0])


if __name__ == "__main__":
    main()