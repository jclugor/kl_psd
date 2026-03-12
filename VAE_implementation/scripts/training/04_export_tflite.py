#!/usr/bin/env python3
"""
04_export_tflite.py ??? Export encoder_mu_only to TFLite INT8 (PTQ) for Raspberry Pi deployment.

What it does:
- Loads preprocessed dataset (.npz) from data/processed/...
- Loads encoder weights from either:
    * GLOBAL_BEST (default)
    * or a specific run_name in models/runs/
- Builds encoder (same architecture as training) and encoder_mu_only (mu output only)
- Exports:
    * encoder_mu.keras (optional)
    * encoder_mu_int8.tflite (full integer quantized)
- Validates quantization by comparing:
    * mu_float (Keras model) vs mu_int8 (TFLite) on a sample batch
- Saves export_report.json with metrics and file paths

Usage:
  python VAE_implementation/scripts/training/04_export_tflite.py --config VAE_implementation/configs/vae_default.yaml
  python VAE_implementation/scripts/training/04_export_tflite.py --config ... --run_name run_current
  python VAE_implementation/scripts/training/04_export_tflite.py --config ... --use_latest
  python VAE_implementation/scripts/training/04_export_tflite.py --config ... --use_global_best

Outputs:
  VAE_implementation/models/GLOBAL_BEST/encoder_mu_int8.tflite   (if global)
  or
  VAE_implementation/models/runs/<run_name>/encoder_mu_int8.tflite

Notes:
- Requires: tensorflow installed (tflite converter is part of TF).
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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
# Architecture (must match training)
# -----------------------------
def build_encoder(input_bins: int = 1024, latent_dim: int = 32, include_dense128: bool = True) -> keras.Model:
    x_in = keras.Input(shape=(input_bins, 1), name="x_in")

    x = layers.Conv1D(16, kernel_size=5, strides=2, padding="same", name="enc_conv1")(x_in)
    x = layers.LeakyReLU(alpha=0.2, name="enc_lrelu1")(x)

    x = layers.Conv1D(32, kernel_size=3, strides=2, padding="same", name="enc_conv2")(x)
    x = layers.LeakyReLU(alpha=0.2, name="enc_lrelu2")(x)

    x = layers.Flatten(name="enc_flatten")(x)

    if include_dense128:
        x = layers.Dense(128, name="enc_dense")(x)
        x = layers.LeakyReLU(alpha=0.2, name="enc_lrelu_dense")(x)

    mu = layers.Dense(latent_dim, name="z_mu")(x)
    logvar = layers.Dense(latent_dim, name="z_logvar")(x)

    return keras.Model(x_in, [mu, logvar], name="encoder")


def build_encoder_mu_only(encoder: keras.Model, input_bins: int = 1024) -> keras.Model:
    x_in = keras.Input(shape=(input_bins, 1), name="x_in")
    mu, logvar = encoder(x_in, training=False)
    return keras.Model(x_in, mu, name="encoder_mu_only")


# -----------------------------
# Data loading (representative dataset)
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
    X = data["X"].astype(np.float32)  # (N,1024)
    return X


def representative_dataset_gen(X: np.ndarray, n_samples: int = 512, seed: int = 2026):
    """
    Generator for TFLite converter representative_dataset.
    Provides samples with shape (1,1024,1), values in [0,1] (expected).
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    for i in idx:
        x = X[i][None, :, None]  # (1,1024,1)
        yield [x.astype(np.float32)]


# -----------------------------
# TFLite inference (for validation)
# -----------------------------
def tflite_infer_mu(tflite_path: Path, X_batch: np.ndarray) -> np.ndarray:
    """
    Run TFLite model (int8) on a batch of float32 inputs.
    Handles input quantization/dequantization using interpreter quant params.
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale, in_zp = input_details["quantization"]
    out_scale, out_zp = output_details["quantization"]

    # X_batch: (B,1024,1) float32
    mu_out = []

    for i in range(X_batch.shape[0]):
        x = X_batch[i:i+1]  # (1,1024,1)
        # Quantize input to int8/uint8 depending on model
        if input_details["dtype"] == np.int8:
            xq = np.round(x / in_scale + in_zp).astype(np.int8)
        elif input_details["dtype"] == np.uint8:
            xq = np.round(x / in_scale + in_zp).astype(np.uint8)
        else:
            xq = x.astype(input_details["dtype"])

        interpreter.set_tensor(input_details["index"], xq)
        interpreter.invoke()
        yq = interpreter.get_tensor(output_details["index"])  # quantized
        # Dequantize output
        if output_details["dtype"] in (np.int8, np.uint8):
            y = (yq.astype(np.float32) - out_zp) * out_scale
        else:
            y = yq.astype(np.float32)
        mu_out.append(y)

    return np.concatenate(mu_out, axis=0)  # (B,32)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", default=None, help="models/runs/<run_name> (default from config)")
    ap.add_argument("--use_global_best", action="store_true", help="Load weights from models/GLOBAL_BEST")
    ap.add_argument("--use_latest", action="store_true", help="Use latest weights instead of best (run only)")
    ap.add_argument("--n_rep", type=int, default=512, help="Representative samples for PTQ")
    ap.add_argument("--n_val", type=int, default=64, help="Validation batch size for comparing mu")
    args = ap.parse_args()

    root = repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())

    processed_dir = root / cfg["paths"]["processed_dir"]
    models_dir = root / cfg["paths"]["models_dir"]

    input_bins = int(cfg.get("preprocess", {}).get("target_bins", 1024))
    latent_dim = 32
    include_dense128 = bool(cfg.get("train", {}).get("include_dense128", True))

    # Choose weights
    if args.use_global_best:
        tag = "GLOBAL_BEST"
        out_dir = models_dir / "GLOBAL_BEST"
        enc_w = out_dir / "enc_best.weights.h5"
    else:
        run_name = args.run_name or cfg.get("train", {}).get("run_name", "run_current")
        tag = f"RUN:{run_name}" + (":latest" if args.use_latest else ":best")
        out_dir = models_dir / "runs" / run_name
        enc_w = out_dir / ("enc_latest.weights.h5" if args.use_latest else "enc_best.weights.h5")

    ensure_dir(out_dir)

    if not enc_w.exists():
        raise FileNotFoundError(f"Encoder weights not found: {enc_w}")

    # Load data for representative dataset
    X = load_npz(processed_dir)

    # Build encoder + load weights
    encoder = build_encoder(input_bins=input_bins, latent_dim=latent_dim, include_dense128=include_dense128)
    _ = encoder(tf.zeros((1, input_bins, 1), dtype=tf.float32), training=False)  # build
    encoder.load_weights(enc_w)

    encoder_mu = build_encoder_mu_only(encoder, input_bins=input_bins)
    _ = encoder_mu(tf.zeros((1, input_bins, 1), dtype=tf.float32), training=False)  # build

    # Export Keras (optional)
    keras_path = out_dir / "encoder_mu.keras"
    try:
        encoder_mu.save(keras_path)
    except Exception:
        # Some envs might not support saving .keras cleanly; ignore if fails
        keras_path = None

    # Convert to TFLite INT8 (PTQ)
    converter = tf.lite.TFLiteConverter.from_keras_model(encoder_mu)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(X, n_samples=args.n_rep)

    # Full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    tflite_path = out_dir / "encoder_mu_int8.tflite"
    tflite_path.write_bytes(tflite_model)

    # Validation: compare mu float vs mu tflite on a small batch
    rng = np.random.default_rng(2026)
    n_val = min(args.n_val, X.shape[0])
    idx = rng.choice(X.shape[0], size=n_val, replace=False)
    Xb = X[idx][:, :, None].astype(np.float32)  # (B,1024,1)

    mu_float = encoder_mu.predict(Xb, verbose=0)  # (B,32)
    mu_tfl = tflite_infer_mu(tflite_path, Xb)     # (B,32)

    diff = mu_tfl - mu_float
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(np.abs(diff)))

    report = {
        "tag": tag,
        "processed_dir": str(processed_dir),
        "encoder_weights": str(enc_w),
        "export_dir": str(out_dir),
        "keras_encoder_mu": str(keras_path) if keras_path is not None else None,
        "tflite_encoder_mu_int8": str(tflite_path),
        "ptq": {
            "n_rep": int(args.n_rep),
            "n_val": int(n_val),
            "mu_mae": mae,
            "mu_rmse": rmse,
            "mu_max_abs": max_abs
        }
    }

    report_path = out_dir / "export_report.json"
    safe_write_json(report_path, report)

    print("[EXPORT] Saved:", tflite_path)
    print("[EXPORT] Report:", report_path)
    print("[EXPORT] mu MAE:", mae, "RMSE:", rmse, "MAX_ABS:", max_abs)


if __name__ == "__main__":
    main()
