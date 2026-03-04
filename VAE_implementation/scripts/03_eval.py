#!/usr/bin/env python3
"""
03_eval.py — Evaluate VAE (encoder/decoder) on the test split and save metrics + plots locally.

What it does:
- Loads preprocessed dataset (.npz) + splits from data/processed/...
- Loads encoder/decoder weights:
    * by default: RUN best (enc_best/dec_best)
    * optional: RUN latest (enc_latest/dec_latest)
    * optional: GLOBAL_BEST (enc_best/dec_best under GLOBAL_BEST)
- Runs deterministic inference (z ≈ mu), consistent with edge deployment
- Computes:
    * recon_loss: mean over batch of sum_{bins}(x - x_hat)^2
    * kl_loss: mean over batch of KL(q(z|x)||N(0,I))
    * total_loss: recon + beta_report * kl
    * peak metrics: top-k peak MSE, peak bias (max recon - max orig)
- Saves:
    * eval/metrics.json
    * eval/ plots: recon overlays, hist, waterfalls

Usage:
  python VAE_implementation/scripts/03_eval.py --config VAE_implementation/configs/vae_default.yaml --run_name run_current
  python VAE_implementation/scripts/03_eval.py --config ... --use_global_best
  python VAE_implementation/scripts/03_eval.py --config ... --run_name run_current --use_latest
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# Paths / utils
# -----------------------------
def repo_root() -> Path:
    # .../kl_psd/VAE_implementation/scripts/03_eval.py -> parents[2] = kl_psd
    return Path(__file__).resolve().parents[2]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))


# -----------------------------
# Architecture (must match 02_train.py)
# -----------------------------
def _has_conv1d_transpose() -> bool:
    return hasattr(layers, "Conv1DTranspose")


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


def build_decoder(input_bins: int = 1024, latent_dim: int = 32) -> keras.Model:
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
# Data loading
# -----------------------------
def load_dataset(processed_dir: Path) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    npz_path = processed_dir / "dataset_psd_1024_norm.npz"
    if not npz_path.exists():
        candidates = sorted(processed_dir.glob("*.npz"), key=lambda p: p.stat().st_size, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No .npz found in {processed_dir}")
        npz_path = candidates[0]

    data = np.load(npz_path, allow_pickle=True)
    if "X" not in data:
        raise KeyError(f"NPZ keys: {list(data.keys())} — expected 'X'.")
    X = data["X"].astype(np.float32)
    freqs = data["freqs_hz"] if "freqs_hz" in data else None

    meta = {"npz_path": str(npz_path)}
    for k in ["gmin", "gmax", "normalize_mode"]:
        if k in data:
            try:
                meta[k] = data[k].item() if hasattr(data[k], "item") else data[k]
            except Exception:
                meta[k] = str(data[k])
    return X, freqs, meta


def load_splits(processed_dir: Path):
    splits_dir = processed_dir / "splits"
    tr = np.load(splits_dir / "train_idx.npy")
    va = np.load(splits_dir / "val_idx.npy")
    te = np.load(splits_dir / "test_idx.npy")
    return tr, va, te


# -----------------------------
# Metrics
# -----------------------------
def recon_loss_sum_mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    # mean over batch of sum over bins
    return float(np.mean(np.sum((x - x_hat) ** 2, axis=(1, 2))))


def kl_diag_gaussian(mu: np.ndarray, logvar: np.ndarray) -> float:
    return float(np.mean(-0.5 * np.sum(1.0 + logvar - (mu ** 2) - np.exp(logvar), axis=1)))


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
# Plot helpers
# -----------------------------
def plot_overlays(save_path: Path, X: np.ndarray, X_hat: np.ndarray, title: str,
                  freqs_mhz: Optional[np.ndarray] = None, n: int = 6) -> None:
    ensure_dir(save_path.parent)
    n = min(n, X.shape[0])
    rng = np.random.default_rng(2026)
    idx = rng.choice(X.shape[0], size=n, replace=False)

    plt.figure(figsize=(12, 2.5 * n))
    for i, j in enumerate(idx, start=1):
        ax = plt.subplot(n, 1, i)
        x = X[j, :, 0]
        y = X_hat[j, :, 0]
        if freqs_mhz is not None:
            ax.plot(freqs_mhz, x, label="orig")
            ax.plot(freqs_mhz, y, label="recon")
            ax.set_xlabel("Frequency (MHz)")
        else:
            ax.plot(x, label="orig")
            ax.plot(y, label="recon")
            ax.set_xlabel("Bin")
        ax.set_ylabel("Value")
        ax.grid(True)
        if i == 1:
            ax.legend()
        ax.set_title(f"{title} — sample {j}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_hist(save_path: Path, X: np.ndarray, title: str) -> None:
    ensure_dir(save_path.parent)
    v = X[np.isfinite(X)].ravel()
    plt.figure(figsize=(8, 4))
    plt.hist(v, bins=120)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_waterfall(save_path: Path, X: np.ndarray, title: str, max_frames: int = 300) -> None:
    ensure_dir(save_path.parent)
    if X.shape[0] > max_frames:
        idx = np.linspace(0, X.shape[0] - 1, max_frames).astype(int)
        W = X[idx, :, 0]
        extra = f"(subset {max_frames}/{X.shape[0]})"
    else:
        W = X[:, :, 0]
        extra = ""

    plt.figure(figsize=(10, 6))
    plt.imshow(W, aspect="auto", interpolation="nearest", origin="lower")
    plt.title(f"{title} {extra}")
    plt.xlabel("Frequency bin")
    plt.ylabel("Frame index (time)")
    plt.colorbar(label="Value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", default=None, help="models/runs/<run_name>")
    ap.add_argument("--use_global_best", action="store_true", help="Evaluate models/GLOBAL_BEST")
    ap.add_argument("--use_latest", action="store_true", help="Use enc_latest/dec_latest instead of enc_best/dec_best (run only)")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--n_plots", type=int, default=6)
    args = ap.parse_args()

    root = repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())

    processed_dir = root / cfg["paths"]["processed_dir"]
    models_dir = root / cfg["paths"]["models_dir"]

    # Select weights
    if args.use_global_best:
        tag = "GLOBAL_BEST"
        run_dir = models_dir / "GLOBAL_BEST"
        enc_w = run_dir / "enc_best.weights.h5"
        dec_w = run_dir / "dec_best.weights.h5"
    else:
        run_name = args.run_name or cfg.get("train", {}).get("run_name", "run_current")
        tag = f"RUN:{run_name}" + (":latest" if args.use_latest else ":best")
        run_dir = models_dir / "runs" / run_name
        if args.use_latest:
            enc_w = run_dir / "enc_latest.weights.h5"
            dec_w = run_dir / "dec_latest.weights.h5"
        else:
            enc_w = run_dir / "enc_best.weights.h5"
            dec_w = run_dir / "dec_best.weights.h5"

    if not enc_w.exists() or not dec_w.exists():
        raise FileNotFoundError(f"Missing weights:\n  encoder: {enc_w}\n  decoder: {dec_w}")

    # Load data
    X, freqs_hz, meta = load_dataset(processed_dir)
    _, _, test_idx = load_splits(processed_dir)
    X_test = X[test_idx].astype(np.float32)[..., None]  # (N,1024,1)
    freqs_mhz = (freqs_hz / 1e6) if freqs_hz is not None else None

    # Build models
    input_bins = int(cfg.get("preprocess", {}).get("target_bins", 1024))
    latent_dim = 32
    include_dense128 = bool(cfg.get("train", {}).get("include_dense128", True))

    encoder = build_encoder(input_bins=input_bins, latent_dim=latent_dim, include_dense128=include_dense128)
    decoder = build_decoder(input_bins=input_bins, latent_dim=latent_dim)

    # Build once
    _ = encoder(tf.zeros((1, input_bins, 1), dtype=tf.float32))
    _ = decoder(tf.zeros((1, latent_dim), dtype=tf.float32))

    # Load weights
    encoder.load_weights(enc_w)
    decoder.load_weights(dec_w)

    # Inference in batches
    batch_size = int(cfg.get("train", {}).get("batch_size", 256))
    ds = tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    mu_all, logvar_all, xhat_all = [], [], []
    for xb in ds:
        mu, logvar = encoder(xb, training=False)
        x_hat = decoder(mu, training=False)  # deterministic z≈mu
        mu_all.append(mu.numpy())
        logvar_all.append(logvar.numpy())
        xhat_all.append(x_hat.numpy())

    mu_all = np.concatenate(mu_all, axis=0)
    logvar_all = np.concatenate(logvar_all, axis=0)
    X_hat = np.concatenate(xhat_all, axis=0)

    # Metrics
    recon = recon_loss_sum_mse(X_test, X_hat)
    kl = kl_diag_gaussian(mu_all, logvar_all)
    beta_report = float(cfg.get("train", {}).get("beta_final", 1.0))
    total = recon + beta_report * kl
    pk_mse = topk_peak_mse(X_test, X_hat, k=args.topk)
    pk_bias = peak_bias(X_test, X_hat)

    metrics = {
        "tag": tag,
        "weights": {"encoder": str(enc_w), "decoder": str(dec_w)},
        "data": {
            "processed_dir": str(processed_dir),
            "npz": meta.get("npz_path"),
            "X_test_shape": list(X_test.shape),
            "normalize_meta": meta,
        },
        "losses": {
            "beta_used_for_reporting": beta_report,
            "recon_loss": recon,
            "kl_loss": kl,
            "total_loss": total,
        },
        "peak_metrics": {
            "topk": int(args.topk),
            "topk_peak_mse": pk_mse,
            "peak_bias_mean_recon_minus_orig": pk_bias,
        }
    }

    # Output dir
    eval_dir = run_dir / "eval"
    ensure_dir(eval_dir)

    safe_write_json(eval_dir / "metrics.json", metrics)
    print("[EVAL] Saved metrics:", eval_dir / "metrics.json")

    # Plots
    plot_overlays(eval_dir / "recon_random.png", X_test, X_hat,
                  f"{tag} — random recon overlays", freqs_mhz=freqs_mhz, n=args.n_plots)

    # Peaky overlays: top samples by max(orig)
    mx = X_test.max(axis=1).squeeze(-1)
    top_idx = np.argsort(-mx)[:min(args.n_plots, len(mx))]
    plot_overlays(eval_dir / "recon_peaky.png", X_test[top_idx], X_hat[top_idx],
                  f"{tag} — peaky overlays (top max)", freqs_mhz=freqs_mhz, n=len(top_idx))

    plot_hist(eval_dir / "hist_values.png", X_test, f"{tag} — histogram (test)")
    plot_waterfall(eval_dir / "waterfall_orig.png", X_test, f"{tag} — waterfall (orig)")
    plot_waterfall(eval_dir / "waterfall_recon.png", X_hat, f"{tag} — waterfall (recon)")

    print("[EVAL] Plots saved in:", eval_dir)
    print("[EVAL] recon_loss:", recon, "kl_loss:", kl, "total:", total)
    print("[EVAL] topk_peak_mse:", pk_mse, "peak_bias:", pk_bias)


if __name__ == "__main__":
    main()