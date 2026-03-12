#!/usr/bin/env python3
"""Compare saved PSD reconstructions and materialize plots plus summary metrics.

Uso (desde raiz del repo):
  python VAE_implementation/scripts/analysis/compare_npy_metrics_plot.py

Opcional:
  python .../compare_npy_metrics_plot.py --orig path/orig.npy --recon path/recon.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_metrics(orig: np.ndarray, recon: np.ndarray) -> dict:
    """Compute scalar reconstruction metrics for two PSD vectors."""

    err = recon - orig
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    max_abs = float(np.max(np.abs(orig))) if orig.size else 0.0
    nrmse = float(rmse / (max_abs + 1e-12))

    sig_pow = float(np.mean(orig**2))
    noise_pow = float(np.mean((orig - recon) ** 2))
    snr_db = float(10.0 * np.log10((sig_pow + 1e-12) / (noise_pow + 1e-12)))

    data_range = float(np.max(orig) - np.min(orig))
    psnr_db = float(20.0 * np.log10((data_range + 1e-12) / (rmse + 1e-12)))

    if orig.size > 1 and np.std(orig) > 0 and np.std(recon) > 0:
        corr = float(np.corrcoef(orig, recon)[0, 1])
    else:
        corr = float("nan")

    return {
        "len": int(orig.size),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "nrmse": nrmse,
        "snr_db": snr_db,
        "psnr_db": psnr_db,
        "corrcoef": corr,
    }


def save_plots(orig: np.ndarray, recon: np.ndarray, out_dir: Path) -> None:
    """Persist overlay, error-trace, and histogram plots for one comparison."""

    err = recon - orig
    x = np.arange(orig.size)

    # Persist the most informative visualizations close to the compared arrays.
    plt.figure(figsize=(12, 4))
    plt.plot(x, orig, label="orig", linewidth=1.5)
    plt.plot(x, recon, label="recon", linewidth=1.2, alpha=0.9)
    plt.title("PSD: Original vs Reconstruccion")
    plt.xlabel("Bin")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_overlay.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 3))
    plt.plot(x, err, label="error (recon-orig)", color="tab:red", linewidth=1.0)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("Error por Bin")
    plt.xlabel("Bin")
    plt.ylabel("Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_error.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(err, bins=40, color="tab:blue", alpha=0.85)
    plt.title("Histograma del Error")
    plt.xlabel("Error")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "compare_error_hist.png", dpi=180)
    plt.close()


def main() -> None:
    """Parse CLI arguments, compare the PSD vectors, and write artifacts."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default=None, help="Path a orig .npy")
    ap.add_argument("--recon", default=None, help="Path a recon .npy")
    ap.add_argument("--out_dir", default=None, help="Directorio de salida")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    orig_path = Path(args.orig) if args.orig else (script_dir / "orig_last_psd.npy")
    recon_path = Path(args.recon) if args.recon else (script_dir / "recon_last_psd.npy")
    out_dir = Path(args.out_dir) if args.out_dir else script_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not orig_path.exists():
        raise FileNotFoundError(f"orig npy no encontrado: {orig_path}")
    if not recon_path.exists():
        raise FileNotFoundError(f"recon npy no encontrado: {recon_path}")

    # Compare both arrays over the common support to avoid shape-related crashes.
    orig = np.load(orig_path).astype(np.float64).reshape(-1)
    recon = np.load(recon_path).astype(np.float64).reshape(-1)
    n = min(orig.size, recon.size)
    orig = orig[:n]
    recon = recon[:n]

    metrics = compute_metrics(orig, recon)
    save_plots(orig, recon, out_dir)

    metrics_path = out_dir / "compare_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("[COMPARE] orig:", orig_path)
    print("[COMPARE] recon:", recon_path)
    print("[COMPARE] out_dir:", out_dir)
    print("[COMPARE] metrics:", metrics_path)
    for k, v in metrics.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
