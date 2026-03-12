#!/usr/bin/env python3
"""Compatibility wrapper for the relocated PSD comparison utility."""

from __future__ import annotations

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np


def _load_compare_module():
    """Load the canonical comparison script from the analysis folder."""

    repo_root = Path(__file__).resolve().parents[4]
    module_path = (
        repo_root / "VAE_implementation/scripts/analysis/compare_npy_metrics_plot.py"
    )
    spec = spec_from_file_location("vae_compare_metrics_plot", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load compare utility from {module_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Preserve the legacy CLI path while delegating to the canonical script."""

    compare_module = _load_compare_module()

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

    orig = np.load(orig_path).astype(np.float64).reshape(-1)
    recon = np.load(recon_path).astype(np.float64).reshape(-1)
    n = min(orig.size, recon.size)
    orig = orig[:n]
    recon = recon[:n]

    metrics = compare_module.compute_metrics(orig, recon)
    compare_module.save_plots(orig, recon, out_dir)

    metrics_path = out_dir / "compare_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[COMPARE] orig:", orig_path)
    print("[COMPARE] recon:", recon_path)
    print("[COMPARE] out_dir:", out_dir)
    print("[COMPARE] metrics:", metrics_path)
    for key, value in metrics.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
