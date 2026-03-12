from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from psd_compression.common.io import (
    load_psd_dataset,
    load_yaml_config,
    resolve_repo_path,
)
from psd_compression.common.metrics import mse, nmse, occupancy_mismatch_rate, snr_db
from psd_compression.kl_pca.model import (
    KLPCAConfig,
    KLPCAModel,
    decode_coefficients,
    encode_frame,
    fit_kl_pca,
)


def _dataset_path(cfg: Dict[str, Any]) -> Path:
    return resolve_repo_path(
        cfg.get("dataset", {}).get(
            "path", "data/processed/psd_1024/dataset_psd_1024_norm.npz"
        )
    )


def _codec_config(cfg: Dict[str, Any]) -> KLPCAConfig:
    fit_cfg = cfg.get("fit", {})
    n_components = int(fit_cfg.get("n_components", 32))
    if n_components < 1:
        raise ValueError("fit.n_components must be >= 1")

    return KLPCAConfig(
        n_components=n_components,
        center=bool(fit_cfg.get("center", True)),
        enforce_nonnegative=bool(cfg.get("codec", {}).get("enforce_nonnegative", True)),
    )


def _save_model(model: KLPCAModel, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mean=model.mean.astype(np.float64),
        components=model.components.astype(np.float64),
        explained_variance_ratio=model.explained_variance_ratio.astype(np.float64),
    )


def _load_model(model_path: Path) -> KLPCAModel:
    data = np.load(model_path, allow_pickle=True)
    return KLPCAModel(
        mean=np.asarray(data["mean"], dtype=np.float64),
        components=np.asarray(data["components"], dtype=np.float64),
        explained_variance_ratio=np.asarray(
            data["explained_variance_ratio"], dtype=np.float64
        ),
    )


def run_fit(
    config_path: str | Path,
    output_path: str | Path | None = None,
    max_frames: int | None = None,
    dry_run: bool = False,
) -> dict:
    cfg = load_yaml_config(config_path)
    ds_path = _dataset_path(cfg)
    out_path = (
        resolve_repo_path(output_path)
        if output_path
        else resolve_repo_path(
            cfg.get("outputs", {}).get(
                "model_path", "data/processed/psd_1024/kl_pca_model.npz"
            )
        )
    )
    codec_cfg = _codec_config(cfg)
    n_frames = int(
        max_frames
        if max_frames is not None
        else cfg.get("fit", {}).get("max_frames", 512)
    )
    n_frames = max(2, n_frames)

    if dry_run:
        return {
            "dry_run": True,
            "dataset_path": str(ds_path),
            "output_path": str(out_path),
            "n_components": int(codec_cfg.n_components),
            "max_frames": int(n_frames),
        }

    frames, _, meta = load_psd_dataset(ds_path)
    n_use = min(n_frames, frames.shape[0])
    model = fit_kl_pca(
        frames[:n_use], n_components=codec_cfg.n_components, center=codec_cfg.center
    )
    _save_model(model, out_path)
    return {
        "dataset_path": meta["dataset_path"],
        "frames_used": int(n_use),
        "output_path": str(out_path),
        "n_components": int(model.components.shape[0]),
        "explained_variance_sum": float(np.sum(model.explained_variance_ratio)),
    }


def run_encode(
    config_path: str | Path,
    model_path: str | Path | None = None,
    frame_index: int = 0,
    output_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict:
    cfg = load_yaml_config(config_path)
    ds_path = _dataset_path(cfg)
    mdl_path = (
        resolve_repo_path(model_path)
        if model_path
        else resolve_repo_path(
            cfg.get("outputs", {}).get(
                "model_path", "data/processed/psd_1024/kl_pca_model.npz"
            )
        )
    )
    out_path = (
        resolve_repo_path(output_path)
        if output_path
        else resolve_repo_path(
            cfg.get("outputs", {}).get(
                "coeff_path", "data/processed/psd_1024/kl_pca_coeff_frame0.npy"
            )
        )
    )

    if dry_run:
        return {
            "dry_run": True,
            "dataset_path": str(ds_path),
            "model_path": str(mdl_path),
            "frame_index": int(frame_index),
            "output_path": str(out_path),
        }

    model = _load_model(mdl_path)
    frames, _, _ = load_psd_dataset(ds_path)
    idx = int(frame_index)
    if idx < 0 or idx >= frames.shape[0]:
        raise IndexError(
            f"frame_index out of range: {idx} for {frames.shape[0]} frames"
        )
    coeffs = encode_frame(frames[idx], model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, coeffs.astype(np.float64))
    return {
        "frame_index": idx,
        "num_coeffs": int(coeffs.size),
        "output_path": str(out_path),
    }


def run_decode(
    config_path: str | Path,
    model_path: str | Path | None = None,
    coeff_path: str | Path | None = None,
    output_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict:
    cfg = load_yaml_config(config_path)
    mdl_path = (
        resolve_repo_path(model_path)
        if model_path
        else resolve_repo_path(
            cfg.get("outputs", {}).get(
                "model_path", "data/processed/psd_1024/kl_pca_model.npz"
            )
        )
    )
    c_path = (
        resolve_repo_path(coeff_path)
        if coeff_path
        else resolve_repo_path(
            cfg.get("outputs", {}).get(
                "coeff_path", "data/processed/psd_1024/kl_pca_coeff_frame0.npy"
            )
        )
    )
    out_path = (
        resolve_repo_path(output_path)
        if output_path
        else resolve_repo_path(
            cfg.get("outputs", {}).get(
                "decode_path",
                "data/processed/psd_1024/kl_pca_reconstruction_frame0.npy",
            )
        )
    )
    enforce_nonnegative = bool(cfg.get("codec", {}).get("enforce_nonnegative", True))

    if dry_run:
        return {
            "dry_run": True,
            "model_path": str(mdl_path),
            "coeff_path": str(c_path),
            "output_path": str(out_path),
        }

    model = _load_model(mdl_path)
    coeffs = np.load(c_path, allow_pickle=True)
    recon = decode_coefficients(coeffs, model, enforce_nonnegative=enforce_nonnegative)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, recon.astype(np.float64))
    return {"num_bins": int(recon.size), "output_path": str(out_path)}


def run_evaluate(
    config_path: str | Path,
    model_path: str | Path | None = None,
    max_frames: int | None = None,
    output_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict:
    cfg = load_yaml_config(config_path)
    ds_path = _dataset_path(cfg)
    mdl_path = (
        resolve_repo_path(model_path)
        if model_path
        else resolve_repo_path(
            cfg.get("outputs", {}).get(
                "model_path", "data/processed/psd_1024/kl_pca_model.npz"
            )
        )
    )
    out_path = (
        resolve_repo_path(output_path)
        if output_path
        else resolve_repo_path(
            cfg.get("outputs", {}).get(
                "report_path", "data/processed/psd_1024/kl_pca_eval_report.json"
            )
        )
    )
    n_frames = int(
        max_frames
        if max_frames is not None
        else cfg.get("evaluation", {}).get("max_frames", 64)
    )
    n_frames = max(1, n_frames)
    bits_per_coeff = int(cfg.get("codec", {}).get("bits_per_coeff", 16))
    enforce_nonnegative = bool(cfg.get("codec", {}).get("enforce_nonnegative", True))
    occ_margin = float(cfg.get("evaluation", {}).get("occupancy_margin_db", 3.0))

    if dry_run:
        return {
            "dry_run": True,
            "dataset_path": str(ds_path),
            "model_path": str(mdl_path),
            "max_frames": int(n_frames),
            "report_path": str(out_path),
        }

    model = _load_model(mdl_path)
    frames, _, meta = load_psd_dataset(ds_path)
    n_use = min(n_frames, frames.shape[0])
    m_rows = []
    for i in range(n_use):
        x = frames[i]
        coeffs = encode_frame(x, model)
        recon = decode_coefficients(
            coeffs, model, enforce_nonnegative=enforce_nonnegative
        )
        m_rows.append(
            {
                "mse": mse(x, recon),
                "nmse": nmse(x, recon),
                "snr_db": snr_db(x, recon),
                "occupancy_mismatch": occupancy_mismatch_rate(
                    x, recon, margin_db=occ_margin
                ),
            }
        )

    report = {
        "dataset_path": meta["dataset_path"],
        "frames_evaluated": int(n_use),
        "avg_bits_per_frame": float(model.components.shape[0] * bits_per_coeff),
        "avg_mse": float(np.mean([r["mse"] for r in m_rows])),
        "avg_nmse": float(np.mean([r["nmse"] for r in m_rows])),
        "avg_snr_db": float(np.mean([r["snr_db"] for r in m_rows])),
        "avg_occupancy_mismatch": float(
            np.mean([r["occupancy_mismatch"] for r in m_rows])
        ),
        "n_components": int(model.components.shape[0]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(out_path)
    return report
