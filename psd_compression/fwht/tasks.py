from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, cast

import numpy as np

from psd_compression.common.io import (
    load_psd_dataset,
    load_yaml_config,
    resolve_repo_path,
)
from psd_compression.fwht.codec import (
    FWHTConfig,
    FWHTPacket,
    NonlinearMode,
    StandardizationSideInfo,
    compress_fwht_frame,
    decompress_fwht_frame,
    estimate_payload_bits,
    reconstruction_metrics,
)


def _build_fwht_config(cfg: Dict[str, Any]) -> FWHTConfig:
    codec = cfg.get("codec", {})
    top_k_coeffs = int(codec.get("top_k_coeffs", 128))
    if top_k_coeffs < 1:
        raise ValueError("codec.top_k_coeffs must be >= 1")
    nonlinear_mode = cast(
        NonlinearMode,
        str(codec.get("nonlinear_mode", "signed_log1p")),
    )

    return FWHTConfig(
        decimation_factor_bins=int(codec.get("decimation_factor_bins", 2)),
        top_k_coeffs=top_k_coeffs,
        quant_step=float(codec.get("quant_step", 0.02)),
        nonlinear_mode=nonlinear_mode,
        nonlinear_alpha=float(codec.get("nonlinear_alpha", 1.5)),
        side_info_bits_per_param=int(codec.get("side_info_bits_per_param", 16)),
        value_bits_per_coeff=int(codec.get("value_bits_per_coeff", 16)),
    )


def _dataset_path(cfg: Dict[str, Any]) -> Path:
    return resolve_repo_path(
        cfg.get("dataset", {}).get(
            "path", "data/processed/psd_1024/dataset_psd_1024_norm.npz"
        )
    )


def _save_packet(packet: FWHTPacket, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        original_length_bins=np.int64(packet.original_length_bins),
        decimated_length_bins=np.int64(packet.decimated_length_bins),
        hadamard_length_bins=np.int64(packet.hadamard_length_bins),
        side_mean=np.float64(packet.side_info.mean),
        side_std=np.float64(packet.side_info.std),
        topk_indices=packet.topk_indices.astype(np.int64),
        quantized_values=packet.quantized_values.astype(np.int32),
        quant_step=np.float64(packet.quant_step),
        nonlinear_mode=np.array(packet.nonlinear_mode),
        nonlinear_alpha=np.float64(packet.nonlinear_alpha),
    )


def _load_packet(packet_path: Path) -> FWHTPacket:
    data = np.load(packet_path, allow_pickle=True)
    nonlinear_mode = cast(
        NonlinearMode,
        str(
            data["nonlinear_mode"].item()
            if hasattr(data["nonlinear_mode"], "item")
            else data["nonlinear_mode"]
        ),
    )
    return FWHTPacket(
        original_length_bins=int(data["original_length_bins"]),
        decimated_length_bins=int(data["decimated_length_bins"]),
        hadamard_length_bins=int(data["hadamard_length_bins"]),
        side_info=StandardizationSideInfo(
            mean=float(data["side_mean"]), std=float(data["side_std"])
        ),
        topk_indices=np.asarray(data["topk_indices"], dtype=np.int64),
        quantized_values=np.asarray(data["quantized_values"], dtype=np.int32),
        quant_step=float(data["quant_step"]),
        nonlinear_mode=nonlinear_mode,
        nonlinear_alpha=float(data["nonlinear_alpha"]),
    )


def run_encode(
    config_path: str | Path,
    frame_index: int,
    output_path: str | Path,
    dry_run: bool = False,
) -> dict:
    cfg = load_yaml_config(config_path)
    config = _build_fwht_config(cfg)
    dataset_path = _dataset_path(cfg)
    output = resolve_repo_path(output_path)

    if dry_run:
        return {
            "dry_run": True,
            "dataset_path": str(dataset_path),
            "frame_index": int(frame_index),
            "output_path": str(output),
            "config": config.__dict__,
        }

    frames_psd, _, _ = load_psd_dataset(dataset_path)
    idx = int(frame_index)
    if idx < 0 or idx >= frames_psd.shape[0]:
        raise IndexError(
            f"frame_index out of range: {idx} for {frames_psd.shape[0]} frames"
        )

    packet = compress_fwht_frame(frames_psd[idx], config)
    payload_bits = estimate_payload_bits(packet, config)
    _save_packet(packet, output)

    return {
        "frame_index": idx,
        "output_path": str(output),
        "payload_bits": int(payload_bits),
        "top_k_coeffs": int(packet.topk_indices.size),
    }


def run_decode(
    packet_path: str | Path, output_path: str | Path, dry_run: bool = False
) -> dict:
    packet_file = resolve_repo_path(packet_path)
    output = resolve_repo_path(output_path)
    if dry_run:
        return {
            "dry_run": True,
            "packet_path": str(packet_file),
            "output_path": str(output),
        }

    packet = _load_packet(packet_file)
    reconstructed = decompress_fwht_frame(packet)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, reconstructed.astype(np.float64))
    return {"output_path": str(output), "num_bins": int(reconstructed.size)}


def run_evaluate(
    config_path: str | Path,
    max_frames: int | None = None,
    output_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict:
    cfg = load_yaml_config(config_path)
    config = _build_fwht_config(cfg)
    dataset_path = _dataset_path(cfg)
    occ_margin = float(cfg.get("evaluation", {}).get("occupancy_margin_db", 3.0))
    default_max = int(cfg.get("evaluation", {}).get("max_frames", 64))
    n_frames = int(max_frames if max_frames is not None else default_max)
    n_frames = max(1, n_frames)

    report_path = (
        resolve_repo_path(output_path)
        if output_path
        else resolve_repo_path(
            cfg.get("outputs", {}).get(
                "report_path", "data/processed/psd_1024/fwht_eval_report.json"
            )
        )
    )

    if dry_run:
        return {
            "dry_run": True,
            "dataset_path": str(dataset_path),
            "max_frames": n_frames,
            "report_path": str(report_path),
            "config": config.__dict__,
        }

    frames_psd, _, metadata = load_psd_dataset(dataset_path)
    n_use = min(n_frames, frames_psd.shape[0])

    metric_rows = []
    payload_bits = []
    for idx in range(n_use):
        original = frames_psd[idx]
        packet = compress_fwht_frame(original, config)
        reconstructed = decompress_fwht_frame(packet)
        row = reconstruction_metrics(
            original, reconstructed, occupancy_margin_db=occ_margin
        )
        metric_rows.append(row)
        payload_bits.append(estimate_payload_bits(packet, config))

    mse_avg = float(np.mean([r["mse"] for r in metric_rows]))
    nmse_avg = float(np.mean([r["nmse"] for r in metric_rows]))
    snr_avg = float(np.mean([r["snr_db"] for r in metric_rows]))
    occ_avg = float(np.mean([r["occupancy_mismatch"] for r in metric_rows]))
    bits_avg = float(np.mean(payload_bits))

    report = {
        "dataset_path": metadata["dataset_path"],
        "frames_evaluated": int(n_use),
        "avg_bits_per_frame": bits_avg,
        "avg_mse": mse_avg,
        "avg_nmse": nmse_avg,
        "avg_snr_db": snr_avg,
        "avg_occupancy_mismatch": occ_avg,
        "codec": config.__dict__,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
