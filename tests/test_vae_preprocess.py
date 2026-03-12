from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from psd_compression.vae.preprocess import (
    expected_dataset_path,
    load_preprocess_config,
    run_preprocess,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_csv_dataset(raw_dir: Path) -> None:
    """Create a minimal PSD CSV dataset compatible with the VAE preprocess step."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    frame_a = np.linspace(-30.0, -10.0, 8, dtype=np.float32).tolist()
    frame_b = np.linspace(-20.0, -5.0, 8, dtype=np.float32).tolist()
    dataframe = pd.DataFrame(
        [
            {
                "id": 1,
                "mac": "00:11:22:33:44:55",
                "campaign_id": 7,
                "pxx": str(frame_a).replace("'", '"'),
                "start_freq_hz": 88_000_000,
                "end_freq_hz": 108_000_000,
                "timestamp": 1_700_000_000_000,
                "created_at": 1_700_000_000_500,
            },
            {
                "id": 2,
                "mac": "00:11:22:33:44:55",
                "campaign_id": 7,
                "pxx": str(frame_b).replace("'", '"'),
                "start_freq_hz": 88_000_000,
                "end_freq_hz": 108_000_000,
                "timestamp": 1_700_000_001_000,
                "created_at": 1_700_000_001_500,
            },
        ]
    )
    dataframe.to_csv(raw_dir / "node.csv", index=False)


def _write_config(
    config_path: Path,
    raw_dir: Path,
    processed_dir: Path,
) -> None:
    """Write a minimal preprocess config pointing at temporary test folders."""

    payload = {
        "paths": {
            "raw_dataset_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
        },
        "preprocess": {
            "target_bins": 4,
            "normalize_mode": "global_minmax",
            "use_db_clip": False,
            "db_min": -120.0,
            "db_max": 10.0,
            "reuse_existing_processed_if_source_missing": False,
            "split": {
                "train": 0.5,
                "val": 0.0,
                "test": 0.5,
                "mode": "time_ordered",
                "seed": 2026,
            },
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_run_preprocess_builds_dataset_from_csv_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    config_path = tmp_path / "config.yaml"
    _write_csv_dataset(raw_dir)
    _write_config(config_path, raw_dir, processed_dir)

    result = run_preprocess(config_path)

    dataset_path = expected_dataset_path(processed_dir, target_bins=4)
    assert result.source_dir == raw_dir
    assert result.processed_dir == processed_dir
    assert result.dataset_path == dataset_path
    assert result.total_frames == 2
    assert result.target_bins == 4
    assert result.train_count == 1
    assert result.val_count == 0
    assert result.test_count == 1
    assert not result.used_existing_processed
    assert dataset_path.exists()
    assert (processed_dir / "metadata.csv").exists()
    assert (processed_dir / "splits" / "train_idx.npy").exists()

    data = np.load(dataset_path)
    assert data["X"].shape == (2, 4)
    assert data["freqs_hz"].shape == (4,)


def test_run_preprocess_reuses_existing_outputs_when_enabled(tmp_path: Path) -> None:
    raw_dir = tmp_path / "missing_raw"
    processed_dir = tmp_path / "processed"
    config_path = tmp_path / "config.yaml"
    fallback_config_path = tmp_path / "reuse.yaml"

    _write_csv_dataset(tmp_path / "initial_raw")
    _write_config(config_path, tmp_path / "initial_raw", processed_dir)
    first_result = run_preprocess(config_path)

    reuse_payload = {
        "paths": {
            "raw_dataset_dir": str(raw_dir),
            "processed_dir": str(processed_dir),
        },
        "preprocess": {
            "target_bins": 4,
            "normalize_mode": "global_minmax",
            "use_db_clip": False,
            "db_min": -120.0,
            "db_max": 10.0,
            "reuse_existing_processed_if_source_missing": True,
            "split": {
                "train": 0.5,
                "val": 0.0,
                "test": 0.5,
                "mode": "time_ordered",
                "seed": 2026,
            },
        },
    }
    fallback_config_path.write_text(yaml.safe_dump(reuse_payload), encoding="utf-8")

    result = run_preprocess(fallback_config_path)

    assert result.used_existing_processed
    assert result.source_dir is None
    assert result.dataset_path == first_result.dataset_path
    assert result.total_frames == first_result.total_frames


def test_preprocess_script_runs_end_to_end_via_legacy_entrypoint(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    config_path = tmp_path / "config.yaml"
    _write_csv_dataset(raw_dir)
    _write_config(config_path, raw_dir, processed_dir)

    process = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "VAE_implementation/scripts/training/01_preprocess.py"),
            "--config",
            str(config_path),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert process.returncode == 0, process.stderr
    assert "[DONE] Preprocess complete." in process.stdout
    assert expected_dataset_path(processed_dir, target_bins=4).exists()


def test_load_preprocess_config_resolves_fallback_dirs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()

    payload = {
        "paths": {
            "raw_dataset_dir": str(tmp_path / "raw"),
            "fallback_raw_dataset_dirs": [str(fallback_dir)],
            "processed_dir": str(tmp_path / "processed"),
        },
        "preprocess": {
            "target_bins": 4,
            "normalize_mode": "global_minmax",
            "use_db_clip": False,
            "db_min": -120.0,
            "db_max": 10.0,
            "reuse_existing_processed_if_source_missing": False,
            "split": {
                "train": 0.5,
                "val": 0.0,
                "test": 0.5,
                "mode": "time_ordered",
                "seed": 2026,
            },
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_preprocess_config(config_path)

    assert config.fallback_raw_dataset_dirs == (fallback_dir,)
