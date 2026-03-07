from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_cli_help_smoke() -> None:
    proc = run_cmd([sys.executable, "-m", "psd_compression.cli", "--help"])
    assert proc.returncode == 0, proc.stderr
    assert "Unified PSD compression task runner" in proc.stdout


def test_fwht_dry_run_smoke() -> None:
    proc = run_cmd(
        [
            sys.executable,
            "-m",
            "psd_compression.cli",
            "fwht",
            "evaluate",
            "--config",
            "configs/fwht_default.yaml",
            "--max-frames",
            "2",
            "--dry-run",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert '"dry_run": true' in proc.stdout.lower()


def test_vae_wrapper_dry_run_smoke() -> None:
    proc = run_cmd(
        [
            sys.executable,
            "-m",
            "psd_compression.cli",
            "vae",
            "preprocess",
            "--dry-run",
            "--",
            "--config",
            "VAE_implementation/configs/vae_default.yaml",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert "01_preprocess.py" in proc.stdout


def test_kl_pca_dry_run_smoke() -> None:
    proc = run_cmd(
        [
            sys.executable,
            "-m",
            "psd_compression.cli",
            "kl-pca",
            "evaluate",
            "--config",
            "configs/kl_pca_default.yaml",
            "--max-frames",
            "4",
            "--dry-run",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert '"dry_run": true' in proc.stdout.lower()
