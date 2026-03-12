"""Compatibility adapters for the legacy VAE script tree."""

from __future__ import annotations

from collections.abc import Sequence
import shlex
import subprocess
import sys
from pathlib import Path

from psd_compression.common.io import repo_root


VAE_SCRIPT_RELATIVE_PATHS: dict[str, Path] = {
    "preprocess": Path("VAE_implementation/scripts/training/01_preprocess.py"),
    "train": Path("VAE_implementation/scripts/training/02_train.py"),
    "eval": Path("VAE_implementation/scripts/training/03_eval.py"),
    "export": Path("VAE_implementation/scripts/training/04_export_tflite.py"),
    "entropy": Path("VAE_implementation/scripts/analysis/05_entropy_stats.py"),
    "benchmark": Path("VAE_implementation/scripts/analysis/06_codec_benchmark.py"),
}


def _normalize_script_args(script_args: Sequence[str]) -> list[str]:
    """Drop the passthrough separator used by the unified CLI wrapper."""

    if script_args and script_args[0] == "--":
        return list(script_args[1:])
    return list(script_args)


def run_vae_task(
    task: str,
    script_args: Sequence[str],
    dry_run: bool = False,
) -> dict:
    """Dispatch one VAE CLI task to the legacy script implementation.

    Parameters
    ----------
    task:
        Named VAE task exposed by ``python -m psd_compression.cli vae``.
    script_args:
        Extra CLI arguments forwarded to the selected legacy script.
    dry_run:
        When ``True``, return the resolved command without executing it.

    Returns
    -------
    dict
        Structured execution metadata for the selected task.

    Side Effects
    ------------
    Executes the legacy VAE Python script in a subprocess when ``dry_run`` is
    ``False``.
    """

    if task not in VAE_SCRIPT_RELATIVE_PATHS:
        raise ValueError(f"Unsupported VAE task: {task}")

    # Keep the legacy VAE tree stable on disk while exposing it through the
    # unified package CLI.
    script_path = repo_root() / VAE_SCRIPT_RELATIVE_PATHS[task]
    if not script_path.exists():
        raise FileNotFoundError(f"Legacy script not found: {script_path}")

    final_args = _normalize_script_args(script_args)
    command = [sys.executable, str(script_path), *final_args]

    if dry_run:
        return {
            "dry_run": True,
            "task": task,
            "script": str(script_path),
            "command": " ".join(shlex.quote(arg) for arg in command),
        }

    completed = subprocess.run(command, cwd=str(repo_root()), check=False)
    return {
        "task": task,
        "script": str(script_path),
        "return_code": int(completed.returncode),
        "command": " ".join(shlex.quote(arg) for arg in command),
    }
