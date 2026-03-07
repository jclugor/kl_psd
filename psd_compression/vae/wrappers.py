from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

from psd_compression.common.io import repo_root

VAE_SCRIPT_MAP = {
    "preprocess": "VAE_implementation/scripts/01_preprocess.py",
    "train": "VAE_implementation/scripts/02_train.py",
    "eval": "VAE_implementation/scripts/03_eval.py",
    "export": "VAE_implementation/scripts/04_export_tflite.py",
    "entropy": "VAE_implementation/scripts/05_entropy_stats.py",
    "benchmark": "VAE_implementation/scripts/06_codec_benchmark.py",
}


def _normalize_script_args(script_args: List[str]) -> List[str]:
    if script_args and script_args[0] == "--":
        return script_args[1:]
    return script_args


def run_vae_task(task: str, script_args: List[str], dry_run: bool = False) -> dict:
    if task not in VAE_SCRIPT_MAP:
        raise ValueError(f"Unsupported VAE task: {task}")

    script_path: Path = repo_root() / VAE_SCRIPT_MAP[task]
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

