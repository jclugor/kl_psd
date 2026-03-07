from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml


def repo_root() -> Path:
    """Return repository root from package location."""
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve absolute path with repository root as base for relative paths."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return repo_root() / path


def load_yaml_config(path_like: str | Path) -> Dict[str, Any]:
    """Load YAML config and return dict."""
    path = resolve_repo_path(path_like)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_psd_dataset(npz_path_like: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load PSD dataset from NPZ with expected `X` and optional metadata fields."""
    npz_path = resolve_repo_path(npz_path_like)
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset NPZ not found: {npz_path}")

    npz = np.load(npz_path, allow_pickle=True)
    if "X" not in npz:
        raise KeyError(f"Expected key `X` in NPZ. Found keys: {list(npz.keys())}")

    frames_psd = np.asarray(npz["X"], dtype=np.float64)
    freqs_hz = np.asarray(npz["freqs_hz"], dtype=np.float64) if "freqs_hz" in npz else np.arange(frames_psd.shape[1], dtype=np.float64)

    metadata: Dict[str, Any] = {}
    for key in ("gmin", "gmax", "normalize_mode"):
        if key in npz:
            value = npz[key]
            if isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
            metadata[key] = value

    metadata["dataset_path"] = str(npz_path)
    metadata["num_frames"] = int(frames_psd.shape[0])
    metadata["num_bins"] = int(frames_psd.shape[1])
    return frames_psd, freqs_hz, metadata

