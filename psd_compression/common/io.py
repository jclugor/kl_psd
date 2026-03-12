"""Repository-aware I/O helpers for configs and PSD datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _is_repository_root(candidate: Path) -> bool:
    """Return whether ``candidate`` looks like the KL PSD repository root."""

    return (candidate / "pyproject.toml").is_file() and (
        candidate / "psd_compression"
    ).is_dir()


def _find_repository_root(start: Path) -> Path | None:
    """Search ``start`` and its parents for the repository root marker set."""

    for candidate in (start, *start.parents):
        if _is_repository_root(candidate):
            return candidate
    return None


def repo_root() -> Path:
    """Return the best available project root for repository-relative paths.

    Resolution order:
    1. ``KL_PSD_ROOT`` environment override.
    2. Search upward from this module location.
    3. Search upward from the current working directory.

    The final fallback is the current working directory so installed-package
    workflows can still resolve user-provided relative paths from the shell.
    """

    env_root = os.environ.get("KL_PSD_ROOT")
    if env_root is not None:
        configured_root = Path(env_root).expanduser().resolve()
        if not _is_repository_root(configured_root):
            raise RuntimeError(
                f"KL_PSD_ROOT does not point to a valid repository root: {configured_root}"
            )
        return configured_root

    module_root = _find_repository_root(Path(__file__).resolve().parent)
    if module_root is not None:
        return module_root

    cwd_root = _find_repository_root(Path.cwd())
    if cwd_root is not None:
        return cwd_root

    return Path.cwd().resolve()


def resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve a path against the working directory, then the project root."""

    path = Path(path_like)
    if path.is_absolute():
        return path

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path

    return (repo_root() / path).resolve()


def load_yaml_config(path_like: str | Path) -> dict[str, Any]:
    """Load one YAML config file and require a mapping-valued document."""

    path = resolve_repo_path(path_like)
    with path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj)

    if payload is None:
        raise ValueError(f"YAML config is empty: {path}")
    if not isinstance(payload, dict):
        raise TypeError(f"YAML config must decode to a mapping: {path}")
    return payload


def load_psd_dataset(
    npz_path_like: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a PSD dataset NPZ and validate the expected matrix contract.

    Parameters
    ----------
    npz_path_like:
        Dataset path. Relative paths are resolved against the current working
        directory first and then against the repository root.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, Any]]
        PSD frame matrix ``X`` with shape ``[num_frames, num_bins]``, frequency
        axis in Hz, and metadata derived from the NPZ payload.
    """

    npz_path = resolve_repo_path(npz_path_like)
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset NPZ not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as npz:
        if "X" not in npz:
            raise KeyError(f"Expected key `X` in NPZ. Found keys: {list(npz.keys())}")

        frames_psd = np.asarray(npz["X"], dtype=np.float64)
        if frames_psd.ndim != 2:
            raise ValueError(
                f"Expected `X` to have shape [num_frames, num_bins], got {frames_psd.shape}"
            )
        if frames_psd.shape[0] < 1 or frames_psd.shape[1] < 1:
            raise ValueError(
                f"Dataset `X` must contain at least one frame and one bin, got {frames_psd.shape}"
            )

        if "freqs_hz" in npz:
            freqs_hz = np.asarray(npz["freqs_hz"], dtype=np.float64)
            if freqs_hz.ndim != 1 or freqs_hz.shape[0] != frames_psd.shape[1]:
                raise ValueError(
                    "Expected `freqs_hz` to be one-dimensional and aligned with the PSD bin axis"
                )
        else:
            freqs_hz = np.arange(frames_psd.shape[1], dtype=np.float64)

        metadata: dict[str, Any] = {}
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
