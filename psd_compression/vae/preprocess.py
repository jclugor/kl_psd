"""VAE dataset preprocessing helpers for CSV-based PSD acquisitions.

This module keeps the numerical transformation logic separate from the legacy
CLI script so the preprocessing contract can be tested without depending on the
numbered script path under ``VAE_implementation/scripts/training``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from psd_compression.common.io import load_psd_dataset, load_yaml_config, repo_root


_SUPPORTED_NORMALIZE_MODES = {"global_minmax", "per_frame_minmax", "none"}
_SUPPORTED_SPLIT_MODES = {"time_ordered", "random"}


@dataclass(frozen=True)
class SplitConfig:
    """Dataset split policy for the preprocessed PSD matrix."""

    train_ratio: float  # Fraction of frames assigned to the training split
    val_ratio: float  # Fraction of frames assigned to the validation split
    test_ratio: float  # Fraction of frames assigned to the test split
    mode: str  # Either ``time_ordered`` or ``random``
    seed: int  # RNG seed used when ``mode == "random"``


@dataclass(frozen=True)
class PreprocessConfig:
    """Resolved preprocessing configuration with repository-aware paths."""

    raw_dataset_dir: Path  # Primary directory that should contain raw CSV files
    fallback_raw_dataset_dirs: tuple[
        Path, ...
    ]  # Secondary CSV roots if the primary one is missing
    processed_dir: Path  # Output directory for the processed dataset and split indices
    target_bins: int  # Number of PSD bins in the output dataset
    normalize_mode: str  # Normalization policy applied to each frame
    use_db_clip: bool  # Whether to clamp PSD values before normalization
    db_min: float  # Lower clipping bound [dB]
    db_max: float  # Upper clipping bound [dB]
    split: SplitConfig  # Train/validation/test splitting policy
    reuse_existing_processed_if_source_missing: (
        bool  # Allow a safe no-op when canonical outputs already exist
    )


@dataclass(frozen=True)
class PreprocessRunResult:
    """Summary of one preprocessing execution."""

    source_dir: (
        Path | None
    )  # Directory that provided raw CSV files, or ``None`` when existing outputs were reused
    processed_dir: Path  # Directory containing the output dataset
    dataset_path: Path  # NPZ dataset path
    metadata_path: Path  # Metadata CSV path
    total_frames: int  # Number of frames in the processed dataset
    target_bins: int  # Number of output PSD bins
    train_count: int  # Number of frames in the training split
    val_count: int  # Number of frames in the validation split
    test_count: int  # Number of frames in the test split
    used_existing_processed: bool  # Whether preprocessing was skipped in favor of an existing processed dataset


@dataclass(frozen=True)
class _SourceSelection:
    """Internal source selection result used by the orchestration boundary."""

    source_dir: Path | None
    csv_files: tuple[Path, ...]
    used_existing_processed: bool


def _resolve_project_path(
    project_root: Path,
    path_value: str | Path,
) -> Path:
    """Resolve one config path against the working directory and project root."""

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (project_root / path).resolve()


def _require_mapping(
    mapping: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """Return one mapping-valued config section or raise a precise error."""

    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Config section `{key}` must be a mapping.")
    return value


def _parse_fallback_dirs(
    project_root: Path,
    raw_value: Any,
) -> tuple[Path, ...]:
    """Parse the optional fallback dataset directory list."""

    if raw_value is None:
        return ()
    if not isinstance(raw_value, list) or not all(
        isinstance(item, str) for item in raw_value
    ):
        raise TypeError(
            "Config key `paths.fallback_raw_dataset_dirs` must be a list of paths."
        )
    return tuple(_resolve_project_path(project_root, item) for item in raw_value)


def _validate_split_config(split: SplitConfig) -> None:
    """Validate split ratios and mode before any filesystem work starts."""

    if split.mode not in _SUPPORTED_SPLIT_MODES:
        raise ValueError(
            f"Unsupported split mode `{split.mode}`. Expected one of {sorted(_SUPPORTED_SPLIT_MODES)}."
        )

    ratios = (split.train_ratio, split.val_ratio, split.test_ratio)
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")

    if not np.isclose(sum(ratios), 1.0, atol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {split.train_ratio + split.val_ratio + split.test_ratio:.6f}."
        )


def _validate_preprocess_config(config: PreprocessConfig) -> None:
    """Validate the full preprocessing configuration."""

    if config.target_bins < 1:
        raise ValueError("`preprocess.target_bins` must be >= 1.")
    if config.normalize_mode not in _SUPPORTED_NORMALIZE_MODES:
        raise ValueError(
            "Unsupported `preprocess.normalize_mode`. "
            f"Expected one of {sorted(_SUPPORTED_NORMALIZE_MODES)}."
        )
    if config.db_min >= config.db_max:
        raise ValueError(
            "`preprocess.db_min` must be smaller than `preprocess.db_max`."
        )

    _validate_split_config(config.split)


def expected_dataset_path(
    processed_dir: Path,
    target_bins: int,
) -> Path:
    """Return the canonical dataset NPZ path for one target bin count."""

    return processed_dir / f"dataset_psd_{target_bins}_norm.npz"


def load_preprocess_config(
    config_path_like: str | Path,
    project_root: Path | None = None,
) -> PreprocessConfig:
    """Load and validate the VAE preprocessing configuration.

    Parameters
    ----------
    config_path_like:
        YAML config path. Relative paths are resolved against the working
        directory and then against the repository root.
    project_root:
        Optional project root override. This is primarily useful for tests.

    Returns
    -------
    PreprocessConfig
        Fully resolved configuration with absolute paths and validated scalar
        parameters.
    """

    root = project_root.resolve() if project_root is not None else repo_root()
    payload = load_yaml_config(config_path_like)

    paths_cfg = _require_mapping(payload, "paths")
    preprocess_cfg = _require_mapping(payload, "preprocess")
    split_cfg = _require_mapping(preprocess_cfg, "split")

    try:
        config = PreprocessConfig(
            raw_dataset_dir=_resolve_project_path(
                root, str(paths_cfg["raw_dataset_dir"])
            ),
            fallback_raw_dataset_dirs=_parse_fallback_dirs(
                root,
                paths_cfg.get("fallback_raw_dataset_dirs"),
            ),
            processed_dir=_resolve_project_path(root, str(paths_cfg["processed_dir"])),
            target_bins=int(preprocess_cfg["target_bins"]),
            normalize_mode=str(preprocess_cfg["normalize_mode"]),
            use_db_clip=bool(preprocess_cfg["use_db_clip"]),
            db_min=float(preprocess_cfg["db_min"]),
            db_max=float(preprocess_cfg["db_max"]),
            split=SplitConfig(
                train_ratio=float(split_cfg["train"]),
                val_ratio=float(split_cfg["val"]),
                test_ratio=float(split_cfg["test"]),
                mode=str(split_cfg["mode"]),
                seed=int(split_cfg["seed"]),
            ),
            reuse_existing_processed_if_source_missing=bool(
                preprocess_cfg.get("reuse_existing_processed_if_source_missing", False)
            ),
        )
    except KeyError as exc:
        raise KeyError(f"Missing preprocessing config key: {exc}") from exc

    _validate_preprocess_config(config)
    return config


def list_csv_files(folder: Path) -> list[Path]:
    """Return all CSV files below ``folder`` in deterministic path order."""

    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(path for path in folder.rglob("*.csv") if path.is_file())


def parse_pxx(pxx_str: str) -> np.ndarray | None:
    """Parse one serialized PSD array into a finite ``float32`` vector."""

    try:
        raw_values = json.loads(pxx_str)
    except json.JSONDecodeError:
        return None

    values = np.asarray(raw_values, dtype=np.float32)
    if values.ndim != 1 or values.size == 0:
        return None

    if not np.isfinite(values).all():
        values[~np.isfinite(values)] = np.nan
        values = (
            pd.Series(values)
            .interpolate(limit_direction="both")
            .to_numpy(dtype=np.float32)
        )

    return values if np.isfinite(values).all() else None


def downsample_to_bins(
    values: np.ndarray,
    target_bins: int,
) -> np.ndarray:
    """Resize one PSD vector to the configured bin count."""

    input_bins = values.size
    if input_bins == target_bins:
        return values.astype(np.float32, copy=False)

    if input_bins % target_bins == 0:
        block_size = input_bins // target_bins
        return (
            values.reshape(target_bins, block_size)
            .mean(axis=1)
            .astype(np.float32, copy=False)
        )

    # Use interpolation only when integer decimation is impossible.
    x_original = np.linspace(0.0, 1.0, input_bins, dtype=np.float32)
    x_target = np.linspace(0.0, 1.0, target_bins, dtype=np.float32)
    return np.interp(
        x_target, x_original, values.astype(np.float32, copy=False)
    ).astype(np.float32)


def _transform_frame(
    values: np.ndarray,
    config: PreprocessConfig,
) -> np.ndarray:
    """Apply resampling and optional clipping to one raw PSD frame."""

    transformed = downsample_to_bins(values, config.target_bins)
    if config.use_db_clip:
        transformed = np.clip(transformed, config.db_min, config.db_max)
    return transformed.astype(np.float32, copy=False)


def _normalize_frame(
    values: np.ndarray,
    config: PreprocessConfig,
    gmin: float,
    gmax: float,
) -> np.ndarray:
    """Normalize one transformed PSD frame according to the configured policy."""

    if config.normalize_mode == "global_minmax":
        return ((values - gmin) / (gmax - gmin + 1e-8)).astype(np.float32, copy=False)
    if config.normalize_mode == "per_frame_minmax":
        return (
            (values - float(values.min()))
            / (float(values.max()) - float(values.min()) + 1e-8)
        ).astype(np.float32, copy=False)
    return values.astype(np.float32, copy=False)


def _candidate_processed_dataset_paths(
    processed_dir: Path,
    target_bins: int,
) -> tuple[Path, ...]:
    """Return the processed dataset candidates used for reuse validation."""

    canonical_path = expected_dataset_path(processed_dir, target_bins)
    extra_candidates = sorted(
        path for path in processed_dir.glob("*.npz") if path != canonical_path
    )
    return (canonical_path, *extra_candidates)


def _has_complete_processed_outputs(
    processed_dir: Path,
    target_bins: int,
) -> bool:
    """Return whether the processed dataset artifacts are complete and readable."""

    metadata_path = processed_dir / "metadata.csv"
    splits_dir = processed_dir / "splits"
    split_paths = (
        splits_dir / "train_idx.npy",
        splits_dir / "val_idx.npy",
        splits_dir / "test_idx.npy",
    )
    if not metadata_path.is_file() or not all(path.is_file() for path in split_paths):
        return False

    for dataset_path in _candidate_processed_dataset_paths(processed_dir, target_bins):
        if not dataset_path.is_file():
            continue
        try:
            load_psd_dataset(dataset_path)
            return True
        except Exception:
            continue
    return False


def _choose_source_selection(config: PreprocessConfig) -> _SourceSelection:
    """Select the data source for preprocessing in a deterministic order.

    Selection order:
    1. Primary raw dataset directory from the config.
    2. Existing processed outputs when reuse is explicitly enabled.
    3. Any configured fallback raw dataset directory.
    """

    primary_csv_files = tuple(list_csv_files(config.raw_dataset_dir))
    if primary_csv_files:
        return _SourceSelection(
            source_dir=config.raw_dataset_dir,
            csv_files=primary_csv_files,
            used_existing_processed=False,
        )

    if (
        config.reuse_existing_processed_if_source_missing
        and _has_complete_processed_outputs(
            config.processed_dir,
            config.target_bins,
        )
    ):
        return _SourceSelection(
            source_dir=None,
            csv_files=(),
            used_existing_processed=True,
        )

    for fallback_dir in config.fallback_raw_dataset_dirs:
        fallback_csv_files = tuple(list_csv_files(fallback_dir))
        if fallback_csv_files:
            return _SourceSelection(
                source_dir=fallback_dir,
                csv_files=fallback_csv_files,
                used_existing_processed=False,
            )

    candidate_dirs = [config.raw_dataset_dir, *config.fallback_raw_dataset_dirs]
    formatted_candidates = "\n".join(f"  - {path}" for path in candidate_dirs)
    raise FileNotFoundError(
        "No CSV dataset source is available for preprocessing.\n"
        f"Searched directories:\n{formatted_candidates}\n"
        "Either provide the external raw dataset, configure a fallback dataset directory, "
        "or keep a complete processed dataset in place and enable "
        "`preprocess.reuse_existing_processed_if_source_missing`."
    )


def _discover_frequency_range(
    dataframe: pd.DataFrame,
) -> tuple[int, int] | None:
    """Extract the frequency range from the first valid metadata row."""

    required_columns = {"start_freq_hz", "end_freq_hz"}
    if not required_columns.issubset(dataframe.columns):
        return None

    frequency_rows = dataframe.dropna(subset=["start_freq_hz", "end_freq_hz"]).head(1)
    if len(frequency_rows) != 1:
        return None

    return (
        int(frequency_rows["start_freq_hz"].iloc[0]),
        int(frequency_rows["end_freq_hz"].iloc[0]),
    )


def _collect_global_statistics(
    csv_files: Sequence[Path],
    config: PreprocessConfig,
) -> tuple[float, float, int, tuple[int, int] | None]:
    """Scan the source CSV files to compute global normalization statistics."""

    gmin = np.inf
    gmax = -np.inf
    total_frames = 0
    frequency_range_hz: tuple[int, int] | None = None

    for csv_path in csv_files:
        dataframe = pd.read_csv(csv_path)
        if "pxx" not in dataframe.columns:
            continue

        if frequency_range_hz is None:
            frequency_range_hz = _discover_frequency_range(dataframe)

        for serialized_pxx in dataframe["pxx"].astype(str).values:
            parsed_values = parse_pxx(serialized_pxx)
            if parsed_values is None:
                continue

            transformed = _transform_frame(parsed_values, config)
            gmin = min(gmin, float(transformed.min()))
            gmax = max(gmax, float(transformed.max()))
            total_frames += 1

    if total_frames < 1:
        raise ValueError("No valid PSD frames were parsed from the selected CSV files.")

    return float(gmin), float(gmax), total_frames, frequency_range_hz


def _build_dataset_and_metadata(
    csv_files: Sequence[Path],
    config: PreprocessConfig,
    gmin: float,
    gmax: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build the normalized PSD matrix and aligned metadata table."""

    frames = np.zeros((len(csv_files), 0), dtype=np.float32)
    metadata: dict[str, list[Any]] = {
        key: []
        for key in (
            "source_file",
            "row_index",
            "id",
            "timestamp",
            "created_at",
            "mac",
            "campaign_id",
            "start_freq_hz",
            "end_freq_hz",
        )
    }

    frame_rows: list[np.ndarray] = []
    for csv_path in csv_files:
        dataframe = pd.read_csv(csv_path)
        if "pxx" not in dataframe.columns:
            continue

        for column in (
            "id",
            "timestamp",
            "created_at",
            "mac",
            "campaign_id",
            "start_freq_hz",
            "end_freq_hz",
        ):
            if column not in dataframe.columns:
                dataframe[column] = np.nan

        for row_position, (_, row) in enumerate(dataframe.iterrows()):
            parsed_values = parse_pxx(str(row["pxx"]))
            if parsed_values is None:
                continue

            transformed = _transform_frame(parsed_values, config)
            normalized = _normalize_frame(transformed, config, gmin, gmax)
            frame_rows.append(normalized)

            metadata["source_file"].append(csv_path.name)
            metadata["row_index"].append(row_position)
            metadata["id"].append(None if pd.isna(row["id"]) else int(row["id"]))
            metadata["timestamp"].append(
                None if pd.isna(row["timestamp"]) else int(row["timestamp"])
            )
            metadata["created_at"].append(
                None if pd.isna(row["created_at"]) else int(row["created_at"])
            )
            metadata["mac"].append(None if pd.isna(row["mac"]) else str(row["mac"]))
            metadata["campaign_id"].append(
                None if pd.isna(row["campaign_id"]) else int(row["campaign_id"])
            )
            metadata["start_freq_hz"].append(
                None if pd.isna(row["start_freq_hz"]) else int(row["start_freq_hz"])
            )
            metadata["end_freq_hz"].append(
                None if pd.isna(row["end_freq_hz"]) else int(row["end_freq_hz"])
            )

    if not frame_rows:
        raise ValueError(
            "No normalized PSD frames were produced from the selected CSV files."
        )

    frames = np.asarray(frame_rows, dtype=np.float32)
    return frames, pd.DataFrame(metadata)


def _split_indices(
    frame_count: int,
    timestamps_ms: np.ndarray,
    split: SplitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic train/validation/test index arrays."""

    indices = np.arange(frame_count, dtype=np.int64)
    if split.mode == "time_ordered" and np.isfinite(timestamps_ms).any():
        finite_values = timestamps_ms[np.isfinite(timestamps_ms)]
        fill_value = (
            float(np.nanmax(finite_values) + 1.0) if finite_values.size else 0.0
        )
        ordered_timestamps_ms = np.where(
            np.isfinite(timestamps_ms), timestamps_ms, fill_value
        )
        indices = indices[np.argsort(ordered_timestamps_ms, kind="mergesort")]
    elif split.mode == "random":
        indices = np.random.default_rng(split.seed).permutation(indices)

    frame_total = int(indices.size)
    train_count = int(round(split.train_ratio * frame_total))
    val_count = int(round(split.val_ratio * frame_total))
    train_idx = indices[:train_count]
    val_idx = indices[train_count : train_count + val_count]
    test_idx = indices[train_count + val_count :]
    return train_idx, val_idx, test_idx


def _write_processed_outputs(
    config: PreprocessConfig,
    frames: np.ndarray,
    metadata: pd.DataFrame,
    gmin: float,
    gmax: float,
    frequency_range_hz: tuple[int, int] | None,
) -> tuple[Path, Path, np.ndarray, np.ndarray, np.ndarray]:
    """Persist the processed dataset, metadata table, and split index arrays."""

    config.processed_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = config.processed_dir / "metadata.csv"
    dataset_path = expected_dataset_path(config.processed_dir, config.target_bins)
    metadata.to_csv(metadata_path, index=False)

    if frequency_range_hz is not None:
        start_freq_hz, end_freq_hz = frequency_range_hz
        freqs_hz = np.linspace(
            start_freq_hz,
            end_freq_hz,
            config.target_bins,
            endpoint=False,
            dtype=np.float64,
        )
        np.savez_compressed(
            dataset_path,
            X=frames,
            freqs_hz=freqs_hz,
            gmin=gmin,
            gmax=gmax,
            normalize_mode=config.normalize_mode,
        )
    else:
        np.savez_compressed(
            dataset_path,
            X=frames,
            gmin=gmin,
            gmax=gmax,
            normalize_mode=config.normalize_mode,
        )

    timestamps_ms = pd.to_numeric(metadata["timestamp"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    train_idx, val_idx, test_idx = _split_indices(
        frame_count=frames.shape[0],
        timestamps_ms=timestamps_ms,
        split=config.split,
    )

    splits_dir = config.processed_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    np.save(splits_dir / "train_idx.npy", train_idx)
    np.save(splits_dir / "val_idx.npy", val_idx)
    np.save(splits_dir / "test_idx.npy", test_idx)

    return dataset_path, metadata_path, train_idx, val_idx, test_idx


def _result_from_existing_processed(config: PreprocessConfig) -> PreprocessRunResult:
    """Build the execution summary for the reuse-existing fast path."""

    dataset_candidates = _candidate_processed_dataset_paths(
        config.processed_dir, config.target_bins
    )
    dataset_path = next(path for path in dataset_candidates if path.is_file())
    frames, _, metadata = load_psd_dataset(dataset_path)
    train_idx = np.load(config.processed_dir / "splits" / "train_idx.npy")
    val_idx = np.load(config.processed_dir / "splits" / "val_idx.npy")
    test_idx = np.load(config.processed_dir / "splits" / "test_idx.npy")

    return PreprocessRunResult(
        source_dir=None,
        processed_dir=config.processed_dir,
        dataset_path=Path(metadata["dataset_path"]),
        metadata_path=config.processed_dir / "metadata.csv",
        total_frames=int(frames.shape[0]),
        target_bins=int(frames.shape[1]),
        train_count=int(train_idx.size),
        val_count=int(val_idx.size),
        test_count=int(test_idx.size),
        used_existing_processed=True,
    )


def run_preprocess(
    config_path_like: str | Path,
    project_root: Path | None = None,
) -> PreprocessRunResult:
    """Run the VAE preprocessing workflow from one YAML configuration.

    Parameters
    ----------
    config_path_like:
        YAML config path describing source directories, output directory,
        normalization policy, and split ratios.
    project_root:
        Optional project root override used to resolve repository-relative
        paths. Tests use this to isolate temporary workspaces.

    Returns
    -------
    PreprocessRunResult
        Structured summary of the produced or reused dataset.

    Side Effects
    ------------
    Reads CSV input files, writes the processed NPZ dataset, writes the aligned
    metadata CSV, and writes the train/validation/test split arrays.
    """

    _ = project_root.resolve() if project_root is not None else repo_root()
    config = load_preprocess_config(config_path_like, project_root=project_root)
    source_selection = _choose_source_selection(config)

    if source_selection.used_existing_processed:
        return _result_from_existing_processed(config)

    gmin, gmax, _, frequency_range_hz = _collect_global_statistics(
        source_selection.csv_files,
        config,
    )
    frames, metadata = _build_dataset_and_metadata(
        source_selection.csv_files,
        config,
        gmin,
        gmax,
    )
    dataset_path, metadata_path, train_idx, val_idx, test_idx = (
        _write_processed_outputs(
            config,
            frames,
            metadata,
            gmin,
            gmax,
            frequency_range_hz,
        )
    )

    return PreprocessRunResult(
        source_dir=source_selection.source_dir,
        processed_dir=config.processed_dir,
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        total_frames=int(frames.shape[0]),
        target_bins=int(frames.shape[1]),
        train_count=int(train_idx.size),
        val_count=int(val_idx.size),
        test_count=int(test_idx.size),
        used_existing_processed=False,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser used by the legacy script wrapper."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML configuration path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the preprocessing CLI and print a compact execution summary."""

    args = build_argument_parser().parse_args(argv)
    result = run_preprocess(args.config)

    if result.used_existing_processed:
        print("[DONE] Reused existing processed dataset.")
    else:
        print("[DONE] Preprocess complete.")
        print(f"Source: {result.source_dir}")

    print(f"Processed: {result.processed_dir}")
    print(
        "X shape:",
        (result.total_frames, result.target_bins),
        "splits:",
        result.train_count,
        result.val_count,
        result.test_count,
    )
    return 0


__all__ = [
    "PreprocessConfig",
    "PreprocessRunResult",
    "SplitConfig",
    "build_argument_parser",
    "expected_dataset_path",
    "list_csv_files",
    "load_preprocess_config",
    "main",
    "parse_pxx",
    "run_preprocess",
]
