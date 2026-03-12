#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def repo_root() -> Path:
    # .../kl_psd/VAE_implementation/scripts/training/01_preprocess.py -> parents[2] = kl_psd
    return Path(__file__).resolve().parents[2]

def list_csv_files(folder: Path):
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"])

def parse_pxx(pxx_str: str):
    try:
        arr = json.loads(pxx_str)
        x = np.asarray(arr, dtype=np.float32)
        if x.ndim != 1 or x.size == 0:
            return None
        if not np.isfinite(x).all():
            x[~np.isfinite(x)] = np.nan
            x = pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype=np.float32)
        return x
    except Exception:
        return None

def downsample_to_bins(x: np.ndarray, target_bins: int) -> np.ndarray:
    n = x.size
    if n == target_bins:
        return x.astype(np.float32, copy=False)
    if n % target_bins == 0:
        k = n // target_bins
        return x.reshape(target_bins, k).mean(axis=1).astype(np.float32, copy=False)
    # fallback: interpolation
    xp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    xq = np.linspace(0.0, 1.0, target_bins, dtype=np.float32)
    return np.interp(xq, xp, x.astype(np.float32, copy=False)).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    root = repo_root()

    raw_dir = root / cfg["paths"]["raw_dataset_dir"]
    out_dir = root / cfg["paths"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    target_bins = int(cfg["preprocess"]["target_bins"])
    norm_mode = cfg["preprocess"]["normalize_mode"]

    use_db_clip = bool(cfg["preprocess"]["use_db_clip"])
    db_min = float(cfg["preprocess"]["db_min"])
    db_max = float(cfg["preprocess"]["db_max"])

    csv_files = list_csv_files(raw_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {raw_dir}. Did you run 00_get_data.sh?")

    # Pass 1: global min/max
    gmin, gmax = np.inf, -np.inf
    total_frames = 0

    freq_info = None

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if "pxx" not in df.columns:
            continue

        if freq_info is None and ("start_freq_hz" in df.columns) and ("end_freq_hz" in df.columns):
            row0 = df.dropna(subset=["start_freq_hz", "end_freq_hz"]).head(1)
            if len(row0) == 1:
                freq_info = (int(row0["start_freq_hz"].iloc[0]), int(row0["end_freq_hz"].iloc[0]))

        for s in df["pxx"].astype(str).values:
            x = parse_pxx(s)
            if x is None:
                continue
            x = downsample_to_bins(x, target_bins)
            if use_db_clip:
                x = np.clip(x, db_min, db_max)
            gmin = min(gmin, float(x.min()))
            gmax = max(gmax, float(x.max()))
            total_frames += 1

    if total_frames == 0:
        raise ValueError("No valid PSD frames parsed.")

    # Pass 2: build dataset + metadata
    X = np.zeros((total_frames, target_bins), dtype=np.float32)
    meta = {k: [] for k in ["source_file","row_index","id","timestamp","created_at","mac","campaign_id","start_freq_hz","end_freq_hz"]}

    frame_idx = 0
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if "pxx" not in df.columns:
            continue
        for col in ["id","timestamp","created_at","mac","campaign_id","start_freq_hz","end_freq_hz"]:
            if col not in df.columns:
                df[col] = np.nan

        for r, row in df.iterrows():
            x = parse_pxx(str(row["pxx"]))
            if x is None:
                continue

            x = downsample_to_bins(x, target_bins)
            if use_db_clip:
                x = np.clip(x, db_min, db_max)

            if norm_mode == "global_minmax":
                x = (x - gmin) / (gmax - gmin + 1e-8)
            elif norm_mode == "per_frame_minmax":
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)

            X[frame_idx] = x.astype(np.float32)

            meta["source_file"].append(csv_path.name)
            meta["row_index"].append(int(r))
            meta["id"].append(None if pd.isna(row["id"]) else int(row["id"]))
            meta["timestamp"].append(None if pd.isna(row["timestamp"]) else int(row["timestamp"]))
            meta["created_at"].append(None if pd.isna(row["created_at"]) else int(row["created_at"]))
            meta["mac"].append(None if pd.isna(row["mac"]) else str(row["mac"]))
            meta["campaign_id"].append(None if pd.isna(row["campaign_id"]) else int(row["campaign_id"]))
            meta["start_freq_hz"].append(None if pd.isna(row["start_freq_hz"]) else int(row["start_freq_hz"]))
            meta["end_freq_hz"].append(None if pd.isna(row["end_freq_hz"]) else int(row["end_freq_hz"]))
            frame_idx += 1

    meta_df = pd.DataFrame(meta)
    meta_df.to_csv(out_dir / "metadata.csv", index=False)

    if freq_info is not None:
        start_f, end_f = freq_info
        freqs_hz = np.linspace(start_f, end_f, target_bins, endpoint=False, dtype=np.float64)
        np.savez_compressed(out_dir / "dataset_psd_1024_norm.npz", X=X, freqs_hz=freqs_hz, gmin=gmin, gmax=gmax, normalize_mode=norm_mode)
    else:
        np.savez_compressed(out_dir / "dataset_psd_1024_norm.npz", X=X, gmin=gmin, gmax=gmax, normalize_mode=norm_mode)

    # Splits
    split_cfg = cfg["preprocess"]["split"]
    train_r, val_r, test_r = float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"])
    mode = split_cfg["mode"]
    seed = int(split_cfg["seed"])

    idx = np.arange(X.shape[0])
    if mode == "time_ordered" and meta_df["timestamp"].notna().any():
        ts = meta_df["timestamp"].to_numpy()
        finite = ts[np.isfinite(ts)]
        fill = np.nanmax(finite) + 1 if finite.size else 0
        ts_filled = np.where(np.isfinite(ts), ts, fill)
        order = np.argsort(ts_filled, kind="mergesort")
        idx = idx[order]
    else:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(idx)

    n = len(idx)
    n_train = int(round(train_r * n))
    n_val = int(round(val_r * n))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    np.save(splits_dir / "train_idx.npy", train_idx.astype(np.int64))
    np.save(splits_dir / "val_idx.npy", val_idx.astype(np.int64))
    np.save(splits_dir / "test_idx.npy", test_idx.astype(np.int64))

    print("[DONE] Preprocess complete.")
    print("Processed:", out_dir)
    print("X shape:", X.shape, "splits:", len(train_idx), len(val_idx), len(test_idx))

if __name__ == "__main__":
    main()
