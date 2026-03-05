#!/usr/bin/env python3
"""
09_udp_sender_indexed.py (improved) — UDP sender with indices for server-side comparison.

- Loads X from dataset_psd_1024_norm.npz
- Loads split indices
- Encodes frames with encoder_mu_int8.tflite -> mu_int8
- Sends packets containing:
   header (KLP1) + zlib(payload)
   payload = meta + raw_mu_block
   meta v1:
     [meta_ver=1][L][indices (L*u32 BE)]
   raw_mu_block:
     encode_mu_block(mu_block)  # includes its own [L][mu0][deltas]

Improvements:
- If split has fewer frames than requested, it automatically reduces blocks and prints it.
- If --n_blocks 0, sends ALL possible blocks in that split.
"""

import argparse
import importlib.util
import socket
import time
from pathlib import Path

import numpy as np
import yaml
import tensorflow as tf
import zlib


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_pack_module() -> object:
    root = repo_root()
    mod_path = root / "VAE_implementation" / "scripts" / "07_pack_unpack.py"
    spec = importlib.util.spec_from_file_location("packmod", str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def load_npz(processed_dir: Path) -> np.ndarray:
    npz_path = processed_dir / "dataset_psd_1024_norm.npz"
    if not npz_path.exists():
        candidates = sorted(processed_dir.glob("*.npz"), key=lambda p: p.stat().st_size, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No .npz found in {processed_dir}")
        npz_path = candidates[0]
    d = np.load(npz_path, allow_pickle=True)
    return d["X"].astype(np.float32)


def load_splits(processed_dir: Path):
    sdir = processed_dir / "splits"
    tr = np.load(sdir / "train_idx.npy")
    va = np.load(sdir / "val_idx.npy")
    te = np.load(sdir / "test_idx.npy")
    return tr, va, te


def load_tflite(tflite_path: Path):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    return interpreter, in_det, out_det


def infer_mu_int8_one(interpreter, in_det, out_det, x_1024: np.ndarray) -> np.ndarray:
    in_scale, in_zp = in_det["quantization"]
    x = x_1024[None, :, None].astype(np.float32)

    if in_det["dtype"] == np.int8:
        xq = np.round(x / in_scale + in_zp).astype(np.int8)
    elif in_det["dtype"] == np.uint8:
        xq = np.round(x / in_scale + in_zp).astype(np.uint8)
    else:
        xq = x.astype(in_det["dtype"])

    interpreter.set_tensor(in_det["index"], xq)
    interpreter.invoke()
    yq = interpreter.get_tensor(out_det["index"])[0]
    return yq.astype(np.int8, copy=False)


def pack_indexed_packet(packmod, mu_block: np.ndarray, indices: np.ndarray,
                        seq: int, zlib_level: int = 1, keyframe: bool = True) -> bytes:
    L = int(mu_block.shape[0])
    if indices.shape[0] != L:
        raise ValueError("indices length must match block length")

    meta = bytes([1, L]) + indices.astype(">u4", copy=False).tobytes()
    raw_block = packmod.encode_mu_block(mu_block)  # includes its own L
    payload_raw = meta + raw_block
    payload = zlib.compress(payload_raw, level=zlib_level)

    flags = packmod.FLAG_KEYFRAME if keyframe else 0
    reserved = 0
    ts_ms = int(time.time() * 1000)

    header = packmod.HDR_STRUCT.pack(
        packmod.MAGIC, packmod.VERSION, flags, reserved,
        int(seq) & 0xFFFFFFFF, int(ts_ms) & 0xFFFFFFFFFFFFFFFF, len(payload)
    )
    return header + payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--use_global_best", action="store_true")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--dest_ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--block_len", type=int, default=30)
    ap.add_argument("--n_blocks", type=int, default=0, help="0 = send all possible blocks")
    ap.add_argument("--zlib_level", type=int, default=1)
    ap.add_argument("--sleep_ms", type=float, default=0.0)
    ap.add_argument("--start_offset_blocks", type=int, default=0)
    args = ap.parse_args()

    root = repo_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text())

    processed_dir = root / cfg["paths"]["processed_dir"]
    models_dir = root / cfg["paths"]["models_dir"]

    if args.use_global_best:
        model_dir = models_dir / "GLOBAL_BEST"
        tag = "GLOBAL_BEST"
    else:
        run = args.run_name or cfg.get("train", {}).get("run_name", "run_current")
        model_dir = models_dir / "runs" / run
        tag = f"RUN:{run}"

    tflite_path = model_dir / "encoder_mu_int8.tflite"
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite encoder not found: {tflite_path} (run 04_export_tflite.py).")

    packmod = load_pack_module()
    X = load_npz(processed_dir)

    tr, va, te = load_splits(processed_dir)
    idx = {"train": tr, "val": va, "test": te}[args.split].astype(np.int64)

    L = int(args.block_len)
    if L <= 0 or L > 255:
        raise ValueError("block_len must be in [1,255]")

    # Drop to full blocks
    total_full_blocks = idx.shape[0] // L
    if total_full_blocks <= 0:
        raise ValueError(f"Split '{args.split}' has not enough frames for block_len={L}.")

    start_b = max(0, int(args.start_offset_blocks))
    if start_b >= total_full_blocks:
        raise ValueError("start_offset_blocks is beyond available blocks.")

    total_full_blocks = total_full_blocks - start_b

    if args.n_blocks == 0:
        n_blocks = total_full_blocks
    else:
        n_blocks = min(int(args.n_blocks), total_full_blocks)

    if n_blocks < int(args.n_blocks):
        print(f"[SEND9] NOTE: requested n_blocks={args.n_blocks}, but split '{args.split}' allows only {n_blocks} blocks of L={L}.")

    idx = idx[start_b * L:(start_b + n_blocks) * L]

    interpreter, in_det, out_det = load_tflite(tflite_path)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.dest_ip, args.port)

    print(f"[SEND9] -> udp://{args.dest_ip}:{args.port} | {tag} | split={args.split} | blocks={n_blocks} L={L} zlib={args.zlib_level}")

    seq = 0
    sent_frames = 0
    sent_bytes = 0
    t0 = time.perf_counter()

    for b in range(n_blocks):
        s = b * L
        e = s + L
        ind_block = idx[s:e]
        mu_block = np.zeros((L, 32), dtype=np.int8)

        for i, k in enumerate(ind_block):
            mu_block[i] = infer_mu_int8_one(interpreter, in_det, out_det, X[int(k)])

        pkt = pack_indexed_packet(packmod, mu_block, ind_block.astype(np.uint32), seq=seq,
                                  zlib_level=args.zlib_level, keyframe=True)
        sock.sendto(pkt, dest)

        seq += 1
        sent_frames += L
        sent_bytes += len(pkt)

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    dt = time.perf_counter() - t0
    print(f"[SEND9] DONE blocks={seq} frames={sent_frames} bytes={sent_bytes} "
          f"| frames/s={sent_frames/dt:.1f} | KB/s={sent_bytes/dt/1024:.2f} | avg bytes/frame={sent_bytes/max(1,sent_frames):.3f}")


if __name__ == "__main__":
    main()