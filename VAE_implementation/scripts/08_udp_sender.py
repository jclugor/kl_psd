#!/usr/bin/env python3
"""
08_udp_sender.py — UDP sender (edge / Raspberry simulator).

- Loads preprocessed dataset (.npz) and splits
- Loads encoder_mu_int8.tflite (INT8 in/out)
- Runs encoder on frames to get mu_int8
- Splits mu stream into blocks (L<=255), packs+zlib via 07_pack_unpack.py, sends UDP

Usage (send to localhost receiver):
  python VAE_implementation/scripts/08_udp_sender.py --config VAE_implementation/configs/vae_default.yaml --use_global_best --dest_ip 127.0.0.1 --port 5005 --split test --block_len 30 --n_blocks 50

Notes:
- This simulates edge. On Raspberry you would replace dataset reading with actual PSD acquisition.
"""

import argparse
import importlib.util
import socket
import time
from pathlib import Path

import numpy as np
import yaml
import tensorflow as tf


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
        # fallback: biggest npz
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
    """
    x_1024: (1024,) float32 in [0,1]
    returns: (32,) int8
    """
    in_scale, in_zp = in_det["quantization"]
    x = x_1024[None, :, None].astype(np.float32)  # (1,1024,1)

    if in_det["dtype"] == np.int8:
        xq = np.round(x / in_scale + in_zp).astype(np.int8)
    elif in_det["dtype"] == np.uint8:
        xq = np.round(x / in_scale + in_zp).astype(np.uint8)
    else:
        xq = x.astype(in_det["dtype"])

    interpreter.set_tensor(in_det["index"], xq)
    interpreter.invoke()
    yq = interpreter.get_tensor(out_det["index"])[0]  # (32,)
    return yq.astype(np.int8, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--use_global_best", action="store_true")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--dest_ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--block_len", type=int, default=30, help="L per packet (<=255)")
    ap.add_argument("--n_blocks", type=int, default=50, help="How many blocks to send")
    ap.add_argument("--zlib_level", type=int, default=1)
    ap.add_argument("--sleep_ms", type=float, default=0.0, help="Sleep between packets (simulate real-time)")
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
        raise FileNotFoundError(f"TFLite encoder not found: {tflite_path}. Run 04_export_tflite.py first.")

    packmod = load_pack_module()

    # Load data
    X = load_npz(processed_dir)
    tr, va, te = load_splits(processed_dir)
    if args.split == "train":
        idx = tr
    elif args.split == "val":
        idx = va
    else:
        idx = te

    # Use first N frames needed
    frames_needed = args.n_blocks * args.block_len
    if len(idx) < frames_needed:
        frames_needed = len(idx) - (len(idx) % args.block_len)
    idx = idx[:frames_needed]

    # TFLite encoder
    interpreter, in_det, out_det = load_tflite(tflite_path)

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.dest_ip, args.port)

    print(f"[SEND] Sending to udp://{args.dest_ip}:{args.port} | {tag} | split={args.split}")
    print(f"[SEND] blocks={args.n_blocks} block_len={args.block_len} zlib={args.zlib_level}")

    seq = 0
    sent_frames = 0
    sent_bytes = 0
    t0 = time.perf_counter()

    # Stream blocks
    for b in range(args.n_blocks):
        s = b * args.block_len
        e = s + args.block_len
        if e > len(idx):
            break

        # Compute mu_int8 block
        mu_block = np.zeros((args.block_len, 32), dtype=np.int8)
        for i, k in enumerate(idx[s:e]):
            mu_block[i] = infer_mu_int8_one(interpreter, in_det, out_det, X[k])

        pkt = packmod.pack_packet(mu_block, seq=seq, zlib_level=args.zlib_level, keyframe=True)
        sock.sendto(pkt, dest)

        seq += 1
        sent_frames += args.block_len
        sent_bytes += len(pkt)

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    dt = time.perf_counter() - t0
    print(f"[SEND] DONE blocks={seq} frames={sent_frames} bytes={sent_bytes} "
          f"| frames/s={sent_frames/dt:.1f} | KB/s={sent_bytes/dt/1024:.1f} | avg bytes/frame={sent_bytes/max(1,sent_frames):.3f}")


if __name__ == "__main__":
    main()