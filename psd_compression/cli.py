from __future__ import annotations

import argparse
import json
import sys

from psd_compression.fwht.tasks import run_decode, run_encode, run_evaluate
from psd_compression.kl_pca.tasks import (
    run_decode as run_kl_decode,
    run_encode as run_kl_encode,
    run_evaluate as run_kl_evaluate,
    run_fit as run_kl_fit,
)
from psd_compression.vae.wrappers import run_vae_task


def _print_json(obj: dict) -> None:
    print(json.dumps(obj, indent=2))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m psd_compression.cli",
        description="Unified PSD compression task runner (KL/PCA + VAE + FWHT).",
    )
    sub = parser.add_subparsers(dest="group", required=True)

    # FWHT
    fwht = sub.add_parser("fwht", help="Deterministic FWHT codec tasks.")
    fwht_sub = fwht.add_subparsers(dest="fwht_task", required=True)

    fwht_encode = fwht_sub.add_parser("encode", help="Encode one PSD frame into FWHT packet.")
    fwht_encode.add_argument("--config", default="configs/fwht_default.yaml")
    fwht_encode.add_argument("--frame-index", type=int, default=0)
    fwht_encode.add_argument("--output", default="data/processed/psd_1024/fwht_packet_frame0.npz")
    fwht_encode.add_argument("--dry-run", action="store_true")

    fwht_decode = fwht_sub.add_parser("decode", help="Decode one FWHT packet into reconstructed frame.")
    fwht_decode.add_argument("--packet", default="data/processed/psd_1024/fwht_packet_frame0.npz")
    fwht_decode.add_argument("--output", default="data/processed/psd_1024/fwht_reconstruction_frame0.npy")
    fwht_decode.add_argument("--dry-run", action="store_true")

    fwht_eval = fwht_sub.add_parser("evaluate", help="Evaluate FWHT codec over multiple frames.")
    fwht_eval.add_argument("--config", default="configs/fwht_default.yaml")
    fwht_eval.add_argument("--max-frames", type=int, default=None)
    fwht_eval.add_argument("--output", default=None, help="Optional report output JSON path.")
    fwht_eval.add_argument("--dry-run", action="store_true")

    # VAE wrappers
    vae = sub.add_parser("vae", help="VAE wrapper tasks delegating to legacy scripts.")
    vae_sub = vae.add_subparsers(dest="vae_task", required=True)
    for task in ("preprocess", "train", "eval", "export", "entropy", "benchmark"):
        task_parser = vae_sub.add_parser(task, help=f"Delegate to legacy VAE `{task}` script.")
        task_parser.add_argument("--dry-run", action="store_true")

    # KL/PCA module tasks
    kl = sub.add_parser("kl-pca", help="KL/PCA module tasks.")
    kl_sub = kl.add_subparsers(dest="kl_task", required=True)

    kl_fit = kl_sub.add_parser("fit", help="Fit KL/PCA model on dataset frames.")
    kl_fit.add_argument("--config", default="configs/kl_pca_default.yaml")
    kl_fit.add_argument("--output", default=None, help="Optional model output path.")
    kl_fit.add_argument("--max-frames", type=int, default=None)
    kl_fit.add_argument("--dry-run", action="store_true")

    kl_encode = kl_sub.add_parser("encode", help="Encode one frame into KL/PCA coefficients.")
    kl_encode.add_argument("--config", default="configs/kl_pca_default.yaml")
    kl_encode.add_argument("--model", default=None, help="Optional model path override.")
    kl_encode.add_argument("--frame-index", type=int, default=0)
    kl_encode.add_argument("--output", default=None, help="Optional coeff output path.")
    kl_encode.add_argument("--dry-run", action="store_true")

    kl_decode = kl_sub.add_parser("decode", help="Decode one KL/PCA coefficient vector into PSD frame.")
    kl_decode.add_argument("--config", default="configs/kl_pca_default.yaml")
    kl_decode.add_argument("--model", default=None, help="Optional model path override.")
    kl_decode.add_argument("--coeff", default=None, help="Optional coefficient path override.")
    kl_decode.add_argument("--output", default=None, help="Optional reconstructed frame output path.")
    kl_decode.add_argument("--dry-run", action="store_true")

    kl_eval = kl_sub.add_parser("evaluate", help="Evaluate KL/PCA reconstruction on multiple frames.")
    kl_eval.add_argument("--config", default="configs/kl_pca_default.yaml")
    kl_eval.add_argument("--model", default=None, help="Optional model path override.")
    kl_eval.add_argument("--max-frames", type=int, default=None)
    kl_eval.add_argument("--output", default=None, help="Optional report output path.")
    kl_eval.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)

    if args.group == "fwht":
        if args.fwht_task == "encode":
            _print_json(run_encode(args.config, args.frame_index, args.output, dry_run=args.dry_run))
            return 0
        if args.fwht_task == "decode":
            _print_json(run_decode(args.packet, args.output, dry_run=args.dry_run))
            return 0
        if args.fwht_task == "evaluate":
            _print_json(run_evaluate(args.config, max_frames=args.max_frames, output_path=args.output, dry_run=args.dry_run))
            return 0
        parser.error(f"Unsupported fwht task: {args.fwht_task}")

    if args.group == "vae":
        result = run_vae_task(args.vae_task, unknown, dry_run=args.dry_run)
        _print_json(result)
        if not args.dry_run and "return_code" in result:
            return int(result["return_code"])
        return 0

    if args.group == "kl-pca":
        if args.kl_task == "fit":
            _print_json(run_kl_fit(args.config, output_path=args.output, max_frames=args.max_frames, dry_run=args.dry_run))
            return 0
        if args.kl_task == "encode":
            _print_json(
                run_kl_encode(
                    args.config,
                    model_path=args.model,
                    frame_index=args.frame_index,
                    output_path=args.output,
                    dry_run=args.dry_run,
                )
            )
            return 0
        if args.kl_task == "decode":
            _print_json(
                run_kl_decode(
                    args.config,
                    model_path=args.model,
                    coeff_path=args.coeff,
                    output_path=args.output,
                    dry_run=args.dry_run,
                )
            )
            return 0
        if args.kl_task == "evaluate":
            _print_json(
                run_kl_evaluate(
                    args.config,
                    model_path=args.model,
                    max_frames=args.max_frames,
                    output_path=args.output,
                    dry_run=args.dry_run,
                )
            )
            return 0

        parser.error(f"Unsupported kl-pca task: {args.kl_task}")

    parser.error(f"Unsupported task group: {args.group}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
