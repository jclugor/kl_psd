#!/usr/bin/env python3
"""Legacy CLI entrypoint for VAE PSD preprocessing."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_packaged_entrypoint() -> int:
    """Execute the packaged preprocessing CLI from the legacy script path."""

    # Keep direct ``python path/to/01_preprocess.py`` execution working by
    # exposing the repository root before importing the packaged module.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from psd_compression.vae.preprocess import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_run_packaged_entrypoint())
