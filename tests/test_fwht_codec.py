from __future__ import annotations

import numpy as np

from psd_compression.fwht.codec import (
    FWHTConfig,
    compress_fwht_frame,
    decompress_fwht_frame,
    estimate_payload_bits,
)


def test_fwht_roundtrip_shape_and_quality() -> None:
    rng = np.random.default_rng(2026)
    frame = np.abs(rng.normal(loc=0.0, scale=1.0, size=1024)).astype(np.float64) + 1e-3
    cfg = FWHTConfig(decimation_factor_bins=2, top_k_coeffs=128, quant_step=0.02)

    packet = compress_fwht_frame(frame, cfg)
    recon = decompress_fwht_frame(packet)

    assert recon.shape == frame.shape
    mse = float(np.mean((frame - recon) ** 2))
    assert np.isfinite(mse)
    assert mse < 0.5


def test_fwht_payload_bits_scales_with_topk() -> None:
    rng = np.random.default_rng(2026)
    frame = np.abs(rng.normal(loc=0.0, scale=1.0, size=1024)).astype(np.float64) + 1e-3

    cfg_small = FWHTConfig(top_k_coeffs=32)
    cfg_large = FWHTConfig(top_k_coeffs=128)
    pkt_small = compress_fwht_frame(frame, cfg_small)
    pkt_large = compress_fwht_frame(frame, cfg_large)

    bits_small = estimate_payload_bits(pkt_small, cfg_small)
    bits_large = estimate_payload_bits(pkt_large, cfg_large)

    assert bits_small > 0
    assert bits_large > bits_small

