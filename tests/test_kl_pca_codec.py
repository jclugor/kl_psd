from __future__ import annotations

import numpy as np

from psd_compression.kl_pca.model import decode_coefficients, encode_frame, fit_kl_pca


def test_kl_pca_roundtrip_synthetic() -> None:
    rng = np.random.default_rng(2026)
    frames = np.abs(rng.normal(size=(128, 256))).astype(np.float64) + 1e-3
    model = fit_kl_pca(frames, n_components=24, center=True)

    x = frames[0]
    c = encode_frame(x, model)
    xr = decode_coefficients(c, model, enforce_nonnegative=True)

    assert c.shape == (24,)
    assert xr.shape == x.shape
    assert np.all(np.isfinite(xr))
    assert float(np.mean((x - xr) ** 2)) < 0.25
