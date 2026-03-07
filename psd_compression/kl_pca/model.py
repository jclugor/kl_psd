from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KLPCAConfig:
    n_components: int = 32
    center: bool = True
    enforce_nonnegative: bool = True


@dataclass(frozen=True)
class KLPCAModel:
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray


def fit_kl_pca(frames: np.ndarray, n_components: int = 32, center: bool = True) -> KLPCAModel:
    x = np.asarray(frames, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("frames must have shape [num_frames, num_bins]")
    n_samples, n_bins = x.shape
    if n_samples < 2:
        raise ValueError("need at least 2 frames to fit KL/PCA")

    r = int(max(1, min(n_components, n_samples, n_bins)))
    mean = np.mean(x, axis=0) if center else np.zeros(n_bins, dtype=np.float64)
    xc = x - mean[None, :]

    # PCA by SVD: Xc = U S V^T, components are rows of V^T.
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[:r].copy()

    eigvals = (s**2) / max(n_samples - 1, 1)
    total = float(np.sum(eigvals) + 1e-30)
    evr = eigvals[:r] / total
    return KLPCAModel(mean=mean, components=components, explained_variance_ratio=evr)


def encode_frame(frame: np.ndarray, model: KLPCAModel) -> np.ndarray:
    x = np.asarray(frame, dtype=np.float64).reshape(-1)
    return (x - model.mean) @ model.components.T


def decode_coefficients(coeffs: np.ndarray, model: KLPCAModel, enforce_nonnegative: bool = True) -> np.ndarray:
    c = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    recon = model.mean + c @ model.components
    if enforce_nonnegative:
        recon = np.maximum(recon, 0.0)
    return recon

