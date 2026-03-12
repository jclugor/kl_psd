"""Deterministic FWHT codec primitives for PSD frame compression."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Literal, Tuple

import numpy as np

from psd_compression.common.metrics import mse, nmse, occupancy_mismatch_rate, snr_db

Array1D = np.ndarray
NonlinearMode = Literal["identity", "signed_log1p", "asinh"]


@dataclass(frozen=True)
class StandardizationSideInfo:
    mean: float
    std: float


@dataclass(frozen=True)
class FWHTConfig:
    decimation_factor_bins: int = 2
    top_k_coeffs: int = 128
    quant_step: float = 0.02
    nonlinear_mode: NonlinearMode = "signed_log1p"
    nonlinear_alpha: float = 1.5
    side_info_bits_per_param: int = 16
    value_bits_per_coeff: int = 16


@dataclass(frozen=True)
class FWHTPacket:
    original_length_bins: int
    decimated_length_bins: int
    hadamard_length_bins: int
    side_info: StandardizationSideInfo
    topk_indices: Array1D
    quantized_values: Array1D
    quant_step: float
    nonlinear_mode: NonlinearMode
    nonlinear_alpha: float


def decimate_frequency_bins(frame_psd: Array1D, decimation_factor_bins: int) -> Array1D:
    if frame_psd.ndim != 1:
        raise ValueError("frame_psd must be one-dimensional")
    if decimation_factor_bins < 1:
        raise ValueError("decimation_factor_bins must be >= 1")
    if decimation_factor_bins == 1:
        return frame_psd.astype(np.float64, copy=True)

    trimmed_length = (frame_psd.size // decimation_factor_bins) * decimation_factor_bins
    if trimmed_length == 0:
        raise ValueError("decimation_factor_bins is too large for frame length")
    trimmed = frame_psd[:trimmed_length]
    return trimmed.reshape(-1, decimation_factor_bins).mean(axis=1).astype(np.float64)


def upsample_linear(values: Array1D, output_length: int) -> Array1D:
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if output_length <= 0:
        raise ValueError("output_length must be > 0")
    if values.size == output_length:
        return values.astype(np.float64, copy=True)
    x_src = np.linspace(0.0, 1.0, values.size, dtype=np.float64)
    x_dst = np.linspace(0.0, 1.0, output_length, dtype=np.float64)
    return np.interp(x_dst, x_src, values.astype(np.float64, copy=False))


def standardize_frame(frame: Array1D) -> Tuple[Array1D, StandardizationSideInfo]:
    mu = float(np.mean(frame))
    sigma = float(np.std(frame))
    if sigma < 1e-12:
        sigma = 1.0
    return ((frame - mu) / sigma).astype(np.float64), StandardizationSideInfo(
        mean=mu, std=sigma
    )


def destandardize_frame(
    frame_standardized: Array1D, side_info: StandardizationSideInfo
) -> Array1D:
    return (frame_standardized * side_info.std + side_info.mean).astype(np.float64)


def apply_nonlinear_map(values: Array1D, mode: NonlinearMode, alpha: float) -> Array1D:
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")
    if mode == "identity":
        return values.astype(np.float64)
    if mode == "signed_log1p":
        return (np.sign(values) * np.log1p(alpha * np.abs(values)) / alpha).astype(
            np.float64
        )
    if mode == "asinh":
        return (np.arcsinh(alpha * values) / alpha).astype(np.float64)
    raise ValueError(f"Unsupported nonlinear mode: {mode}")


def invert_nonlinear_map(
    transformed: Array1D, mode: NonlinearMode, alpha: float
) -> Array1D:
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")
    if mode == "identity":
        return transformed.astype(np.float64)
    if mode == "signed_log1p":
        return (
            np.sign(transformed) * (np.expm1(alpha * np.abs(transformed)) / alpha)
        ).astype(np.float64)
    if mode == "asinh":
        return (np.sinh(alpha * transformed) / alpha).astype(np.float64)
    raise ValueError(f"Unsupported nonlinear mode: {mode}")


def next_power_of_two(n: int) -> int:
    if n < 1:
        raise ValueError("n must be >= 1")
    return 1 if n == 1 else 1 << int(ceil(log2(n)))


def pad_with_zeros(values: Array1D, output_length: int) -> Array1D:
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if output_length < values.size:
        raise ValueError("output_length must be >= len(values)")
    padded = np.zeros(output_length, dtype=np.float64)
    padded[: values.size] = values.astype(np.float64, copy=False)
    return padded


def _assert_power_of_two(n: int) -> None:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("length must be a power of two")


def fwht_orthonormal(values: Array1D) -> Array1D:
    x = values.astype(np.float64, copy=True)
    n = x.size
    _assert_power_of_two(n)
    h = 1
    while h < n:
        step = h * 2
        for i in range(0, n, step):
            a = x[i : i + h].copy()
            b = x[i + h : i + step].copy()
            x[i : i + h] = a + b
            x[i + h : i + step] = a - b
        h = step
    x /= np.sqrt(n)
    return x


def inverse_fwht_orthonormal(values: Array1D) -> Array1D:
    # Hadamard transform is self-inverse under orthonormal scaling.
    return fwht_orthonormal(values)


def select_topk_coefficients(
    coefficients: Array1D, top_k: int
) -> Tuple[Array1D, Array1D]:
    """Return the ``top_k`` highest-magnitude coefficients and their indices."""

    n = coefficients.size
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if top_k > n:
        raise ValueError(f"top_k={top_k} exceeds the available coefficient count {n}")

    k = int(top_k)
    # argpartition is O(n) average and deterministic for fixed input.
    idx_unsorted = np.argpartition(np.abs(coefficients), -k)[-k:]
    # Sort indices for stable packet serialization.
    idx = np.sort(idx_unsorted.astype(np.int64))
    values = coefficients[idx].astype(np.float64)
    return idx, values


def quantize_uniform(values: Array1D, quant_step: float) -> Array1D:
    if quant_step <= 0.0:
        raise ValueError("quant_step must be > 0")
    return np.round(values / quant_step).astype(np.int32)


def dequantize_uniform(quantized_values: Array1D, quant_step: float) -> Array1D:
    if quant_step <= 0.0:
        raise ValueError("quant_step must be > 0")
    return (quantized_values.astype(np.float64) * quant_step).astype(np.float64)


def build_sparse_vector(length: int, indices: Array1D, values: Array1D) -> Array1D:
    out = np.zeros(int(length), dtype=np.float64)
    out[indices.astype(np.int64)] = values.astype(np.float64)
    return out


def compress_fwht_frame(frame_psd: Array1D, config: FWHTConfig) -> FWHTPacket:
    frame_decimated = decimate_frequency_bins(frame_psd, config.decimation_factor_bins)
    frame_standardized, side_info = standardize_frame(frame_decimated)
    sigma_t = apply_nonlinear_map(
        frame_standardized, mode=config.nonlinear_mode, alpha=config.nonlinear_alpha
    )

    hadamard_length = next_power_of_two(sigma_t.size)
    sigma_padded = pad_with_zeros(sigma_t, hadamard_length)
    coefficients = fwht_orthonormal(sigma_padded)

    topk_indices, topk_values = select_topk_coefficients(
        coefficients, config.top_k_coeffs
    )
    quantized_values = quantize_uniform(topk_values, config.quant_step)

    return FWHTPacket(
        original_length_bins=int(frame_psd.size),
        decimated_length_bins=int(frame_decimated.size),
        hadamard_length_bins=int(hadamard_length),
        side_info=side_info,
        topk_indices=topk_indices,
        quantized_values=quantized_values,
        quant_step=float(config.quant_step),
        nonlinear_mode=config.nonlinear_mode,
        nonlinear_alpha=float(config.nonlinear_alpha),
    )


def decompress_fwht_frame(packet: FWHTPacket) -> Array1D:
    dequantized = dequantize_uniform(packet.quantized_values, packet.quant_step)
    sparse_coefficients = build_sparse_vector(
        packet.hadamard_length_bins, packet.topk_indices, dequantized
    )

    sigma_padded = inverse_fwht_orthonormal(sparse_coefficients)
    sigma_t = sigma_padded[: packet.decimated_length_bins]

    frame_standardized = invert_nonlinear_map(
        sigma_t, mode=packet.nonlinear_mode, alpha=packet.nonlinear_alpha
    )
    frame_decimated = destandardize_frame(frame_standardized, packet.side_info)
    frame_reconstructed = upsample_linear(frame_decimated, packet.original_length_bins)
    return frame_reconstructed.astype(np.float64)


def estimate_payload_bits(packet: FWHTPacket, config: FWHTConfig) -> int:
    index_bits = int(np.ceil(np.log2(max(packet.hadamard_length_bins, 2))))
    side_bits = 2 * int(config.side_info_bits_per_param)
    coeff_bits = int(packet.topk_indices.size) * (
        index_bits + int(config.value_bits_per_coeff)
    )
    return int(side_bits + coeff_bits)


def reconstruction_metrics(
    original_frame: Array1D,
    reconstructed_frame: Array1D,
    occupancy_margin_db: float = 3.0,
) -> dict:
    return {
        "mse": mse(original_frame, reconstructed_frame),
        "nmse": nmse(original_frame, reconstructed_frame),
        "snr_db": snr_db(original_frame, reconstructed_frame),
        "occupancy_mismatch": occupancy_mismatch_rate(
            original_frame, reconstructed_frame, margin_db=occupancy_margin_db
        ),
    }
