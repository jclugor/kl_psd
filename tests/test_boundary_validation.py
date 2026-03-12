from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from psd_compression.common.io import load_psd_dataset, load_yaml_config
from psd_compression.fwht.codec import select_topk_coefficients
from psd_compression.kl_pca.model import fit_kl_pca


def test_load_yaml_config_rejects_empty_document() -> None:
    with TemporaryDirectory() as tmp_dir:
        config_path = Path(tmp_dir) / "empty.yaml"
        config_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="YAML config is empty"):
            load_yaml_config(config_path)


def test_load_psd_dataset_rejects_non_matrix_x() -> None:
    with TemporaryDirectory() as tmp_dir:
        dataset_path = Path(tmp_dir) / "bad_dataset.npz"
        np.savez(dataset_path, X=np.array([1.0, 2.0, 3.0], dtype=np.float64))

        with pytest.raises(ValueError, match="shape \\[num_frames, num_bins\\]"):
            load_psd_dataset(dataset_path)


def test_fit_kl_pca_rejects_nonpositive_component_count() -> None:
    frames = np.ones((4, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="n_components must be >= 1"):
        fit_kl_pca(frames, n_components=0)


def test_select_topk_coefficients_rejects_nonpositive_top_k() -> None:
    coeffs = np.array([1.0, -2.0, 0.5], dtype=np.float64)

    with pytest.raises(ValueError, match="top_k must be >= 1"):
        select_topk_coefficients(coeffs, top_k=0)
