# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import scipy.sparse as scipy_sparse
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness

from cuml.manifold.umap import UMAP as cuUMAP


def test_densmap_fit_transform_no_nans():
    """densMAP fit_transform produces finite embeddings on blobs."""
    data, _ = make_blobs(
        n_samples=500, n_features=10, centers=5, random_state=42
    )
    model = cuUMAP(
        densmap=True,
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
    )
    embedding = model.fit_transform(data, convert_dtype=True)
    assert embedding.shape == (500, 2)
    assert not np.isnan(embedding).any()
    assert not np.isinf(embedding).any()


def test_densmap_trustworthiness_on_iris():
    """densMAP should produce a reasonable embedding on iris."""
    iris = datasets.load_iris()
    embedding = cuUMAP(
        densmap=True,
        n_neighbors=10,
        min_dist=0.01,
        random_state=0,
    ).fit_transform(iris.data, convert_dtype=True)

    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert trust >= 0.90


def test_densmap_supervised_fit():
    """densMAP works with supervised labels."""
    iris = datasets.load_iris()
    model = cuUMAP(
        densmap=True,
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
    )
    embedding = model.fit_transform(iris.data, iris.target, convert_dtype=True)
    assert embedding.shape == (150, 2)
    assert not np.isnan(embedding).any()


def test_densmap_reproducibility():
    """densMAP with fixed random_state produces similar embeddings.

    Exact bitwise equality is not expected because densMAP scatter-add
    kernels use atomicAdd, whose floating-point accumulation order varies
    across runs.
    """
    data, _ = make_blobs(
        n_samples=500, n_features=10, centers=5, random_state=42
    )
    kwargs = dict(
        densmap=True,
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        init="random",
        build_algo="brute_force_knn",
    )
    e1 = cuUMAP(**kwargs).fit_transform(data, convert_dtype=True)
    e2 = cuUMAP(**kwargs).fit_transform(data, convert_dtype=True)

    mean_diff = np.mean(np.abs(e1 - e2))
    assert mean_diff < 1.0, f"Mean absolute difference too large: {mean_diff}"


def test_densmap_transform_raises():
    """transform() must raise NotImplementedError when densmap=True."""
    data, _ = make_blobs(n_samples=200, n_features=10, random_state=42)
    model = cuUMAP(densmap=True, n_neighbors=10, random_state=42)
    model.fit(data, convert_dtype=True)

    with pytest.raises(NotImplementedError, match="densMAP"):
        model.transform(data, convert_dtype=True)


def test_densmap_sparse_input_raises():
    """densMAP must reject sparse input."""
    data, _ = make_blobs(n_samples=200, n_features=10, random_state=42)
    sparse_data = scipy_sparse.csr_matrix(data)

    model = cuUMAP(densmap=True, n_neighbors=10, random_state=42)
    with pytest.raises(NotImplementedError, match="sparse"):
        model.fit(sparse_data)


@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
        "l2",
        "sqeuclidean",
        "manhattan",
        "chebyshev",
        "cosine",
        "correlation",
        "canberra",
        "hellinger",
        "hamming",
    ],
)
def test_densmap_multiple_metrics(metric):
    """densMAP produces finite, non-degenerate embeddings for every supported metric."""
    data, _ = make_blobs(
        n_samples=500, n_features=10, centers=5, random_state=42
    )

    # Hellinger requires non-negative inputs
    if metric == "hellinger":
        data = np.abs(data)

    embedding = cuUMAP(
        densmap=True,
        metric=metric,
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        build_algo="brute_force_knn",
    ).fit_transform(data, convert_dtype=True)

    assert embedding.shape == (500, 2)
    assert not np.isnan(embedding).any(), (
        f"NaN in embedding for metric={metric}"
    )
    assert not np.isinf(embedding).any(), (
        f"Inf in embedding for metric={metric}"
    )


@pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 3.0])
def test_densmap_minkowski(p):
    """densMAP with Minkowski metric at various p values."""
    data, _ = make_blobs(
        n_samples=500, n_features=10, centers=5, random_state=42
    )
    embedding = cuUMAP(
        densmap=True,
        metric="minkowski",
        metric_kwds={"p": p},
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        build_algo="brute_force_knn",
    ).fit_transform(data, convert_dtype=True)

    assert embedding.shape == (500, 2)
    assert not np.isnan(embedding).any(), f"NaN in embedding for p={p}"
    assert not np.isinf(embedding).any(), f"Inf in embedding for p={p}"


def test_densmap_metric_changes_embedding():
    """Different metrics should produce meaningfully different embeddings."""
    data, _ = make_blobs(
        n_samples=500, n_features=10, centers=5, random_state=42
    )
    kwargs = dict(
        densmap=True,
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        init="random",
        build_algo="brute_force_knn",
    )
    e_euclidean = cuUMAP(metric="euclidean", **kwargs).fit_transform(
        data, convert_dtype=True
    )
    e_cosine = cuUMAP(metric="cosine", **kwargs).fit_transform(
        data, convert_dtype=True
    )
    e_manhattan = cuUMAP(metric="manhattan", **kwargs).fit_transform(
        data, convert_dtype=True
    )

    # Embeddings from different metrics should not be identical
    assert not np.allclose(e_euclidean, e_cosine, atol=0.1), (
        "euclidean and cosine embeddings are unexpectedly similar"
    )
    assert not np.allclose(e_euclidean, e_manhattan, atol=0.1), (
        "euclidean and manhattan embeddings are unexpectedly similar"
    )


@pytest.mark.parametrize(
    "param,value,match",
    [
        ("dens_lambda", -1.0, "dens_lambda"),
        ("dens_frac", -0.1, "dens_frac"),
        ("dens_frac", 1.5, "dens_frac"),
        ("dens_var_shift", -0.5, "dens_var_shift"),
    ],
)
def test_densmap_invalid_params(param, value, match):
    """densMAP parameter validation rejects invalid values."""
    data, _ = make_blobs(n_samples=200, n_features=10, random_state=42)
    model = cuUMAP(densmap=True, n_neighbors=10, **{param: value})
    with pytest.raises(ValueError, match=match):
        model.fit(data)


def test_densmap_params_preserved():
    """densMAP parameters are stored and retrievable."""
    model = cuUMAP(
        densmap=True,
        dens_lambda=3.0,
        dens_frac=0.5,
        dens_var_shift=0.2,
    )
    assert model.densmap is True
    assert model.dens_lambda == 3.0
    assert model.dens_frac == 0.5
    assert model.dens_var_shift == 0.2


def test_densmap_custom_hyperparams():
    """densMAP works with non-default dens_lambda / dens_frac."""
    data, _ = make_blobs(
        n_samples=400, n_features=10, centers=5, random_state=42
    )
    embedding = cuUMAP(
        densmap=True,
        dens_lambda=5.0,
        dens_frac=0.5,
        dens_var_shift=0.05,
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
    ).fit_transform(data, convert_dtype=True)

    assert embedding.shape == (400, 2)
    assert not np.isnan(embedding).any()


def test_densmap_disabled_matches_standard():
    """densmap=False should behave identically to default UMAP."""
    data, _ = make_blobs(
        n_samples=300, n_features=10, centers=5, random_state=42
    )
    kwargs = dict(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        init="random",
        build_algo="brute_force_knn",
    )
    e_standard = cuUMAP(**kwargs).fit_transform(data, convert_dtype=True)
    e_off = cuUMAP(densmap=False, **kwargs).fit_transform(
        data, convert_dtype=True
    )

    np.testing.assert_array_equal(e_standard, e_off)


def test_densmap_get_param_names():
    """densMAP parameters appear in _get_param_names."""
    names = cuUMAP._get_param_names()
    for p in ("densmap", "dens_lambda", "dens_frac", "dens_var_shift"):
        assert p in names
