#
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import platform

import sklearn.metrics
from cuml.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import v_measure_score as sklearn_v_measure_score
from scipy.special import rel_entr as scipy_kl_divergence
from sklearn.metrics import pairwise_distances as sklearn_pairwise_distances
from cuml.metrics import (
    pairwise_distances,
    sparse_pairwise_distances,
    PAIRWISE_DISTANCE_METRICS,
    PAIRWISE_DISTANCE_SPARSE_METRICS,
)
from sklearn.metrics import (
    precision_recall_curve as sklearn_precision_recall_curve,
)
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
from cuml.metrics import log_loss
from cuml.metrics import precision_recall_curve
from cuml.metrics import roc_auc_score
from cuml.common.sparsefuncs import csr_row_normalize_l1
from cuml.common import has_scipy
from cuml.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from cuml.model_selection import train_test_split
from cuml.metrics.cluster import entropy
from cuml.metrics import kl_divergence as cu_kl_divergence
from cuml.metrics import hinge_loss as cuml_hinge
from cuml import LogisticRegression as cu_log
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import silhouette_samples as sk_silhouette_samples
from sklearn.metrics.cluster import silhouette_score as sk_silhouette_score
from sklearn.metrics.cluster import mutual_info_score as sk_mutual_info_score
from sklearn.metrics.cluster import completeness_score as sk_completeness_score
from sklearn.metrics.cluster import homogeneity_score as sk_homogeneity_score
from sklearn.metrics.cluster import adjusted_rand_score as sk_ars
from sklearn.metrics import log_loss as sklearn_log_loss
from sklearn.datasets import make_classification, make_blobs
from sklearn.metrics import hinge_loss as sk_hinge
from cuml.internals.safe_imports import cpu_only_import_from
from cuml.internals.safe_imports import gpu_only_import_from
from cuml.testing.utils import (
    get_handle,
    get_pattern,
    array_equal,
    unit_param,
    quality_param,
    stress_param,
    generate_random_labels,
    score_labeling_with_handle,
)
from cuml.metrics.cluster import silhouette_samples as cu_silhouette_samples
from cuml.metrics.cluster import silhouette_score as cu_silhouette_score
from cuml.metrics.cluster import adjusted_rand_score as cu_ars
from cuml.internals.safe_imports import cpu_only_import
import pytest

import random
from itertools import chain, permutations

import cuml
import cuml.internals.logger as logger
from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")
np = cpu_only_import("numpy")
cudf = gpu_only_import("cudf")


cuda = gpu_only_import_from("numba", "cuda")
assert_almost_equal = cpu_only_import_from(
    "numpy.testing", "assert_almost_equal"
)


scipy_pairwise_distances = cpu_only_import_from("scipy.spatial", "distance")

IS_ARM = platform.processor() == "aarch64"


@pytest.fixture(scope="module")
def random_state():
    random_state = random.randint(0, 10**6)
    with logger.set_level(logger.level_enum.debug):
        logger.debug("Random seed: {}".format(random_state))
    return random_state


@pytest.fixture(
    scope="module",
    params=(
        {
            "n_clusters": 2,
            "n_features": 2,
            "label_type": "int64",
            "data_type": "float32",
        },
        {
            "n_clusters": 5,
            "n_features": 1000,
            "label_type": "int32",
            "data_type": "float64",
        },
    ),
)
def labeled_clusters(request, random_state):
    data, labels = make_blobs(
        n_samples=1000,
        n_features=request.param["n_features"],
        random_state=random_state,
        centers=request.param["n_clusters"],
        center_box=(-1, 1),
        cluster_std=1.5,  # Allow some cluster overlap
    )

    return (
        data.astype(request.param["data_type"]),
        labels.astype(request.param["label_type"]),
    )


# Ignore FutureWarning: Using `__dataframe__` is deprecated
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_sklearn_search():
    """Test ensures scoring function works with sklearn machinery"""
    import numpy as np
    from cuml import Ridge as cumlRidge
    import cudf
    from sklearn import datasets
    from sklearn.model_selection import train_test_split, GridSearchCV

    diabetes = datasets.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data,
        diabetes.target,
        test_size=0.2,
        shuffle=False,
        random_state=1,
    )

    alpha = np.array([1.0])
    fit_intercept = True
    normalize = False

    params = {"alpha": np.logspace(-3, -1, 10)}
    cu_clf = cumlRidge(
        alpha=alpha,
        fit_intercept=fit_intercept,
        normalize=normalize,
        solver="eig",
    )

    assert getattr(cu_clf, "score", False)
    sk_cu_grid = GridSearchCV(cu_clf, params, cv=5)

    gdf_data = cudf.DataFrame(X_train)
    gdf_train = cudf.DataFrame(dict(train=y_train))

    sk_cu_grid.fit(gdf_data, gdf_train.train)
    assert_almost_equal(sk_cu_grid.best_params_["alpha"], 0.1)


@pytest.mark.parametrize(
    "true_kind, pred_kind, true_dtype, pred_dtype",
    [
        ("cupy", "cupy", "int32", "int32"),
        ("cupy", "numpy", "int64", "int32"),
        ("numpy", "cupy", "float16", "float32"),
        ("cudf", "cudf", "int32", "int64"),
        ("numpy", "cudf", "object", "object"),
        ("numpy", "cudf", "object", "category"),
        ("cudf", "numpy", "category", "object"),
        ("cudf", "cudf", "object", "object"),
        ("cudf", "cudf", "object", "category"),
        ("cudf", "cudf", "category", "object"),
        ("cudf", "cudf", "category", "category"),
    ],
)
@pytest.mark.parametrize(
    "weight_kind, weight_dtype",
    [(None, None), ("cupy", "float32"), ("numpy", "float64")],
)
@pytest.mark.parametrize("normalize", [True, False])
def test_accuracy_score(
    true_kind,
    pred_kind,
    true_dtype,
    pred_dtype,
    weight_kind,
    weight_dtype,
    normalize,
):
    N = 30
    rng = np.random.RandomState(42)
    np_true = rng.randint(0, 3, N)
    np_pred = (rng.randint(0, 2, N) + np_true) % 3
    np_weight = rng.random(N).astype(weight_dtype) if weight_kind else None
    if true_dtype in ("object", "category"):
        assert pred_dtype in ("object", "category")
        labels = np.array(["a", "b", "c"], dtype="object")
        np_true = labels.take(np_true)
        np_pred = labels.take(np_pred)
    else:
        np_true = np_true.astype(true_dtype)
        np_pred = np_pred.astype(pred_dtype)

    def convert(x, kind, dtype):
        if kind == "cupy":
            return cp.array(x, dtype=dtype)
        elif kind == "cudf":
            return cudf.Series(x).astype(dtype)
        else:
            return x

    true = convert(np_true, true_kind, true_dtype)
    pred = convert(np_pred, pred_kind, true_dtype)
    weight = (
        convert(np_weight, weight_kind, weight_dtype) if weight_kind else None
    )

    sol = sklearn.metrics.accuracy_score(
        np_true, np_pred, sample_weight=np_weight, normalize=normalize
    )
    res = cuml.metrics.accuracy_score(
        true, pred, sample_weight=weight, normalize=normalize
    )
    assert isinstance(res, float)
    np.testing.assert_allclose(res, sol)


@pytest.mark.parametrize("true_kind", ["pandas", "cudf"])
@pytest.mark.parametrize("pred_kind", ["pandas", "cudf"])
def test_accuracy_score_index_unaligned(true_kind, pred_kind):
    """Check that accuracy_score ignores the index of the input."""
    pd = pytest.importorskip("pandas")
    true_ns = pd if true_kind == "pandas" else cudf
    pred_ns = pd if pred_kind == "pandas" else cudf

    true = true_ns.Series([1, 2, 1, 2], index=[10, 2, 4, 6])
    pred = pred_ns.Series([1, 2, 2, 1], index=[2, 4, 6, 8])
    assert cuml.metrics.accuracy_score(true, pred) == 0.5


def test_accuracy_score_errors():
    arr_3 = np.array([1, 2, 3])
    arr_4 = np.array([1, 2, 3, 4])
    arr_3x3 = np.ones((3, 3))

    with pytest.raises(ValueError, match="Expected 3 rows"):
        cuml.metrics.accuracy_score(arr_3, arr_4)

    with pytest.raises(ValueError, match="Expected 3 rows"):
        cuml.metrics.accuracy_score(arr_3, arr_3, sample_weight=arr_4)

    for true, pred, sw in [
        (arr_3x3, arr_3, None),
        (arr_3, arr_3x3, None),
        (arr_3, arr_3, arr_3x3),
    ]:
        with pytest.raises(ValueError, match="Expected 1 column"):
            cuml.metrics.accuracy_score(true, pred, sample_weight=sw)


dataset_names = ["noisy_circles", "noisy_moons", "aniso"] + [
    pytest.param(ds, marks=pytest.mark.xfail) for ds in ["blobs", "varied"]
]


@pytest.mark.parametrize("name", dataset_names)
@pytest.mark.parametrize(
    "nrows", [unit_param(20), quality_param(5000), stress_param(500000)]
)
def test_rand_index_score(name, nrows):

    default_base = {
        "quantile": 0.3,
        "eps": 0.3,
        "damping": 0.9,
        "preference": -200,
        "n_neighbors": 10,
        "n_clusters": 3,
    }

    pat = get_pattern(name, nrows)

    params = default_base.copy()
    params.update(pat[1])

    cuml_kmeans = cuml.KMeans(n_clusters=params["n_clusters"], n_init="auto")

    X, y = pat[0]

    X = StandardScaler().fit_transform(X)

    cu_y_pred = cuml_kmeans.fit_predict(X)

    cu_score = cu_ars(y, cu_y_pred)
    cu_score_using_sk = sk_ars(y, cp.asnumpy(cu_y_pred))

    assert array_equal(cu_score, cu_score_using_sk)


@pytest.mark.parametrize(
    "metric", ("cityblock", "cosine", "euclidean", "l1", "sqeuclidean")
)
@pytest.mark.parametrize("chunk_divider", [1, 3, 5])
@pytest.mark.skipif(
    IS_ARM,
    reason="Test fails unexpectedly on ARM. "
    "github.com/rapidsai/cuml/issues/5025",
)
def test_silhouette_score_batched(metric, chunk_divider, labeled_clusters):

    X, labels = labeled_clusters
    cuml_score = cu_silhouette_score(
        X, labels, metric=metric, chunksize=int(X.shape[0] / chunk_divider)
    )
    sk_score = sk_silhouette_score(X, labels, metric=metric)
    assert_almost_equal(cuml_score, sk_score, decimal=2)


@pytest.mark.parametrize(
    "metric", ("cityblock", "cosine", "euclidean", "l1", "sqeuclidean")
)
@pytest.mark.parametrize("chunk_divider", [1, 3, 5])
def test_silhouette_samples_batched(metric, chunk_divider, labeled_clusters):
    X, labels = labeled_clusters
    cuml_scores = cu_silhouette_samples(
        X, labels, metric=metric, chunksize=int(X.shape[0] / chunk_divider)
    )
    sk_scores = sk_silhouette_samples(X, labels, metric=metric)

    cu_trunc = cp.around(cuml_scores, decimals=3)
    sk_trunc = cp.around(sk_scores, decimals=3)

    diff = cp.absolute(cu_trunc - sk_trunc) > 0
    over_diff = cp.all(diff)

    # 0.5% elements allowed to be different
    if len(over_diff.shape) > 0:
        assert over_diff.shape[0] <= 0.005 * X.shape[0]

    # different elements should not differ more than 1e-1
    tolerance_diff = cp.absolute(cu_trunc[diff] - sk_trunc[diff]) > 1e-1
    diff_change = cp.all(tolerance_diff)
    if len(diff_change.shape) > 0:
        assert False


@pytest.mark.xfail
def test_silhouette_score_batched_non_monotonic():
    vecs = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [10.0, 10.0, 10.0]]
    )
    labels = np.array([0, 0, 1, 3])

    cuml_samples = cu_silhouette_samples(X=vecs, labels=labels)
    sk_samples = sk_silhouette_samples(X=vecs, labels=labels)
    assert array_equal(cuml_samples, sk_samples)

    vecs = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [10.0, 10.0, 10.0]])
    labels = np.array([1, 1, 3])

    cuml_samples = cu_silhouette_samples(X=vecs, labels=labels)
    sk_samples = sk_silhouette_samples(X=vecs, labels=labels)
    assert array_equal(cuml_samples, sk_samples)


def score_homogeneity(ground_truth, predictions, use_handle):
    return score_labeling_with_handle(
        cuml.metrics.homogeneity_score,
        ground_truth,
        predictions,
        use_handle,
        dtype=np.int32,
    )


def score_completeness(ground_truth, predictions, use_handle):
    return score_labeling_with_handle(
        cuml.metrics.completeness_score,
        ground_truth,
        predictions,
        use_handle,
        dtype=np.int32,
    )


def score_mutual_info(ground_truth, predictions, use_handle):
    return score_labeling_with_handle(
        cuml.metrics.mutual_info_score,
        ground_truth,
        predictions,
        use_handle,
        dtype=np.int32,
    )


@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize(
    "data", [([0, 0, 1, 1], [1, 1, 0, 0]), ([0, 0, 1, 1], [0, 0, 1, 1])]
)
def test_homogeneity_perfect_labeling(use_handle, data):
    # Perfect labelings are homogeneous
    hom = score_homogeneity(*data, use_handle)
    assert_almost_equal(hom, 1.0, decimal=4)


@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize(
    "data", [([0, 0, 1, 1], [0, 0, 1, 2]), ([0, 0, 1, 1], [0, 1, 2, 3])]
)
def test_homogeneity_non_perfect_labeling(use_handle, data):
    # Non-perfect labelings that further split classes into more clusters can
    # be perfectly homogeneous
    hom = score_homogeneity(*data, use_handle)
    assert_almost_equal(hom, 1.0, decimal=4)


@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize(
    "data", [([0, 0, 1, 1], [0, 1, 0, 1]), ([0, 0, 1, 1], [0, 0, 0, 0])]
)
def test_homogeneity_non_homogeneous_labeling(use_handle, data):
    # Clusters that include samples from different classes do not make for an
    # homogeneous labeling
    hom = score_homogeneity(*data, use_handle)
    assert_almost_equal(hom, 0.0, decimal=4)


@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize("input_range", [[0, 1000], [-1000, 1000]])
def test_homogeneity_score_big_array(use_handle, input_range):
    a, b, _, _ = generate_random_labels(
        lambda rd: rd.randint(*input_range, int(10e4), dtype=np.int32)
    )
    score = score_homogeneity(a, b, use_handle)
    ref = sk_homogeneity_score(a, b)
    np.testing.assert_almost_equal(score, ref, decimal=4)


@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize(
    "input_range", [[0, 2], [-5, 20], [int(-10e2), int(10e2)]]
)
def test_homogeneity_completeness_symmetry(use_handle, input_range):
    a, b, _, _ = generate_random_labels(
        lambda rd: rd.randint(*input_range, int(10e3), dtype=np.int32)
    )
    hom = score_homogeneity(a, b, use_handle)
    com = score_completeness(b, a, use_handle)
    np.testing.assert_almost_equal(hom, com, decimal=4)


@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize(
    "input_labels",
    [
        ([0, 0, 1, 1], [1, 1, 0, 0]),
        ([0, 0, 1, 1], [0, 0, 1, 1]),
        ([0, 0, 1, 1], [0, 0, 1, 2]),
        ([0, 0, 1, 1], [0, 1, 2, 3]),
        ([0, 0, 1, 1], [0, 1, 0, 1]),
        ([0, 0, 1, 1], [0, 0, 0, 0]),
    ],
)
def test_mutual_info_score(use_handle, input_labels):
    score = score_mutual_info(*input_labels, use_handle)
    ref = sk_mutual_info_score(*input_labels)
    np.testing.assert_almost_equal(score, ref, decimal=4)


@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize("input_range", [[0, 1000], [-1000, 1000]])
def test_mutual_info_score_big_array(use_handle, input_range):
    a, b, _, _ = generate_random_labels(
        lambda rd: rd.randint(*input_range, int(10e4), dtype=np.int32)
    )
    score = score_mutual_info(a, b, use_handle)
    ref = sk_mutual_info_score(a, b)
    np.testing.assert_almost_equal(score, ref, decimal=4)


@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize("n", [14])
def test_mutual_info_score_range_equal_samples(use_handle, n):
    input_range = (-n, n)
    a, b, _, _ = generate_random_labels(
        lambda rd: rd.randint(*input_range, n, dtype=np.int32)
    )
    score = score_mutual_info(a, b, use_handle)
    ref = sk_mutual_info_score(a, b)
    np.testing.assert_almost_equal(score, ref, decimal=4)


@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize("input_range", [[0, 19], [0, 2], [-5, 20]])
@pytest.mark.parametrize("n_samples", [129, 258])
def test_mutual_info_score_many_blocks(use_handle, input_range, n_samples):
    a, b, _, _ = generate_random_labels(
        lambda rd: rd.randint(*input_range, n_samples, dtype=np.int32)
    )
    score = score_mutual_info(a, b, use_handle)
    ref = sk_mutual_info_score(a, b)
    np.testing.assert_almost_equal(score, ref, decimal=4)


@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize(
    "data", [([0, 0, 1, 1], [1, 1, 0, 0]), ([0, 0, 1, 1], [0, 0, 1, 1])]
)
def test_completeness_perfect_labeling(use_handle, data):
    # Perfect labelings are complete
    com = score_completeness(*data, use_handle)
    np.testing.assert_almost_equal(com, 1.0, decimal=4)


@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize(
    "data", [([0, 0, 1, 1], [0, 0, 0, 0]), ([0, 1, 2, 3], [0, 0, 1, 1])]
)
def test_completeness_non_perfect_labeling(use_handle, data):
    # Non-perfect labelings that assign all classes members to the same
    # clusters are still complete
    com = score_completeness(*data, use_handle)
    np.testing.assert_almost_equal(com, 1.0, decimal=4)


@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize(
    "data", [([0, 0, 1, 1], [0, 1, 0, 1]), ([0, 0, 0, 0], [0, 1, 2, 3])]
)
def test_completeness_non_complete_labeling(use_handle, data):
    # If classes members are split across different clusters, the assignment
    # cannot be complete
    com = score_completeness(*data, use_handle)
    np.testing.assert_almost_equal(com, 0.0, decimal=4)


@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize("input_range", [[0, 1000], [-1000, 1000]])
def test_completeness_score_big_array(use_handle, input_range):
    a, b, _, _ = generate_random_labels(
        lambda rd: rd.randint(*input_range, int(10e4), dtype=np.int32)
    )
    score = score_completeness(a, b, use_handle)
    ref = sk_completeness_score(a, b)
    np.testing.assert_almost_equal(score, ref, decimal=4)


@pytest.mark.parametrize("n_samples", [50, stress_param(500000)])
@pytest.mark.parametrize(
    "y_dtype", [np.int32, np.int64, np.float32, np.float64]
)
@pytest.mark.parametrize(
    "pred_dtype", [np.int32, np.int64, np.float32, np.float64]
)
@pytest.mark.parametrize(
    "func",
    [
        "r2_score",
        "mean_squared_error",
        "mean_absolute_error",
        "mean_squared_log_error",
    ],
)
def test_regression_metrics(n_samples, y_dtype, pred_dtype, func):
    rng = np.random.RandomState(42)
    y_true = rng.randint(10, 1000, n_samples).astype(y_dtype)
    y_pred = (rng.randint(-5, 5, n_samples) + y_true).astype(pred_dtype)

    cu_metric = getattr(cuml.metrics, func)
    sk_metric = getattr(sklearn.metrics, func)

    res = cu_metric(y_true, y_pred)
    ref = sk_metric(y_true, y_pred)
    assert_almost_equal(res, ref)


@pytest.mark.parametrize(
    "func",
    [
        "r2_score",
        "mean_squared_error",
        "mean_absolute_error",
        "mean_squared_log_error",
    ],
)
def test_regression_metrics_cudf(func):
    a = cudf.Series([1.1, 2.2, 3.3, 4.4])
    b = cudf.Series([0.1, 0.2, 0.3, 0.4])

    cu_metric = getattr(cuml.metrics, func)
    err1 = cu_metric(a, b)
    err2 = cu_metric(a.values, b.values)
    assert err1 == err2


@pytest.mark.parametrize(
    "func",
    [
        "r2_score",
        "mean_squared_error",
        "mean_absolute_error",
        "mean_squared_log_error",
    ],
)
def test_regression_metrics_zero_error(func):
    y_true = y_pred = np.ones(3)

    cu_metric = getattr(cuml.metrics, func)
    sk_metric = getattr(sklearn.metrics, func)

    res = cu_metric(y_true, y_pred)
    ref = sk_metric(y_true, y_pred)
    assert_almost_equal(res, ref)


@pytest.mark.parametrize(
    "func",
    [
        "r2_score",
        "mean_squared_error",
        "mean_absolute_error",
        "mean_squared_log_error",
    ],
)
def test_regression_metrics_multioutput(func):
    y_true = np.array([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 1]])
    y_pred = np.array([[0, 0, 0, 1], [1, 0, 1, 1], [0, 0, 0, 1]])

    cu_metric = getattr(cuml.metrics, func)
    sk_metric = getattr(sklearn.metrics, func)

    res = cu_metric(y_true, y_pred)
    sol = sk_metric(y_true, y_pred)
    assert_almost_equal(res, sol)


@pytest.mark.parametrize(
    "func",
    [
        "r2_score",
        "mean_squared_error",
        "mean_absolute_error",
        "mean_squared_log_error",
    ],
)
def test_regression_metrics_multioutput_raw_values(func):
    y_true = np.array([[1, 2], [2.5, 1], [4.5, 3], [5, 7]], dtype=float)
    y_pred = np.array([[1, 1], [2, 1], [5, 4], [5, 6.5]], dtype=float)

    cu_metric = getattr(cuml.metrics, func)
    sk_metric = getattr(sklearn.metrics, func)

    res = cu_metric(y_true, y_pred, multioutput="raw_values")
    sol = sk_metric(y_true, y_pred, multioutput="raw_values")
    cp.testing.assert_array_almost_equal(res, sol)


@pytest.mark.parametrize(
    "func",
    [
        "r2_score",
        "mean_squared_error",
        "mean_absolute_error",
        "mean_squared_log_error",
    ],
)
def test_regression_metrics_multioutput_custom_weights(func):
    y_true = np.array([[1, 2], [2.5, 1], [4.5, 3], [5, 7]], dtype=float)
    y_pred = np.array([[1, 1], [2, 1], [5, 4], [5, 6.5]], dtype=float)
    multioutput = np.array([0.3, 0.7])

    cu_metric = getattr(cuml.metrics, func)
    sk_metric = getattr(sklearn.metrics, func)

    res = cu_metric(y_true, y_pred, multioutput=multioutput)
    sol = sk_metric(y_true, y_pred, multioutput=multioutput)
    assert_almost_equal(res, sol)


@pytest.mark.parametrize(
    "func",
    [
        "r2_score",
        "mean_squared_error",
        "mean_absolute_error",
        "mean_squared_log_error",
    ],
)
@pytest.mark.parametrize("multioutput", [False, True])
def test_regression_metrics_sample_weight(multioutput, func):
    y_true = np.array([[1, 2], [2.5, 1], [4.5, 3], [5, 7]], dtype=float)
    y_pred = np.array([[1, 1], [2, 1], [5, 4], [5, 6.5]], dtype=float)
    if not multioutput:
        y_true = y_true[:, 0]
        y_pred = y_pred[:, 0]
    weights = np.array([0.2, 0.25, 0.4, 0.15], dtype=float)

    cu_metric = getattr(cuml.metrics, func)
    sk_metric = getattr(sklearn.metrics, func)

    res = cu_metric(y_true, y_pred, sample_weight=weights)
    sol = sk_metric(y_true, y_pred, sample_weight=weights)
    assert_almost_equal(res, sol)


@pytest.mark.parametrize("true, pred", [(1, 1), (0, 0)])
def test_r2_score_force_finite(true, pred):
    y_true = np.array([true] * 3, dtype="float64")
    y_pred = np.array([pred] * 3, dtype="float64")

    res = cuml.metrics.r2_score(y_true, y_pred)
    sol = sklearn.metrics.r2_score(y_true, y_pred)
    assert_almost_equal(res, sol)

    with pytest.warns(RuntimeWarning):
        res = cuml.metrics.r2_score(y_true, y_pred, force_finite=False)
        sol = sklearn.metrics.r2_score(y_true, y_pred, force_finite=False)

    assert_almost_equal(res, sol)


def test_r2_score_multioutput_variance_weighted():
    y_true = np.array([[1, 2], [2.5, 1], [4.5, 3], [5, 7]], dtype=float)
    y_pred = np.array([[1, 1], [2, 1], [5, 4], [5, 6.5]], dtype=float)

    res = cuml.metrics.r2_score(
        y_true, y_pred, multioutput="variance_weighted"
    )
    sol = sklearn.metrics.r2_score(
        y_true, y_pred, multioutput="variance_weighted"
    )
    assert_almost_equal(res, sol)


@pytest.mark.parametrize(
    "inputs",
    [
        ([-1.0], [-1.0]),
        ([1.0, 2.0, 3.0], [1.0, -2.0, 3.0]),
        ([1.0, -2.0, 3.0], [1.0, 2.0, 3.0]),
    ],
)
def test_mean_squared_log_error_negative_values(inputs):
    with pytest.raises(ValueError, match="targets contain negative values"):
        cuml.metrics.mean_squared_log_error(
            np.array(inputs[0]), np.array(inputs[1])
        )


@pytest.mark.parametrize("use_handle", [True, False])
def test_entropy(use_handle):
    handle, stream = get_handle(use_handle)

    # The outcome of a fair coin is the most uncertain:
    # in base 2 the result is 1 (One bit of entropy).
    cluster = np.array([0, 1], dtype=np.int32)
    assert_almost_equal(entropy(cluster, base=2.0, handle=handle), 1.0)

    # The outcome of a biased coin is less uncertain:
    cluster = np.array(([0] * 9) + [1], dtype=np.int32)
    assert_almost_equal(entropy(cluster, base=2.0, handle=handle), 0.468995593)
    # base e
    assert_almost_equal(entropy(cluster, handle=handle), 0.32508297339144826)


@pytest.mark.parametrize("n_samples", [50, stress_param(500000)])
@pytest.mark.parametrize("base", [None, 2, 10, 50])
@pytest.mark.parametrize("use_handle", [True, False])
def test_entropy_random(n_samples, base, use_handle):
    if has_scipy():
        from scipy.stats import entropy as sp_entropy
    else:
        pytest.skip("Skipping test_entropy_random because Scipy is missing")

    handle, stream = get_handle(use_handle)

    clustering, _, _, _ = generate_random_labels(
        lambda rng: rng.randint(0, 1000, n_samples)
    )

    # generate unormalized probabilities from clustering
    pk = np.bincount(clustering)

    # scipy's entropy uses probabilities
    sp_S = sp_entropy(pk, base=base)
    # we use a clustering
    S = entropy(np.array(clustering, dtype=np.int32), base, handle=handle)

    assert_almost_equal(S, sp_S, decimal=2)


def test_confusion_matrix():
    y_true = cp.array([2, 0, 2, 2, 0, 1])
    y_pred = cp.array([0, 0, 2, 2, 0, 2])
    cm = confusion_matrix(y_true, y_pred)
    ref = cp.array([[2, 0, 0], [0, 0, 1], [1, 0, 2]])
    cp.testing.assert_array_equal(cm, ref)


def test_confusion_matrix_binary():
    y_true = cp.array([0, 1, 0, 1])
    y_pred = cp.array([1, 1, 1, 0])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    ref = cp.array([0, 2, 1, 1])
    cp.testing.assert_array_equal(ref, cp.array([tn, fp, fn, tp]))


@pytest.mark.parametrize("n_samples", [50, 3000, stress_param(500000)])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32])
@pytest.mark.parametrize("problem_type", ["binary", "multiclass"])
def test_confusion_matrix_random(n_samples, dtype, problem_type):
    upper_range = 2 if problem_type == "binary" else 1000

    y_true, y_pred, _, _ = generate_random_labels(
        lambda rng: rng.randint(0, upper_range, n_samples).astype(dtype)
    )
    convert_dtype = True if dtype == np.float32 else False
    cm = confusion_matrix(y_true, y_pred, convert_dtype=convert_dtype)
    ref = sk_confusion_matrix(y_true, y_pred)
    cp.testing.assert_array_almost_equal(ref, cm, decimal=4)


@pytest.mark.parametrize(
    "normalize, expected_results",
    [
        ("true", 0.333333333),
        ("pred", 0.333333333),
        ("all", 0.1111111111),
        (None, 2),
    ],
)
def test_confusion_matrix_normalize(normalize, expected_results):
    y_test = cp.array([0, 1, 2] * 6)
    y_pred = cp.array(list(chain(*permutations([0, 1, 2]))))
    cm = confusion_matrix(y_test, y_pred, normalize=normalize)
    cp.testing.assert_allclose(cm, cp.array(expected_results))


@pytest.mark.parametrize("labels", [(0, 1), (2, 1), (2, 1, 4, 7), (2, 20)])
def test_confusion_matrix_multiclass_subset_labels(labels):
    y_true, y_pred, _, _ = generate_random_labels(
        lambda rng: rng.randint(0, 3, 10).astype(np.int32)
    )

    ref = sk_confusion_matrix(y_true, y_pred, labels=labels)
    labels = cp.array(labels, dtype=np.int32)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cp.testing.assert_array_almost_equal(ref, cm, decimal=4)


@pytest.mark.parametrize("n_samples", [50, 3000, stress_param(500000)])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("weights_dtype", ["int", "float"])
def test_confusion_matrix_random_weights(n_samples, dtype, weights_dtype):
    y_true, y_pred, _, _ = generate_random_labels(
        lambda rng: rng.randint(0, 10, n_samples).astype(dtype)
    )

    if weights_dtype == "int":
        sample_weight = np.random.RandomState(0).randint(0, 10, n_samples)
    else:
        sample_weight = np.random.RandomState(0).rand(n_samples)

    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    ref = sk_confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    cp.testing.assert_array_almost_equal(ref, cm, decimal=4)


def test_roc_auc_score():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    assert_almost_equal(
        roc_auc_score(y_true, y_pred), sklearn_roc_auc_score(y_true, y_pred)
    )

    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0.8, 0.4, 0.4, 0.8, 0.8])
    assert_almost_equal(
        roc_auc_score(y_true, y_pred), sklearn_roc_auc_score(y_true, y_pred)
    )


@pytest.mark.parametrize("n_samples", [50, 500000])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_roc_auc_score_random(n_samples, dtype):

    y_true, _, _, _ = generate_random_labels(
        lambda rng: rng.randint(0, 2, n_samples).astype(dtype)
    )

    y_pred, _, _, _ = generate_random_labels(
        lambda rng: rng.randint(0, 1000, n_samples).astype(dtype)
    )

    auc = roc_auc_score(y_true, y_pred)
    skl_auc = sklearn_roc_auc_score(y_true, y_pred)
    assert_almost_equal(auc, skl_auc)


def test_roc_auc_score_at_limits():
    y_true = np.array([0.0, 0.0, 0.0], dtype=float)
    y_pred = np.array([0.0, 0.5, 1.0], dtype=float)

    err_msg = (
        "roc_auc_score cannot be used when "
        "only one class present in y_true. ROC AUC score "
        "is not defined in that case."
    )

    with pytest.raises(ValueError, match=err_msg):
        roc_auc_score(y_true, y_pred)

    y_true = np.array([0.0, 0.5, 1.0], dtype=float)
    y_pred = np.array([0.0, 0.5, 1.0], dtype=float)

    err_msg = "Continuous format of y_true  " "is not supported."

    with pytest.raises(ValueError, match=err_msg):
        roc_auc_score(y_true, y_pred)


@pytest.mark.skip(
    reason="shape discrepancy with sklearn 1.2"
    "https://github.com/rapidsai/cuml/issues/5164"
)
def test_precision_recall_curve():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    (
        precision_using_sk,
        recall_using_sk,
        thresholds_using_sk,
    ) = sklearn_precision_recall_curve(y_true, y_score)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    assert array_equal(precision, precision_using_sk)
    assert array_equal(recall, recall_using_sk)
    assert array_equal(thresholds, thresholds_using_sk)


def test_precision_recall_curve_at_limits():
    y_true = np.array([0.0, 0.0, 0.0], dtype=float)
    y_pred = np.array([0.0, 0.5, 1.0], dtype=float)

    err_msg = (
        "precision_recall_curve cannot be used when " "y_true is all zero."
    )

    with pytest.raises(ValueError, match=err_msg):
        precision_recall_curve(y_true, y_pred)

    y_true = np.array([0.0, 0.5, 1.0], dtype=float)
    y_pred = np.array([0.0, 0.5, 1.0], dtype=float)

    err_msg = "Continuous format of y_true  " "is not supported."

    with pytest.raises(ValueError, match=err_msg):
        precision_recall_curve(y_true, y_pred)


@pytest.mark.skip(
    reason="shape discrepancy with sklearn 1.2"
    "https://github.com/rapidsai/cuml/issues/5164"
)
@pytest.mark.parametrize("n_samples", [50, 500000])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_precision_recall_curve_random(n_samples, dtype):

    y_true, _, _, _ = generate_random_labels(
        lambda rng: rng.randint(0, 2, n_samples).astype(dtype)
    )

    y_score, _, _, _ = generate_random_labels(
        lambda rng: rng.randint(0, 1000, n_samples).astype(dtype)
    )

    (
        precision_using_sk,
        recall_using_sk,
        thresholds_using_sk,
    ) = sklearn_precision_recall_curve(y_true, y_score)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    assert array_equal(precision, precision_using_sk)
    assert array_equal(recall, recall_using_sk)
    assert array_equal(thresholds, thresholds_using_sk)


def test_log_loss():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    assert_almost_equal(
        log_loss(y_true, y_pred), sklearn_log_loss(y_true, y_pred)
    )

    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0.8, 0.4, 0.4, 0.8, 0.8])
    assert_almost_equal(
        log_loss(y_true, y_pred), sklearn_log_loss(y_true, y_pred)
    )


@pytest.mark.parametrize("n_samples", [500, 500000])
@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_log_loss_random(n_samples, dtype):

    y_true, _, _, _ = generate_random_labels(
        lambda rng: rng.randint(0, 10, n_samples).astype(dtype)
    )

    _, _, y_pred, _ = generate_random_labels(
        lambda rng: rng.rand(n_samples, 10)
    )
    # Make sure the probabilities sum to 1 per sample
    y_pred /= y_pred.sum(axis=1)[:, None]
    y_pred = cuda.to_device(y_pred)

    assert_almost_equal(
        log_loss(y_true, y_pred), sklearn_log_loss(y_true, y_pred)
    )


def test_log_loss_at_limits():
    y_true = np.array([0.0, 1.0, 2.0], dtype=float)
    y_pred = np.array([0.0, 0.5, 1.0], dtype=float)

    err_msg = "The shape of y_pred doesn't " "match the number of classes"

    with pytest.raises(ValueError, match=err_msg):
        log_loss(y_true, y_pred)

    y_true = np.array([0.0, 0.5, 1.0], dtype=float)
    y_pred = np.array([0.0, 0.5, 1.0], dtype=float)

    err_msg = "'y_true' can only have integer values"
    with pytest.raises(ValueError, match=err_msg):
        log_loss(y_true, y_pred)


def naive_kl_divergence_dist(X, Y):
    return 0.5 * np.array(
        [
            [
                np.sum(np.where(yj != 0, scipy_kl_divergence(xi, yj), 0.0))
                for yj in Y
            ]
            for xi in X
        ]
    )


def ref_dense_pairwise_dist(X, Y=None, metric=None, convert_dtype=False):
    # Select sklearn except for Hellinger that
    # sklearn doesn't support
    if Y is None:
        Y = X
    if metric == "hellinger":
        return naive_hellinger(X, Y)
    elif metric == "jensenshannon":
        return scipy_pairwise_distances.cdist(X, Y, "jensenshannon")
    elif metric == "kldivergence":
        return naive_kl_divergence_dist(X, Y)
    else:
        return sklearn_pairwise_distances(X, Y, metric)


def prep_dense_array(array, metric, col_major=0):
    if metric in ["hellinger", "jensenshannon", "kldivergence"]:
        norm_array = preprocessing.normalize(array, norm="l1")
        return np.asfortranarray(norm_array) if col_major else norm_array
    else:
        return np.asfortranarray(array) if col_major else array


@pytest.mark.parametrize("metric", PAIRWISE_DISTANCE_METRICS.keys())
@pytest.mark.parametrize(
    "matrix_size", [(5, 4), (1000, 3), (2, 10), (500, 400)]
)
@pytest.mark.parametrize("is_col_major", [True, False])
def test_pairwise_distances(metric: str, matrix_size, is_col_major):
    # Test the pairwise_distance helper function.
    rng = np.random.RandomState(0)

    compare_precision = 2 if metric == "nan_euclidean" else 4

    # Compare to sklearn, single input
    X = prep_dense_array(
        rng.random_sample(matrix_size), metric=metric, col_major=is_col_major
    )
    S = pairwise_distances(X, metric=metric)
    S2 = ref_dense_pairwise_dist(X, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Compare to sklearn, double input with same dimensions
    Y = X
    S = pairwise_distances(X, Y, metric=metric)
    S2 = ref_dense_pairwise_dist(X, Y, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Compare single and double inputs to each other
    S = pairwise_distances(X, metric=metric)
    S2 = pairwise_distances(X, Y, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Compare to sklearn, with Y dim != X dim
    Y = prep_dense_array(
        rng.random_sample((2, matrix_size[1])),
        metric=metric,
        col_major=is_col_major,
    )
    S = pairwise_distances(X, Y, metric=metric)
    S2 = ref_dense_pairwise_dist(X, Y, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Change precision of one parameter
    Y = np.asarray(Y, dtype=np.float32)
    S = pairwise_distances(X, Y, metric=metric)
    S2 = ref_dense_pairwise_dist(X, Y, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # For fp32, compare at 5 decimals, (2 places less than the ~7 max)
    compare_precision = 2

    # Change precision of both parameters to float
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    S = pairwise_distances(X, Y, metric=metric)
    S2 = ref_dense_pairwise_dist(X, Y, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Test sending an int type with convert_dtype=True
    if metric != "kldivergence":
        Y = prep_dense_array(
            rng.randint(10, size=Y.shape),
            metric=metric,
            col_major=is_col_major,
        )
        S = pairwise_distances(X, Y, metric=metric, convert_dtype=True)
        S2 = ref_dense_pairwise_dist(X, Y, metric=metric, convert_dtype=True)
        cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Test that uppercase on the metric name throws an error.
    with pytest.raises(ValueError):
        pairwise_distances(X, Y, metric=metric.capitalize())


@pytest.mark.parametrize("metric", PAIRWISE_DISTANCE_METRICS.keys())
@pytest.mark.parametrize(
    "matrix_size",
    [
        unit_param((1000, 100)),
        quality_param((2000, 1000)),
        stress_param((10000, 10000)),
    ],
)
def test_pairwise_distances_sklearn_comparison(metric: str, matrix_size):
    # Test larger sizes to sklearn
    rng = np.random.RandomState(1)

    element_count = matrix_size[0] * matrix_size[1]

    X = prep_dense_array(
        rng.random_sample(matrix_size), metric=metric, col_major=0
    )
    Y = prep_dense_array(
        rng.random_sample(matrix_size), metric=metric, col_major=0
    )

    # For fp64, compare at 10 decimals, (5 places less than the ~15 max)
    compare_precision = 10

    print(X.shape, Y.shape, metric)
    # Compare to sklearn, fp64
    S = pairwise_distances(X, Y, metric=metric)

    if element_count <= 2000000:
        S2 = ref_dense_pairwise_dist(X, Y, metric=metric)
        cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # For fp32, compare at 4 decimals, (3 places less than the ~7 max)
    compare_precision = 4

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    # Compare to sklearn, fp32
    S = pairwise_distances(X, Y, metric=metric)

    if element_count <= 2000000:
        S2 = ref_dense_pairwise_dist(X, Y, metric=metric)
        cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)


@pytest.mark.parametrize("metric", PAIRWISE_DISTANCE_METRICS.keys())
def test_pairwise_distances_one_dimension_order(metric: str):
    # Test the pairwise_distance helper function for 1 dimensional cases which
    # can break down when using a size of 1 for either dimension
    rng = np.random.RandomState(2)

    Xc = prep_dense_array(
        rng.random_sample((1, 4)), metric=metric, col_major=0
    )
    Yc = prep_dense_array(
        rng.random_sample((10, 4)), metric=metric, col_major=0
    )
    Xf = np.asfortranarray(Xc)
    Yf = np.asfortranarray(Yc)

    # For fp64, compare at 13 decimals, (2 places less than the ~15 max)
    compare_precision = 13

    # Compare to sklearn, C/C order
    S = pairwise_distances(Xc, Yc, metric=metric)
    S2 = ref_dense_pairwise_dist(Xc, Yc, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Compare to sklearn, C/F order
    S = pairwise_distances(Xc, Yf, metric=metric)
    S2 = ref_dense_pairwise_dist(Xc, Yf, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Compare to sklearn, F/C order
    S = pairwise_distances(Xf, Yc, metric=metric)
    S2 = ref_dense_pairwise_dist(Xf, Yc, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Compare to sklearn, F/F order
    S = pairwise_distances(Xf, Yf, metric=metric)
    S2 = ref_dense_pairwise_dist(Xf, Yf, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Switch which input has single dimension
    Xc = prep_dense_array(
        rng.random_sample((1, 4)), metric=metric, col_major=0
    )
    Yc = prep_dense_array(
        rng.random_sample((10, 4)), metric=metric, col_major=0
    )
    Xf = np.asfortranarray(Xc)
    Yf = np.asfortranarray(Yc)

    # Compare to sklearn, C/C order
    S = pairwise_distances(Xc, Yc, metric=metric)
    S2 = ref_dense_pairwise_dist(Xc, Yc, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Compare to sklearn, C/F order
    S = pairwise_distances(Xc, Yf, metric=metric)
    S2 = ref_dense_pairwise_dist(Xc, Yf, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Compare to sklearn, F/C order
    S = pairwise_distances(Xf, Yc, metric=metric)
    S2 = ref_dense_pairwise_dist(Xf, Yc, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Compare to sklearn, F/F order
    S = pairwise_distances(Xf, Yf, metric=metric)
    S2 = ref_dense_pairwise_dist(Xf, Yf, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)


@pytest.mark.parametrize("metric", ["haversine"])
def test_pairwise_distances_unsuppored_metrics(metric):
    rng = np.random.RandomState(3)

    X = rng.random_sample((5, 4))

    with pytest.raises(ValueError):
        pairwise_distances(X, metric=metric)


def test_pairwise_distances_exceptions():

    rng = np.random.RandomState(4)

    X_int = rng.randint(10, size=(5, 4))
    X_double = rng.random_sample((5, 4))
    X_float = np.asarray(X_double, dtype=np.float32)

    # Test second int inputs (should not have an exception with
    # convert_dtype=True)
    pairwise_distances(X_double, X_int, metric="euclidean")

    # Test sending different types with convert_dtype=False
    with pytest.raises(TypeError):
        pairwise_distances(
            X_double, X_float, metric="euclidean", convert_dtype=False
        )

    # Invalid metric name
    with pytest.raises(ValueError):
        pairwise_distances(X_double, metric="Not a metric")

    # Invalid dimensions
    X = rng.random_sample((5, 4))
    Y = rng.random_sample((5, 7))

    with pytest.raises(ValueError):
        pairwise_distances(X, Y, metric="euclidean")


@pytest.mark.parametrize("input_type", ["cudf", "numpy", "cupy"])
@pytest.mark.parametrize("output_type", ["cudf", "numpy", "cupy"])
@pytest.mark.parametrize("use_global", [True, False])
def test_pairwise_distances_output_types(input_type, output_type, use_global):
    # Test larger sizes to sklearn
    rng = np.random.RandomState(5)

    X = rng.random_sample((100, 100))
    Y = rng.random_sample((100, 100))

    if input_type == "cudf":
        X = cudf.DataFrame(X)
        Y = cudf.DataFrame(Y)
    elif input_type == "cupy":
        X = cp.asarray(X)
        Y = cp.asarray(Y)

    # Set to None if we are using the global object
    output_type_param = None if use_global else output_type

    # Use the global manager object. Should do nothing unless use_global is set
    with cuml.using_output_type(output_type):

        # Compare to sklearn, fp64
        S = pairwise_distances(
            X, Y, metric="euclidean", output_type=output_type_param
        )

        if output_type == "input":
            assert isinstance(S, type(X))
        elif output_type == "cudf":
            assert isinstance(S, cudf.DataFrame)
        elif output_type == "numpy":
            assert isinstance(S, np.ndarray)
        elif output_type == "cupy":
            assert isinstance(S, cp.ndarray)


def naive_inner(X, Y, metric=None):
    return X.dot(Y.T)


def naive_hellinger(X, Y, metric=None):
    return sklearn_pairwise_distances(
        np.sqrt(X), np.sqrt(Y), metric="euclidean"
    ) / np.sqrt(2)


def prepare_sparse_data(size0, size1, dtype, density, metric):
    # create sparse array, then normalize every row to one
    data = cupyx.scipy.sparse.random(
        size0, size1, dtype=dtype, random_state=123, density=density
    ).tocsr()
    if metric == "hellinger":
        data = csr_row_normalize_l1(data)
    return data


def ref_sparse_pairwise_dist(X, Y=None, metric=None):
    # Select sklearn except for IP and Hellinger that sklearn doesn't support
    # Use sparse input for sklearn calls when possible
    if Y is None:
        Y = X
    if metric not in [
        "cityblock",
        "cosine",
        "euclidean",
        "l1",
        "l2",
        "manhattan",
        "haversine",
    ]:
        X = X.todense()
        Y = Y.todense()
    X = X.get()
    Y = Y.get()
    if metric == "inner_product":
        return naive_inner(X, Y, metric)
    elif metric == "hellinger":
        return naive_hellinger(X, Y)
    else:
        return sklearn_pairwise_distances(X, Y, metric)


@pytest.mark.parametrize("metric", PAIRWISE_DISTANCE_SPARSE_METRICS.keys())
@pytest.mark.parametrize(
    "matrix_size, density", [((3, 3), 0.7), ((5, 40), 0.2)]
)
# ignoring boolean conversion warning for both cuml and sklearn
@pytest.mark.filterwarnings("ignore:(.*)converted(.*)::")
def test_sparse_pairwise_distances_corner_cases(
    metric: str, matrix_size, density: float
):
    # Test the sparse_pairwise_distance helper function.
    # For fp64, compare at 7 decimals, (5 places less than the ~15 max)
    compare_precision = 7

    # Compare to sklearn, single input
    X = prepare_sparse_data(
        matrix_size[0], matrix_size[1], cp.float64, density, metric
    )
    S = sparse_pairwise_distances(X, metric=metric)
    S2 = ref_sparse_pairwise_dist(X, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Compare to sklearn, double input with same dimensions
    Y = X
    S = pairwise_distances(X, Y, metric=metric)
    S2 = ref_sparse_pairwise_dist(X, Y, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Compare to sklearn, with Y dim != X dim
    Y = prepare_sparse_data(2, matrix_size[1], cp.float64, density, metric)
    S = pairwise_distances(X, Y, metric=metric)
    S2 = ref_sparse_pairwise_dist(X, Y, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Change precision of one parameter, should work (convert_dtype=True)
    Y = Y.astype(cp.float32)
    S = sparse_pairwise_distances(X, Y, metric=metric)
    S2 = ref_sparse_pairwise_dist(X, Y, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # For fp32, compare at 3 decimals, (4 places less than the ~7 max)
    compare_precision = 3

    # Change precision of both parameters to float
    X = prepare_sparse_data(
        matrix_size[0], matrix_size[1], cp.float32, density, metric
    )
    Y = prepare_sparse_data(
        matrix_size[0], matrix_size[1], cp.float32, density, metric
    )
    S = sparse_pairwise_distances(X, Y, metric=metric)
    S2 = ref_sparse_pairwise_dist(X, Y, metric=metric)
    cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # Test sending an int type (convert_dtype=True)
    if metric != "hellinger":
        compare_precision = 2
        Y = Y * 100
        Y.data = Y.data.astype(cp.int32)
        S = sparse_pairwise_distances(X, Y, metric=metric)
        S2 = ref_sparse_pairwise_dist(X, Y, metric=metric)
        cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)
    # Test that uppercase on the metric name throws an error.
    with pytest.raises(ValueError):
        sparse_pairwise_distances(X, Y, metric=metric.capitalize())


def test_sparse_pairwise_distances_exceptions():
    if not has_scipy():
        pytest.skip(
            "Skipping sparse_pairwise_distances_exceptions "
            "if Scipy is missing"
        )
    from scipy import sparse

    X_int = (
        sparse.random(5, 4, dtype=np.float32, random_state=123, density=0.3)
        * 10
    )
    X_int.dtype = cp.int32
    X_bool = sparse.random(5, 4, dtype=bool, random_state=123, density=0.3)
    X_double = cupyx.scipy.sparse.random(
        5, 4, dtype=cp.float64, random_state=123, density=0.3
    )
    X_float = cupyx.scipy.sparse.random(
        5, 4, dtype=cp.float32, random_state=123, density=0.3
    )

    # Test int inputs (only float/double accepted at this time)
    with pytest.raises(TypeError):
        sparse_pairwise_distances(X_int, metric="euclidean")

    # Test second int inputs (should not have an exception with
    # convert_dtype=True)
    sparse_pairwise_distances(X_double, X_int, metric="euclidean")

    # Test bool inputs (only float/double accepted at this time)
    with pytest.raises(TypeError):
        sparse_pairwise_distances(X_bool, metric="euclidean")

    # Test sending different types with convert_dtype=False
    with pytest.raises(TypeError):
        sparse_pairwise_distances(
            X_double, X_float, metric="euclidean", convert_dtype=False
        )

    # Invalid metric name
    with pytest.raises(ValueError):
        sparse_pairwise_distances(X_double, metric="Not a metric")

    # Invalid dimensions
    X = cupyx.scipy.sparse.random(5, 4, dtype=np.float32, random_state=123)
    Y = cupyx.scipy.sparse.random(5, 7, dtype=np.float32, random_state=123)

    with pytest.raises(ValueError):
        sparse_pairwise_distances(X, Y, metric="euclidean")


@pytest.mark.parametrize(
    "metric",
    [
        metric
        if metric != "hellinger"
        else pytest.param(
            metric,
            marks=pytest.mark.xfail(
                reason="intermittent failure (Issue #4354)"
            ),
        )
        for metric in PAIRWISE_DISTANCE_SPARSE_METRICS.keys()
    ],
)
@pytest.mark.parametrize(
    "matrix_size,density",
    [
        unit_param((1000, 100), 0.4),
        unit_param((20, 10000), 0.01),
        quality_param((2000, 1000), 0.05),
        stress_param((10000, 10000), 0.01),
    ],
)
# ignoring boolean conversion warning for both cuml and sklearn
@pytest.mark.filterwarnings("ignore:(.*)converted(.*)::")
def test_sparse_pairwise_distances_sklearn_comparison(
    metric: str, matrix_size, density: float
):
    # Test larger sizes to sklearn
    element_count = matrix_size[0] * matrix_size[1]

    X = prepare_sparse_data(
        matrix_size[0], matrix_size[1], cp.float64, density, metric
    )
    Y = prepare_sparse_data(
        matrix_size[0], matrix_size[1], cp.float64, density, metric
    )

    # For fp64, compare at 7 decimals, (8 places less than the ~15 max)
    compare_precision = 7

    # Compare to sklearn, fp64
    S = sparse_pairwise_distances(X, Y, metric=metric)

    if element_count <= 2000000:
        S2 = ref_sparse_pairwise_dist(X, Y, metric=metric)
        cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)

    # For fp32, compare at 3 decimals, (4 places less than the ~7 max)
    compare_precision = 3

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    # Compare to sklearn, fp32
    S = sparse_pairwise_distances(X, Y, metric=metric)

    if element_count <= 2000000:
        S2 = ref_sparse_pairwise_dist(X, Y, metric=metric)
        cp.testing.assert_array_almost_equal(S, S2, decimal=compare_precision)


@pytest.mark.parametrize("input_type", ["numpy", "cupy"])
@pytest.mark.parametrize("output_type", ["cudf", "numpy", "cupy"])
def test_sparse_pairwise_distances_output_types(input_type, output_type):
    # Test larger sizes to sklearn
    if not has_scipy():
        pytest.skip("Skipping sparse_pairwise_distances if Scipy is missing")
    import scipy

    if input_type == "cupy":
        X = cupyx.scipy.sparse.random(
            100, 100, dtype=cp.float64, random_state=123
        )
        Y = cupyx.scipy.sparse.random(
            100, 100, dtype=cp.float64, random_state=456
        )
    else:
        X = scipy.sparse.random(100, 100, dtype=np.float64, random_state=123)
        Y = scipy.sparse.random(100, 100, dtype=np.float64, random_state=456)

    # Use the global manager object.
    with cuml.using_output_type(output_type):
        S = sparse_pairwise_distances(X, Y, metric="euclidean")
        if output_type == "cudf":
            assert isinstance(S, cudf.DataFrame)
        elif output_type == "numpy":
            assert isinstance(S, np.ndarray)
        elif output_type == "cupy":
            assert isinstance(S, cp.ndarray)


@pytest.mark.xfail(
    reason="Temporarily disabling this test. " "See rapidsai/cuml#3569"
)
@pytest.mark.parametrize(
    "nrows, ncols, n_info",
    [
        unit_param(30, 10, 7),
        quality_param(5000, 100, 50),
        stress_param(500000, 200, 100),
    ],
)
@pytest.mark.parametrize("input_type", ["cudf", "cupy"])
@pytest.mark.parametrize("n_classes", [2, 5])
def test_hinge_loss(nrows, ncols, n_info, input_type, n_classes):
    train_rows = np.int32(nrows * 0.8)
    X, y = make_classification(
        n_samples=nrows,
        n_features=ncols,
        n_clusters_per_class=1,
        n_informative=n_info,
        random_state=123,
        n_classes=n_classes,
    )

    if input_type == "cudf":
        X = cudf.DataFrame(X)
        y = cudf.Series(y)
    elif input_type == "cupy":
        X = cp.asarray(X)
        y = cp.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_rows, shuffle=True
    )
    cuml_model = cu_log()
    cuml_model.fit(X_train, y_train)
    cu_predict_decision = cuml_model.decision_function(X_test)
    cu_loss = cuml_hinge(y_test, cu_predict_decision.T, labels=cp.unique(y))
    if input_type == "cudf":
        y_test = y_test.to_numpy()
        y = y.to_numpy()
        cu_predict_decision = cp.asnumpy(cu_predict_decision.values)
    elif input_type == "cupy":
        y = cp.asnumpy(y)
        y_test = cp.asnumpy(y_test)
        cu_predict_decision = cp.asnumpy(cu_predict_decision)

    cu_loss_using_sk = sk_hinge(
        y_test, cu_predict_decision.T, labels=np.unique(y)
    )
    # compare the accuracy of the two models
    cp.testing.assert_array_almost_equal(cu_loss, cu_loss_using_sk)


@pytest.mark.parametrize(
    "nfeatures",
    [
        unit_param(10),
        unit_param(300),
        unit_param(30000),
        stress_param(500000000),
    ],
)
@pytest.mark.parametrize("input_type", ["cudf", "cupy"])
@pytest.mark.parametrize("dtypeP", [cp.float32, cp.float64])
@pytest.mark.parametrize("dtypeQ", [cp.float32, cp.float64])
def test_kl_divergence(nfeatures, input_type, dtypeP, dtypeQ):
    if not has_scipy():
        pytest.skip("Skipping test_kl_divergence because Scipy is missing")

    from scipy.stats import entropy as sp_entropy

    rng = np.random.RandomState(5)

    P = rng.random_sample((nfeatures))
    Q = rng.random_sample((nfeatures))

    P /= P.sum()
    Q /= Q.sum()
    sk_res = sp_entropy(P, Q)

    if input_type == "cudf":
        P = cudf.DataFrame(P, dtype=dtypeP)
        Q = cudf.DataFrame(Q, dtype=dtypeQ)
    elif input_type == "cupy":
        P = cp.asarray(P, dtype=dtypeP)
        Q = cp.asarray(Q, dtype=dtypeQ)

    if dtypeP != dtypeQ:
        with pytest.raises(TypeError):
            cu_kl_divergence(P, Q, convert_dtype=False)
        cu_res = cu_kl_divergence(P, Q)
    else:
        cu_res = cu_kl_divergence(P, Q, convert_dtype=False)

    cp.testing.assert_array_almost_equal(cu_res, sk_res)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0, 2.0])
def test_v_measure_score(beta):
    labels_true = np.array([0, 0, 1, 1], dtype=np.int32)
    labels_pred = np.array([1, 0, 1, 1], dtype=np.int32)
    res = v_measure_score(labels_true, labels_pred, beta=beta)
    ref = sklearn_v_measure_score(labels_true, labels_pred, beta=beta)
    assert_almost_equal(res, ref)
