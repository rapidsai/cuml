# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import cupy as cp
import numpy as np
import pytest
import scipy.sparse
import sklearn.kernel_ridge
import sklearn.svm
import umap
from numpy.testing import assert_allclose
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.decomposition import PCA as SkPCA
from sklearn.decomposition import TruncatedSVD as SkTruncatedSVD
from sklearn.linear_model import ElasticNet as SkElasticNet
from sklearn.linear_model import Lasso as SkLasso
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.linear_model import Ridge as SkRidge
from sklearn.manifold import trustworthiness
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

import cuml
from cuml.cluster import DBSCAN, KMeans
from cuml.decomposition import PCA, TruncatedSVD
from cuml.internals.interop import UnsupportedOnCPU, UnsupportedOnGPU
from cuml.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from cuml.manifold import TSNE, SpectralEmbedding
from cuml.testing.utils import array_equal

###############################################################################
#                              Helper functions                               #
###############################################################################


@pytest.fixture
def random_state():
    return 42


def assert_roundtrip_consistency(original, roundtrip, exclude=()):
    original_params = original.get_params()
    roundtrip_params = roundtrip.get_params()

    # Remove parameters that are not guaranteed to be equivalent
    for name in [*exclude, "handle", "output_type", "verbose"]:
        original_params.pop(name, None)
        roundtrip_params.pop(name, None)

    def dict_diff(a, b):
        # Get all keys from both dictionaries
        all_keys = set(a.keys()) | set(b.keys())
        differences = {}
        for key in all_keys:
            if a.get(key) != b.get(key):
                differences[key] = {"a_dict": a.get(key), "b_dict": b.get(key)}
        return differences

    assert (
        original_params == roundtrip_params
    ), f"Differences found: {dict_diff(original_params, roundtrip_params)}"

    # Next check attributes are consistent. We don't check for equality
    # since that might change. We do check for consistency in attribute
    # availability, as well as dtype/shape/order of array values
    attrs = [
        name
        for name in dir(original)
        if name.endswith("_")
        and not name.startswith("_")
        and not name == "feature_names_in_"
        and name not in exclude
        and hasattr(original, name)
    ]
    with cuml.using_output_type("cupy"):
        for name in attrs:
            orig_attr = getattr(original, name)
            roundtrip_attr = getattr(roundtrip, name)
            if isinstance(orig_attr, cp.ndarray):
                assert orig_attr.shape == roundtrip_attr.shape, name
                assert orig_attr.dtype == roundtrip_attr.dtype, name
                assert (
                    orig_attr.flags["C_CONTIGUOUS"]
                    == roundtrip_attr.flags["C_CONTIGUOUS"]
                ), name
                assert (
                    orig_attr.flags["F_CONTIGUOUS"]
                    == roundtrip_attr.flags["F_CONTIGUOUS"]
                ), name


def assert_estimator_roundtrip(
    cuml_model, sklearn_class, X, y=None, transform=False, exclude_params=()
):
    """
    Generic assertion helper to test round-trip conversion:
    fit original custom model
    convert to sklearn
    convert back to custom model
    compare predictions or transform outputs
    """
    # Fit original model
    if y is not None:
        cuml_model.fit(X, y)
    else:
        cuml_model.fit(X)

    # Convert to sklearn model
    sklearn_model = cuml_model.as_sklearn()
    assert isinstance(sklearn_model, sklearn_class)
    check_is_fitted(sklearn_model)

    # Convert back
    roundtrip_model = type(cuml_model).from_sklearn(sklearn_model)

    # Ensure params/attrs roundtrip
    assert_roundtrip_consistency(
        cuml_model, roundtrip_model, exclude=exclude_params
    )

    # Ensure roundtrip model is fitted
    check_is_fitted(roundtrip_model)

    # Compare predictions or transforms
    if transform:
        original_output = cuml_model.transform(X)
        roundtrip_output = roundtrip_model.transform(X)
        assert array_equal(original_output, roundtrip_output)
    else:
        # For predict methods
        if hasattr(cuml_model, "predict"):
            original_pred = cuml_model.predict(X)
            roundtrip_pred = roundtrip_model.predict(X)
            assert array_equal(original_pred, roundtrip_pred)
        # For models that only produce labels_ or similar attributes (e.g., clustering)
        elif hasattr(cuml_model, "labels_"):
            assert array_equal(cuml_model.labels_, roundtrip_model.labels_)
        else:
            # If we get here, need a custom handling for that type
            raise NotImplementedError(
                "No known method to compare outputs of this model."
            )

    # Check that the scikit-learn estimator can be fitted again which checks
    # that the hyper-parameter translation from cuml to scikit-learn works
    # Has to happen after comparing predictions/transforms as refitting might
    # change cluster IDs and the like
    if y is not None:
        sklearn_model.fit(X, y)
    else:
        sklearn_model.fit(X)


###############################################################################
#                                     Tests                                   #
###############################################################################


def test_basic_roundtrip():
    km = SkKMeans(n_clusters=13)
    ckm = KMeans.from_sklearn(km)

    assert ckm.n_clusters == 13


def test_kmeans(random_state):
    # Using sklearn directly for demonstration
    X, _ = make_blobs(
        n_samples=50, n_features=2, centers=3, random_state=random_state
    )
    original = KMeans(n_clusters=3, random_state=random_state)
    assert_estimator_roundtrip(original, SkKMeans, X)


def test_dbscan(random_state):
    X, _ = make_blobs(
        n_samples=50, n_features=2, centers=3, random_state=random_state
    )
    original = DBSCAN(eps=0.5, min_samples=5)
    # DBSCAN assigns labels_ after fit
    original.fit(X)
    sklearn_model = original.as_sklearn()
    roundtrip_model = DBSCAN.from_sklearn(sklearn_model)
    assert array_equal(original.labels_, roundtrip_model.labels_)


def test_pca(random_state):
    X = np.random.RandomState(random_state).rand(50, 5)
    original = PCA(n_components=2)
    assert_estimator_roundtrip(original, SkPCA, X, transform=True)


def test_truncated_svd(random_state):
    X = np.random.RandomState(random_state).rand(50, 5)
    original = TruncatedSVD(n_components=2)
    assert_estimator_roundtrip(original, SkTruncatedSVD, X, transform=True)


def test_linear_regression(random_state):
    X, y = make_regression(
        n_samples=50, n_features=5, noise=0.1, random_state=random_state
    )
    original = LinearRegression()
    assert_estimator_roundtrip(original, SkLinearRegression, X, y)


def test_logistic_regression(random_state):
    X, y = make_classification(
        n_samples=50, n_features=5, n_informative=3, random_state=random_state
    )
    original = LogisticRegression(C=5.0, max_iter=500)
    assert_estimator_roundtrip(original, SkLogisticRegression, X, y)


def test_elasticnet(random_state):
    X, y = make_regression(
        n_samples=50, n_features=5, noise=0.1, random_state=random_state
    )
    original = ElasticNet(alpha=0.1)
    assert_estimator_roundtrip(original, SkElasticNet, X, y)


def test_ridge(random_state):
    X, y = make_regression(
        n_samples=50, n_features=5, noise=0.1, random_state=random_state
    )
    original = Ridge(alpha=1.0)
    assert_estimator_roundtrip(original, SkRidge, X, y)


def test_lasso(random_state):
    X, y = make_regression(
        n_samples=50, n_features=5, noise=0.1, random_state=random_state
    )
    original = Lasso(alpha=0.1)
    assert_estimator_roundtrip(original, SkLasso, X, y)


@pytest.mark.filterwarnings(
    "ignore:Starting from version 22.04, the default method of TSNE is 'fft'."
)
def test_tsne(random_state):
    # TSNE is a bit tricky as it is non-deterministic. For test simplicity:
    X = np.random.RandomState(random_state).rand(50, 5)
    original = TSNE(n_components=2, random_state=random_state)
    original.fit(X)
    sklearn_model = original.as_sklearn()
    roundtrip_model = TSNE.from_sklearn(sklearn_model)
    # Since TSNE is non-deterministic, exact match is unlikely.
    # We can at least check output dimensions are the same.
    original_embedding = original.embedding_
    sklearn_embedding = sklearn_model.embedding_
    roundtrip_embedding = roundtrip_model.embedding_

    assert array_equal(original_embedding, sklearn_embedding)
    assert array_equal(original_embedding, roundtrip_embedding)


def test_spectral_embedding(random_state):
    X, _ = make_blobs(
        n_samples=100, centers=3, n_features=20, random_state=random_state
    )
    original = SpectralEmbedding(n_components=2, random_state=random_state)
    original.fit(X)
    sklearn_model = original.as_sklearn()
    roundtrip_model = SpectralEmbedding.from_sklearn(sklearn_model)
    assert_roundtrip_consistency(original, roundtrip_model)

    original_embedding = original.embedding_
    sklearn_embedding = sklearn_model.embedding_
    roundtrip_embedding = roundtrip_model.embedding_

    assert array_equal(original_embedding, sklearn_embedding)
    assert array_equal(original_embedding, roundtrip_embedding)


@pytest.mark.parametrize("alpha_kind", ["scalar", "numpy", "cupy"])
def test_kernel_ridge(random_state, alpha_kind):
    X, y = make_regression(
        n_samples=50,
        n_features=10,
        n_targets=2,
        noise=0.1,
        random_state=random_state,
    )
    if alpha_kind == "scalar":
        cu_alpha = sk_alpha = 1
    elif alpha_kind == "numpy":
        cu_alpha = sk_alpha = np.array([1.0, 2.0])
    elif alpha_kind == "cupy":
        cu_alpha = cp.array([1.0, 2.0])
        sk_alpha = cu_alpha.get()

    cu_model = cuml.KernelRidge(alpha=cu_alpha).fit(X, y)
    sk_model = sklearn.kernel_ridge.KernelRidge(alpha=sk_alpha).fit(X, y)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.KernelRidge.from_sklearn(sk_model)

    # Can infer on converted models
    assert sk_model2.score(X, y) > 0.7
    assert cu_model2.score(X, y) > 0.7

    # Can refit on converted models
    cu_model2.fit(X, y)
    sk_model2.fit(X, y)

    # Refit models have similar results
    assert sk_model2.score(X, y) > 0.7
    assert cu_model2.score(X, y) > 0.7


@pytest.mark.parametrize("sparse", [False, True])
def test_svr(random_state, sparse):
    X, y = make_regression(n_samples=100, random_state=random_state)
    if sparse:
        X = scipy.sparse.coo_matrix(X)
    original = cuml.SVR()
    assert_estimator_roundtrip(original, sklearn.svm.SVR, X, y)

    # Check inference works after conversion
    cu_model = cuml.SVR(kernel="linear").fit(X, y)
    sk_model = sklearn.svm.SVR(kernel="linear").fit(X, y)

    sk_score = cu_model.as_sklearn().score(X, y)
    assert sk_score > 0.7

    cu_model_from_sklearn = cuml.SVR.from_sklearn(sk_model)

    cu_score = cu_model_from_sklearn.score(X, y)
    assert cu_score > 0.7

    # Check n_support is set correctly
    assert (
        cu_model_from_sklearn.n_support_
        == cu_model_from_sklearn.support_vectors_.shape[0]
    )


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("probability", [False, True])
def test_svc(random_state, sparse, probability):
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, random_state=random_state
    )
    if sparse:
        X = scipy.sparse.coo_matrix(X)
    original = cuml.SVC()
    assert_estimator_roundtrip(original, sklearn.svm.SVC, X, y)

    # Check inference works after conversion
    cu_model = cuml.SVC(probability=probability).fit(X, y)
    sk_model = sklearn.svm.SVC(probability=probability).fit(X, y)

    cu_model2 = cuml.SVC.from_sklearn(sk_model)
    sk_model2 = cu_model.as_sklearn()

    cu_score = cu_model2.score(X, y)
    assert cu_score > 0.7

    sk_score = sk_model2.score(X, y)
    assert sk_score > 0.7

    if probability:
        # Check that predict_proba works
        cu_pred_prob = cu_model2.predict_proba(X).argmax(axis=1)
        assert accuracy_score(cu_pred_prob, y) > 0.7
        sk_pred_prob = sk_model2.predict_proba(X).argmax(axis=1)
        assert accuracy_score(sk_pred_prob, y) > 0.7

        # Check that probA_, probB_ are wired up properly
        for attr in ["probA_", "probB_"]:
            val = getattr(sk_model2, attr)
            assert isinstance(val, np.ndarray)
            assert val.dtype == "float64"
            assert val.shape == (1,)

    # Check n_support_ is correctly set
    assert cu_model2.n_support_ == cu_model2.support_vectors_.shape[0]


@pytest.mark.parametrize("kind", ["SVC", "SVR"])
def test_svm_big_sparse(random_state, kind):
    """SVC/SVR models are only sparse if fit with a sparse matrix AND storing
    `support_vectors_` as dense exceeds a limit (currently 1 GiB). Here we
    create a large enough input matrix to ensure a sparse repr, and check that
    the conversion is plumbed through properly"""
    n_samples = 1024
    noise_cols = 300000
    noise_density = 0.0001
    n_noise = int(noise_cols * n_samples * noise_density)

    if kind == "SVR":
        X, y = make_regression(
            n_samples=n_samples, n_features=4, random_state=random_state
        )
        model = cuml.SVR(kernel="linear")
    else:
        X, y = make_classification(
            n_samples=n_samples, n_features=4, random_state=random_state
        )
        model = cuml.SVC()

    rng = np.random.default_rng(random_state)
    rows = rng.integers(0, n_samples, size=n_noise)
    cols = rng.integers(0, noise_cols, size=n_noise)
    data = rng.uniform(size=n_noise)

    X = scipy.sparse.hstack(
        [
            scipy.sparse.csr_matrix(X),
            scipy.sparse.coo_matrix(
                (data, (rows, cols)), shape=(n_samples, noise_cols)
            ).asformat("csr"),
        ]
    )

    model.fit(X, y)

    sk_model = model.as_sklearn()
    assert scipy.sparse.issparse(sk_model.support_vectors_)

    score = sk_model.score(X[:100], y[:100])
    assert score > 0.7


def test_svc_multiclass_unsupported(random_state):
    X, y = make_classification(
        n_samples=50,
        n_features=10,
        n_classes=3,
        n_informative=5,
        random_state=random_state,
    )
    cu_model = cuml.SVC().fit(X, y)
    sk_model = sklearn.svm.SVC().fit(X, y)

    with pytest.raises(UnsupportedOnGPU):
        cuml.SVC.from_sklearn(sk_model)

    with pytest.raises(UnsupportedOnCPU):
        cu_model.as_sklearn()


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("supervised", [False, True])
def test_umap(random_state, sparse, supervised):
    n_neighbors = 10
    X, y = make_blobs(n_samples=200, random_state=random_state)
    X = X.astype("float32")
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    if sparse:
        X_train = scipy.sparse.csr_matrix(X_train)
        X_test = scipy.sparse.csr_matrix(X_test)
    if not supervised:
        y_train = None

    cu_model = cuml.UMAP(n_neighbors=n_neighbors, hash_input=True).fit(
        X_train, y_train
    )
    sk_model = umap.UMAP(n_neighbors=n_neighbors).fit(X_train, y_train)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.UMAP.from_sklearn(sk_model)

    # Ensure params/attrs roundtrip
    assert_roundtrip_consistency(cu_model, cu_model2, exclude=["build_algo"])

    # Can infer on converted models
    np.testing.assert_array_equal(
        sk_model2.transform(X_train), sk_model2.embedding_
    )
    with pytest.raises(NotImplementedError):
        # Can't currently infer on new data in umap.UMAP.transform implementation
        assert isinstance(sk_model2.transform(X_test), np.ndarray)

    np.testing.assert_array_equal(
        cu_model2.transform(X_train), cu_model2.embedding_
    )
    assert isinstance(cu_model2.transform(X_test), np.ndarray)

    # Can refit on converted models
    cu_model2.fit(X_train, y_train)
    sk_model2.fit(X_train, y_train)

    # Refit embeddings have similar scores
    cu_trust1 = trustworthiness(
        X_train, cu_model.embedding_, n_neighbors=n_neighbors
    )
    cu_trust2 = trustworthiness(
        X_train, cu_model2.embedding_, n_neighbors=n_neighbors
    )
    np.testing.assert_allclose(cu_trust1, cu_trust2, atol=0.05)

    sk_trust1 = trustworthiness(
        X_train, sk_model.embedding_, n_neighbors=n_neighbors
    )
    sk_trust2 = trustworthiness(
        X_train, sk_model2.embedding_, n_neighbors=n_neighbors
    )
    np.testing.assert_allclose(sk_trust1, sk_trust2, atol=0.05)


@pytest.mark.parametrize("sparse", [False, True])
def test_nearest_neighbors(random_state, sparse):
    if sparse:
        X = scipy.sparse.rand(
            50,
            20,
            density=0.25,
            rng=random_state,
            dtype="float32",
            format="csr",
        )
    else:
        X = np.random.default_rng(random_state).random(
            (50, 20), dtype="float32"
        )

    cu_model = cuml.NearestNeighbors(n_neighbors=10).fit(X)
    sk_model = sklearn.neighbors.NearestNeighbors(n_neighbors=10).fit(X)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.NearestNeighbors.from_sklearn(sk_model)

    # Ensure params/attrs roundtrip
    assert_roundtrip_consistency(cu_model, cu_model2)

    def assert_kneighbors_close(m1, m2):
        inds1, dists1 = m1.kneighbors(X)
        inds2, dists2 = m2.kneighbors(X)
        np.testing.assert_allclose(inds1, inds2, atol=1e-7)
        np.testing.assert_allclose(dists1, dists2, atol=1e-4)

    # Can infer on converted models
    assert_kneighbors_close(sk_model, sk_model2)
    assert_kneighbors_close(cu_model, cu_model2)

    # Can refit on converted models
    cu_model2.fit(X)
    sk_model2.fit(X)

    # Refit models have similar results
    assert_kneighbors_close(sk_model, sk_model2)
    assert_kneighbors_close(cu_model, cu_model2)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("n_targets", [1, 3])
def test_kneighbors_regressor(random_state, sparse, n_targets):
    X, y = make_regression(
        100, 50, n_targets=n_targets, random_state=random_state
    )
    X = X.astype("float32")
    if sparse:
        X[X < -0.5] = 0
        X = scipy.sparse.csr_matrix(X)

    cu_model = cuml.KNeighborsRegressor(n_neighbors=10).fit(X, y)
    sk_model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=10).fit(X, y)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.KNeighborsRegressor.from_sklearn(sk_model)

    # Ensure params/attrs roundtrip
    assert_roundtrip_consistency(cu_model, cu_model2)

    # Can infer on converted models
    assert_allclose(sk_model.predict(X), sk_model2.predict(X), atol=1e-3)
    assert_allclose(cu_model.predict(X), cu_model2.predict(X), atol=1e-3)

    # Can refit on converted models
    cu_model2.fit(X, y)
    sk_model2.fit(X, y)

    # Refit models have similar results
    assert_allclose(sk_model.predict(X), sk_model2.predict(X), atol=1e-3)
    assert_allclose(cu_model.predict(X), cu_model2.predict(X), atol=1e-3)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("n_labels", [1, 3])
def test_kneighbors_classifier(random_state, sparse, n_labels):
    if n_labels > 1:
        X, y = make_multilabel_classification(
            100,
            50,
            n_labels=n_labels,
            random_state=random_state,
            sparse=sparse,
        )
    else:
        X, y = make_classification(100, 50, random_state=random_state)
        if sparse:
            X[X < -0.5] = 0
            X = scipy.sparse.csr_matrix(X)

    X = X.astype("float32")

    cu_model = cuml.KNeighborsClassifier(n_neighbors=10).fit(X, y)
    sk_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=10).fit(X, y)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.KNeighborsClassifier.from_sklearn(sk_model)

    # `classes_` attribute transfers properly
    if n_labels > 1:
        assert all(isinstance(c, np.ndarray) for c in sk_model2.classes_)
        assert all(isinstance(c, np.ndarray) for c in cu_model2.classes_)
    else:
        assert isinstance(sk_model2.classes_, np.ndarray)
        assert isinstance(cu_model2.classes_, np.ndarray)

    # Ensure params/attrs roundtrip
    assert_roundtrip_consistency(cu_model, cu_model2)

    # Can infer on converted models
    np.testing.assert_array_equal(sk_model.predict(X), sk_model2.predict(X))
    np.testing.assert_array_equal(cu_model.predict(X), cu_model2.predict(X))

    # Can refit on converted models
    cu_model2.fit(X, y)
    sk_model2.fit(X, y)

    # Refit models have similar results
    np.testing.assert_array_equal(sk_model.predict(X), sk_model2.predict(X))
    np.testing.assert_array_equal(cu_model.predict(X), cu_model2.predict(X))


def test_random_forest_classifier(random_state):
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3, random_state=random_state
    )

    cu_model = cuml.RandomForestClassifier().fit(X, y)
    sk_model = sklearn.ensemble.RandomForestClassifier().fit(X, y)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.RandomForestClassifier.from_sklearn(sk_model)

    # `classes_` attribute transfers properly
    assert isinstance(sk_model2.classes_, np.ndarray)
    assert isinstance(cu_model2.classes_, np.ndarray)
    assert (sk_model2.classes_ == cu_model2.classes_).all()

    # Can infer on converted models
    assert sk_model2.score(X, y) > 0.7
    assert cu_model2.score(X, y) > 0.7

    # Can refit on converted models
    cu_model2.fit(X, y)
    sk_model2.fit(X, y)

    # Refit models have similar results
    assert sk_model2.score(X, y) > 0.7
    assert cu_model2.score(X, y) > 0.7


def test_random_forest_regressor(random_state):
    X, y = make_regression(n_samples=200, random_state=random_state)
    X = X.astype("float32")

    cu_model = cuml.RandomForestRegressor().fit(X, y)
    sk_model = sklearn.ensemble.RandomForestRegressor().fit(X, y)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.RandomForestRegressor.from_sklearn(sk_model)

    # Ensure params/attrs roundtrip
    assert_roundtrip_consistency(cu_model, cu_model2)

    # Can infer on converted models
    assert sk_model2.score(X, y) > 0.7
    assert cu_model2.score(X, y) > 0.7

    # Can refit on converted models
    cu_model2.fit(X, y)
    sk_model2.fit(X, y)

    # Refit models have similar results
    assert sk_model2.score(X, y) > 0.7
    assert cu_model2.score(X, y) > 0.7


@pytest.mark.parametrize("prediction_data", [False, True])
@pytest.mark.parametrize("gen_min_span_tree", [False, True])
def test_hdbscan(random_state, prediction_data, gen_min_span_tree):
    hdbscan = pytest.importorskip("hdbscan")

    X, y = make_blobs(random_state=random_state)

    cu_model = cuml.HDBSCAN(
        prediction_data=prediction_data, gen_min_span_tree=gen_min_span_tree
    ).fit(X, y)
    sk_model = hdbscan.HDBSCAN(
        prediction_data=prediction_data, gen_min_span_tree=gen_min_span_tree
    ).fit(X, y)

    sk_model2 = cu_model.as_sklearn()
    cu_model2 = cuml.HDBSCAN.from_sklearn(sk_model)

    # Ensure params/attrs roundtrip
    assert_roundtrip_consistency(cu_model, cu_model2)

    # tree attributes all available
    for attr in ["single_linkage_tree_", "condensed_tree_"]:
        assert getattr(cu_model2, attr) is not None
        assert getattr(sk_model2, attr) is not None

    # minimum_spanning_tree_ available if generated
    if gen_min_span_tree:
        assert getattr(cu_model2, "minimum_spanning_tree_") is not None
        assert getattr(sk_model2, "minimum_spanning_tree_") is not None
    else:
        assert not hasattr(cu_model2, "minimum_spanning_tree_")
        assert not hasattr(sk_model2, "minimum_spanning_tree_")

    # prediction data available if generated
    if prediction_data:
        assert getattr(cu_model2, "prediction_data_") is not None
        assert getattr(sk_model2, "prediction_data_") is not None
    else:
        assert not hasattr(cu_model2, "prediction_data_")
        assert not hasattr(sk_model2, "prediction_data_")

        # Generate prediction data
        cu_model2.generate_prediction_data()
        sk_model2.generate_prediction_data()

    # Can infer on converted models
    cu_labels, cu_probs = cuml.cluster.hdbscan.approximate_predict(
        cu_model2, X
    )
    sk_labels, sk_probs = hdbscan.approximate_predict(sk_model2, X)
    assert accuracy_score(cu_labels, cu_model2.labels_) > 0.9
    assert accuracy_score(sk_labels, sk_model2.labels_) > 0.9

    # Can refit on converted models
    cu_model2.fit(X, y)
    sk_model2.fit(X, y)


def test_linear_svr(random_state):
    X, y = make_regression(n_samples=100, random_state=random_state)
    # For `dual=False`, scikit-learn only supports the squared epsilon-insensitive loss
    original = cuml.LinearSVR(loss="squared_epsilon_insensitive")
    assert_estimator_roundtrip(original, sklearn.svm.LinearSVR, X, y)

    # Check inference works after conversion
    cu_model = cuml.LinearSVR(loss="squared_epsilon_insensitive").fit(X, y)
    sk_model = sklearn.svm.LinearSVR(
        loss="squared_epsilon_insensitive", dual=False
    ).fit(X, y)

    sk_score = cu_model.as_sklearn().score(X, y)
    assert sk_score > 0.7

    cu_score = cuml.LinearSVR.from_sklearn(sk_model).score(X, y)
    assert cu_score > 0.7


def test_linear_svc(random_state):
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, random_state=random_state
    )
    original = cuml.LinearSVC()
    assert_estimator_roundtrip(original, sklearn.svm.LinearSVC, X, y)

    # Check inference works after conversion
    cu_model = cuml.LinearSVC().fit(X, y)
    sk_model = sklearn.svm.LinearSVC().fit(X, y)

    cu_model2 = cuml.LinearSVC.from_sklearn(sk_model)
    sk_model2 = cu_model.as_sklearn()

    cu_score = cu_model2.score(X, y)
    assert cu_score > 0.7

    sk_score = sk_model2.score(X, y)
    assert sk_score > 0.7
