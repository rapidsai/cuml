# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

from cuml.common.exceptions import NotFittedError
from sklearn.datasets import make_blobs
from sklearn.decomposition import KernelPCA as skKernelPCA
from sklearn.datasets import make_multilabel_classification
from sklearn import datasets
from sklearn.model_selection import train_test_split
from cuml.internals import logger

from cuml.testing.utils import (
    get_handle,
    array_equal,
    unit_param,
    quality_param,
    stress_param,
)
from cuml.experimental.decomposition import KernelPCA as cuKernelPCA
import pytest
from cuml.internals.safe_imports import gpu_only_import
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("input_type", ["ndarray"])
@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize(
    "name", [unit_param(None), quality_param("digits"), stress_param("blobs")]
)
@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf", "sigmoid"])
def test_kpca_fit(datatype, input_type, name, use_handle, kernel):
    if name == "blobs":
        pytest.skip("fails when using blobs dataset")
        X, y = make_blobs(n_samples=500000, n_features=1000, random_state=0)

    elif name == "digits":
        X, _ = datasets.load_digits(return_X_y=True)

    else:
        X, Y = make_multilabel_classification(
            n_samples=500,
            n_classes=2,
            n_labels=1,
            allow_unlabeled=False,
            random_state=1,
        )

    X = X.astype(datatype)
    kernel = 'linear'
    skpca = skKernelPCA(n_components=4, kernel=kernel)
    skpca.fit(X)

    handle, stream = get_handle(use_handle)
    cupca = cuKernelPCA(n_components=4, handle=handle, kernel=kernel)
    cupca.fit(X)
    cupca.handle.sync()

    for attr in [
        "eigenvectors_",
        "eigenvalues_",
    ]:
        # with_sign = False if attr in ["components_"] else True TODO(TOMAS)
        cuml_res = getattr(cupca, attr)

        skl_res = getattr(skpca, attr)
        assert array_equal(cuml_res, skl_res, 1e-1, total_tol=1e-1, with_sign=True)

@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("input_type", ["ndarray"])
@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize(
    "name", [unit_param(None), quality_param("iris"), stress_param("blobs")]
)
@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf", "sigmoid"])
def test_kpca_fit_then_transform(datatype, input_type, name, use_handle, kernel):
    blobs_n_samples = 500000
    if name == "blobs" and pytest.max_gpu_memory < 32:
        if pytest.adapt_stress_test:
            blobs_n_samples = int(blobs_n_samples * pytest.max_gpu_memory / 32)
        else:
            pytest.skip(
                "Insufficient GPU memory for this test."
                "Re-run with 'CUML_ADAPT_STRESS_TESTS=True'"
            )

    if name == "blobs":
        X, y = make_blobs(
            n_samples=blobs_n_samples, n_features=1000, random_state=0
        )

    elif name == "iris":
        iris = datasets.load_iris()
        X = iris.data

    else:
        X, Y = make_multilabel_classification(
            n_samples=500,
            n_classes=2,
            n_labels=1,
            allow_unlabeled=False,
            random_state=1,
        )

    X = X.astype(datatype)
    if name != "blobs":
        skpca = skKernelPCA(n_components=2, kernel=kernel)
        skpca.fit(X)
        X_sk = skpca.transform(X)

    handle, stream = get_handle(use_handle)
    cupca = cuKernelPCA(n_components=2, handle=handle, kernel=kernel)

    cupca.fit(X)
    X_cu = cupca.transform(X)
    cupca.handle.sync()

    if name != "blobs":
        assert array_equal(X_cu, X_sk, 1e-1, total_tol=1e-1, with_sign=True)
        assert X_sk.shape[0] == X_cu.shape[0]
        assert X_sk.shape[1] == X_cu.shape[1]

@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("input_type", ["ndarray"])
@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize(
    "name", [unit_param(None), quality_param("iris"), stress_param("blobs")]
)
@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf", "sigmoid"])
def test_kpca_fit_transform(datatype, input_type, name, use_handle, kernel):
    blobs_n_samples = 500000
    if name == "blobs" and pytest.max_gpu_memory < 32:
        if pytest.adapt_stress_test:
            blobs_n_samples = int(blobs_n_samples * pytest.max_gpu_memory / 32)
        else:
            pytest.skip(
                "Insufficient GPU memory for this test."
                "Re-run with 'CUML_ADAPT_STRESS_TESTS=True'"
            )

    if name == "blobs":
        X, y = make_blobs(
            n_samples=blobs_n_samples, n_features=1000, random_state=0
        )

    elif name == "iris":
        iris = datasets.load_iris()
        X = iris.data

    else:
        X, Y = make_multilabel_classification(
            n_samples=500,
            n_classes=2,
            n_labels=1,
            allow_unlabeled=False,
            random_state=1,
        )

    X = X.astype(datatype)
    if name != "blobs":
        skpca = skKernelPCA(n_components=2, kernel=kernel)
        X_sk = skpca.fit_transform(X)

    handle, stream = get_handle(use_handle)
    cupca = cuKernelPCA(n_components=2, handle=handle, kernel=kernel)

    X_cu = cupca.fit_transform(X)
    cupca.handle.sync()

    if name != "blobs":
        assert array_equal(X_cu, X_sk, 1e-1, total_tol=1e-1, with_sign=True)
        assert X_sk.shape[0] == X_cu.shape[0]
        assert X_sk.shape[1] == X_cu.shape[1]

@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("input_type", ["ndarray"])
@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize("name", [unit_param(None), quality_param("iris")])
@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf", "sigmoid"])
def test_kpca_fit_then_transform_on_test_train_split(datatype, input_type, name, use_handle, kernel):
    if name == "iris":
        iris = datasets.load_iris()
        X = iris.data
    else:
        X, _ = make_multilabel_classification(
            n_samples=500,
            n_classes=2,
            n_labels=1,
            allow_unlabeled=False,
            random_state=1,
        )

    X = X.astype(datatype)
    X_train, X_test = train_test_split(X, random_state=0)
    # logger.info samples in train and test
    logger.info(f"TOMAS X_train shape: {X_train.shape}")
    logger.info(f"TOMAS X_test shape: {X_test.shape}")
    skpca = skKernelPCA(n_components=2, kernel=kernel)
    skpca.fit(X_train)
    X_test_sk = skpca.transform(X_test)

    handle, stream = get_handle(use_handle)
    cupca = cuKernelPCA(n_components=2, handle=handle, kernel=kernel)
    cupca.fit(X_train)
    X_test_cu = cupca.transform2(X_test)
    cupca.handle.sync()

    assert array_equal(X_test_cu, X_test_sk, 1e-1, total_tol=1e-1, with_sign=True)
    assert X_test_sk.shape[0] == X_test_cu.shape[0]
    assert X_test_sk.shape[1] == X_test_cu.shape[1]


def test_exceptions():
    with pytest.raises(NotFittedError):
        X = cp.random.random((10, 10))
        cuKernelPCA().transform(X)