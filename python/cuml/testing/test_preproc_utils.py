# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from cuml.common import input_to_cuml_array
from scipy.sparse import coo_matrix as cpu_coo_matrix
from scipy.sparse import csc_matrix as cpu_csc_matrix
from cupyx.scipy.sparse import coo_matrix as gpu_coo_matrix
from cuml.internals.safe_imports import gpu_only_import_from
from cuml.internals.safe_imports import gpu_only_import
from cuml.internals.safe_imports import cpu_only_import
import pytest

from cuml.datasets import make_classification, make_blobs
from cuml.internals.safe_imports import cpu_only_import_from

np_assert_allclose = cpu_only_import_from("numpy.testing", "assert_allclose")

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")
gpu_sparse = gpu_only_import("cupyx.scipy.sparse")
cpu_sparse = cpu_only_import("scipy.sparse")
gpu_csr_matrix = gpu_only_import_from("cupyx.scipy.sparse", "csr_matrix")
gpu_csc_matrix = gpu_only_import_from("cupyx.scipy.sparse", "csc_matrix")
cpu_csr_matrix = cpu_only_import_from("scipy.sparse", "csr_matrix")


def to_output_type(array, output_type, order="F"):
    """Used to convert arrays while creating datasets
    for testing.

    Parameters
    ----------
    array : array
        Input array to convert
    output_type : string
        Type of to convert to

    Returns
    -------
    Converted array
    """
    if output_type == "scipy_csr":
        return cpu_sparse.csr_matrix(array.get())
    if output_type == "scipy_csc":
        return cpu_sparse.csc_matrix(array.get())
    if output_type == "scipy_coo":
        return cpu_sparse.coo_matrix(array.get())
    if output_type == "cupy_csr":
        if array.format in ["csc", "coo"]:
            return array.tocsr()
        else:
            return array
    if output_type == "cupy_csc":
        if array.format in ["csr", "coo"]:
            return array.tocsc()
        else:
            return array
    if output_type == "cupy_coo":
        if array.format in ["csr", "csc"]:
            return array.tocoo()
        else:
            return array

    if cpu_sparse.issparse(array):
        if output_type == "numpy":
            return array.todense()
        elif output_type == "cupy":
            return cp.array(array.todense())
        else:
            array = array.todense()
    elif gpu_sparse.issparse(array):
        if output_type == "numpy":
            return array.get().todense()
        elif output_type == "cupy":
            return array.todense()
        else:
            array = array.todense()

    cuml_array = input_to_cuml_array(array, order=order)[0]
    if output_type == "series" and len(array.shape) > 1:
        output_type = "cudf"

    output = cuml_array.to_output(output_type)

    if output_type in ["dataframe", "cudf"]:
        renaming = {i: "c" + str(i) for i in range(output.shape[1])}
        output = output.rename(columns=renaming)

    return output


def create_rand_clf(random_state):
    clf, _ = make_classification(
        n_samples=500,
        n_features=20,
        n_clusters_per_class=1,
        n_informative=12,
        n_classes=5,
        order="C",
        random_state=random_state,
    )
    return clf


def create_rand_blobs(random_state):
    blobs, _ = make_blobs(
        n_samples=500,
        n_features=20,
        centers=20,
        order="C",
        random_state=random_state,
    )
    return blobs


def create_rand_integers(random_state):
    cp.random.seed(random_state)
    randint = cp.random.randint(30, size=(500, 20)).astype(cp.float64)
    return randint


def create_positive_rand(random_state):
    cp.random.seed(random_state)
    rand = cp.random.rand(500, 20).astype(cp.float64)
    rand = cp.abs(rand) + 0.1
    return rand


def convert(dataset, conversion_format):
    converted_dataset = to_output_type(dataset, conversion_format)
    dataset = cp.asnumpy(dataset)
    return dataset, converted_dataset


def sparsify_and_convert(dataset, conversion_format, sparsify_ratio=0.3):
    """Randomly set values to 0 and produce a sparse array.

    Parameters
    ----------
    dataset : array
        Input array to convert
    conversion_format : string
        Type of sparse array :
        - scipy-csr: SciPy CSR sparse array
        - scipy-csc: SciPy CSC sparse array
        - scipy-coo: SciPy COO sparse array
        - cupy-csr: CuPy CSR sparse array
        - cupy-csc: CuPy CSC sparse array
        - cupy-coo: CuPy COO sparse array
    sparsify_ratio: float [0-1]
        Ratio of zeros in the sparse array

    Returns
    -------
    SciPy CSR array and converted array
    """
    random_loc = cp.random.choice(
        dataset.size, int(dataset.size * sparsify_ratio), replace=False
    )
    dataset.ravel()[random_loc] = 0

    if conversion_format.startswith("scipy"):
        dataset = cp.asnumpy(dataset)

    if conversion_format == "scipy-csr":
        converted_dataset = cpu_csr_matrix(dataset)
    elif conversion_format == "scipy-csc":
        converted_dataset = cpu_csc_matrix(dataset)
    elif conversion_format == "scipy-coo":
        converted_dataset = cpu_coo_matrix(dataset)
    elif conversion_format == "cupy-csr":
        converted_dataset = gpu_csr_matrix(dataset)
    elif conversion_format == "cupy-csc":
        converted_dataset = gpu_csc_matrix(dataset)
    elif conversion_format == "cupy-coo":
        np_array = cp.asnumpy(dataset)
        np_coo_array = cpu_coo_matrix(np_array)
        converted_dataset = gpu_coo_matrix(np_coo_array)

    if conversion_format.startswith("cupy"):
        dataset = cp.asnumpy(dataset)

    return cpu_csr_matrix(dataset), converted_dataset


@pytest.fixture(
    scope="session", params=["numpy", "dataframe", "cupy", "cudf", "numba"]
)
def clf_dataset(request, random_seed):
    clf = create_rand_clf(random_seed)
    return convert(clf, request.param)


@pytest.fixture(
    scope="session", params=["numpy", "dataframe", "cupy", "cudf", "numba"]
)
def blobs_dataset(request, random_seed):
    blobs = create_rand_blobs(random_seed)
    return convert(blobs, request.param)


@pytest.fixture(
    scope="session", params=["numpy", "dataframe", "cupy", "cudf", "numba"]
)
def int_dataset(request, random_seed):
    randint = create_rand_integers(random_seed)
    cp.random.seed(random_seed)
    random_loc = cp.random.choice(
        randint.size, int(randint.size * 0.3), replace=False
    )

    zero_filled = randint.copy().ravel()
    zero_filled[random_loc] = 0
    zero_filled = zero_filled.reshape(randint.shape)
    zero_filled = convert(zero_filled, request.param)

    one_filled = randint.copy().ravel()
    one_filled[random_loc] = 1
    one_filled = one_filled.reshape(randint.shape)
    one_filled = convert(one_filled, request.param)

    nan_filled = randint.copy().ravel()
    nan_filled[random_loc] = cp.nan
    nan_filled = nan_filled.reshape(randint.shape)
    nan_filled = convert(nan_filled, request.param)

    return zero_filled, one_filled, nan_filled


@pytest.fixture(
    scope="session", params=["scipy-csr", "scipy-csc", "cupy-csr", "cupy-csc"]
)
def sparse_clf_dataset(request, random_seed):
    clf = create_rand_clf(random_seed)
    return sparsify_and_convert(clf, request.param)


@pytest.fixture(
    scope="session",
    params=[
        "scipy-csr",
        "scipy-csc",
        "scipy-coo",
        "cupy-csr",
        "cupy-csc",
        "cupy-coo",
    ],
)
def sparse_dataset_with_coo(request, random_seed):
    clf = create_rand_clf(random_seed)
    return sparsify_and_convert(clf, request.param)


@pytest.fixture(
    scope="session", params=["scipy-csr", "scipy-csc", "cupy-csr", "cupy-csc"]
)
def sparse_blobs_dataset(request, random_seed):
    blobs = create_rand_blobs(random_seed)
    return sparsify_and_convert(blobs, request.param)


@pytest.fixture(
    scope="session", params=["scipy-csr", "scipy-csc", "cupy-csr", "cupy-csc"]
)
def sparse_int_dataset(request, random_seed):
    randint = create_rand_integers(random_seed)
    return sparsify_and_convert(randint, request.param)


@pytest.fixture(
    scope="session",
    params=[
        ("scipy-csr", np.nan),
        ("scipy-csc", np.nan),
        ("cupy-csr", np.nan),
        ("cupy-csc", np.nan),
        ("scipy-csr", 1.0),
        ("scipy-csc", 1.0),
        ("cupy-csr", 1.0),
        ("cupy-csc", 1.0),
    ],
)
def sparse_imputer_dataset(request, random_seed):
    datatype, val = request.param
    randint = create_rand_integers(random_seed)
    random_loc = cp.random.choice(
        randint.size, int(randint.size * 0.3), replace=False
    )

    randint.ravel()[random_loc] = val
    X_sp, X = sparsify_and_convert(randint, datatype, sparsify_ratio=0.15)
    X_sp = X_sp.tocsc()
    return val, X_sp, X


@pytest.fixture(
    scope="session", params=["numpy", "dataframe", "cupy", "cudf", "numba"]
)
def nan_filled_positive(request, random_seed):
    rand = create_positive_rand(random_seed)
    cp.random.seed(random_seed)
    random_loc = cp.random.choice(
        rand.size, int(rand.size * 0.3), replace=False
    )
    rand.ravel()[random_loc] = cp.nan
    rand = convert(rand, request.param)
    return rand


@pytest.fixture(scope="session", params=["scipy-csc", "cupy-csc"])
def sparse_nan_filled_positive(request, random_seed):
    rand = create_positive_rand(random_seed)
    cp.random.seed(random_seed)
    random_loc = cp.random.choice(
        rand.size, int(rand.size * 0.3), replace=False
    )
    rand.ravel()[random_loc] = cp.nan
    return sparsify_and_convert(rand, request.param)


def assert_allclose(actual, desired, rtol=1e-05, atol=1e-05, ratio_tol=None):
    if not isinstance(actual, np.ndarray):
        actual = to_output_type(actual, "numpy")
    if not isinstance(desired, np.ndarray):
        desired = to_output_type(desired, "numpy")

    if ratio_tol:
        assert actual.shape == desired.shape
        diff_ratio = (actual != desired).sum() / actual.size
        assert diff_ratio <= ratio_tol
    else:
        return np_assert_allclose(actual, desired, rtol=rtol, atol=atol)
