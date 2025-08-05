#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
import cupyx
import pytest
import scipy.sparse as scipy_sparse

from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray

test_input_types = ["cupy", "scipy"]


@pytest.mark.parametrize("input_type", test_input_types)
@pytest.mark.parametrize("sparse_format", ["csr", "coo", "csc"])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("convert_format", [True, False])
def test_input(input_type, sparse_format, dtype, convert_format):

    rand_func = cupyx.scipy.sparse if input_type == "cupy" else scipy_sparse

    X = rand_func.random(
        100, 100, format=sparse_format, density=0.5, dtype=dtype
    )

    if convert_format or sparse_format == "csr":
        X_m = SparseCumlArray(X, convert_format=convert_format)

        assert X.shape == X_m.shape
        assert X.nnz == X_m.nnz
        assert X.dtype == X_m.dtype

        # Just a sanity check
        assert isinstance(X_m.indptr, CumlArray)
        assert isinstance(X_m.indices, CumlArray)
        assert isinstance(X_m.data, CumlArray)

        assert X_m.indptr.dtype == cp.int32
        assert X_m.indices.dtype == cp.int32
        assert X_m.data.dtype == dtype

    elif not convert_format:
        with pytest.raises(ValueError):
            SparseCumlArray(X, convert_format=convert_format)


def test_nonsparse_input_fails():

    X = cp.random.random((100, 100))

    with pytest.raises(ValueError):
        SparseCumlArray(X)


@pytest.mark.parametrize("input_type", test_input_types)
def test_convert_to_dtype(input_type):

    rand_func = cupyx.scipy.sparse if input_type == "cupy" else scipy_sparse

    X = rand_func.random(100, 100, format="csr", density=0.5, dtype=cp.float64)

    X_m = SparseCumlArray(X, convert_to_dtype=cp.float32)

    assert X_m.dtype == cp.float32

    assert X_m.indptr.dtype == cp.int32
    assert X_m.indices.dtype == cp.int32
    assert X_m.data.dtype == cp.float32

    X_m = SparseCumlArray(X)

    assert X_m.dtype == X.dtype


@pytest.mark.parametrize("input_type", test_input_types)
def test_convert_index(input_type):

    rand_func = cupyx.scipy.sparse if input_type == "cupy" else scipy_sparse

    X = rand_func.random(100, 100, format="csr", density=0.5, dtype=cp.float64)

    # Will convert to 32-bit by default
    X.indptr = X.indptr.astype(cp.int64)
    X.indices = X.indices.astype(cp.int64)

    X_m = SparseCumlArray(X)

    assert X_m.indptr.dtype == cp.int32
    assert X_m.indices.dtype == cp.int32

    X_m = SparseCumlArray(X, convert_index=cp.int64)

    assert X_m.indptr.dtype == cp.int64
    assert X_m.indices.dtype == cp.int64


@pytest.mark.parametrize("input_type", test_input_types)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("output_type", test_input_types)
@pytest.mark.parametrize("output_format", [None, "coo", "csc"])
def test_output(input_type, output_type, dtype, output_format):

    rand_func = cupyx.scipy.sparse if input_type == "cupy" else scipy_sparse

    X = rand_func.random(100, 100, format="csr", density=0.5, dtype=dtype)

    X_m = SparseCumlArray(X)

    output = X_m.to_output(output_type, output_format=output_format)

    if output_type == "scipy":
        if output_format is None:
            assert isinstance(output, scipy_sparse.csr_matrix)
        elif output_format == "coo":
            assert isinstance(output, scipy_sparse.coo_matrix)
        elif output_format == "csc":
            assert isinstance(output, scipy_sparse.csc_matrix)
        else:
            pytest.fail("unecpected output format")
    else:
        if output_format is None:
            assert isinstance(output, cupyx.scipy.sparse.csr_matrix)
        elif output_format == "coo":
            assert isinstance(output, cupyx.scipy.sparse.coo_matrix)
        elif output_format == "csc":
            assert isinstance(output, cupyx.scipy.sparse.csc_matrix)
        else:
            pytest.fail("unecpected output format")

    cp.testing.assert_array_equal(X.todense(), output.todense())
