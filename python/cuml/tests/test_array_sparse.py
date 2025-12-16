#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import cupyx.scipy.sparse
import pytest
import scipy.sparse

from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray

test_input_kinds = ["cupy", "scipy", "scipy-array"]


def rand(m, n, dtype=cp.float32, format="csr", kind="cupy"):
    if kind == "scipy-array":
        return scipy.sparse.random_array(
            (m, n), dtype=dtype, format=format, density=0.5, random_state=42
        )
    assert kind in ("cupy", "scipy")
    func = (
        scipy.sparse.random if kind == "scipy" else cupyx.scipy.sparse.random
    )
    return func(m, n, dtype=dtype, format=format, density=0.5, random_state=42)


@pytest.mark.parametrize("kind", test_input_kinds)
@pytest.mark.parametrize("format", ["csr", "coo", "csc"])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("convert_format", [True, False])
def test_input(kind, format, dtype, convert_format):
    X = rand(100, 100, kind=kind, format=format, dtype=dtype)

    if convert_format or format == "csr":
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


def test_non_2D_sparse_array():
    X = scipy.sparse.random_array((10, 20, 30), random_state=42)

    with pytest.raises(ValueError, match="Expected 2D input"):
        SparseCumlArray(X)


def test_shape_checks():
    X = rand(8, 9)

    with pytest.raises(ValueError, match="Expected 5 rows but got 8 rows"):
        SparseCumlArray(X, check_rows=5)

    with pytest.raises(
        ValueError, match="Expected 5 columns but got 9 columns"
    ):
        SparseCumlArray(X, check_cols=5)


@pytest.mark.parametrize("kind", test_input_kinds)
def test_convert_to_dtype(kind):
    X = rand(100, 100, kind=kind, format="csr", dtype=cp.float64)

    X_m = SparseCumlArray(X, convert_to_dtype=cp.float32)

    assert X_m.dtype == cp.float32

    assert X_m.indptr.dtype == cp.int32
    assert X_m.indices.dtype == cp.int32
    assert X_m.data.dtype == cp.float32

    X_m = SparseCumlArray(X)

    assert X_m.dtype == X.dtype


@pytest.mark.parametrize("kind", test_input_kinds)
def test_convert_index(kind):
    X = rand(100, 100, kind=kind, format="csr", dtype=cp.float64)

    # Will convert to 32-bit by default
    X.indptr = X.indptr.astype(cp.int64)
    X.indices = X.indices.astype(cp.int64)

    X_m = SparseCumlArray(X)

    assert X_m.indptr.dtype == cp.int32
    assert X_m.indices.dtype == cp.int32

    X_m = SparseCumlArray(X, convert_index=cp.int64)

    assert X_m.indptr.dtype == cp.int64
    assert X_m.indices.dtype == cp.int64


@pytest.mark.parametrize("kind", test_input_kinds)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("output_type", ["scipy", "cupy"])
@pytest.mark.parametrize("output_format", [None, "coo", "csc"])
def test_output(kind, output_type, dtype, output_format):
    X = rand(100, 100, kind=kind, format="csr", dtype=dtype)

    X_m = SparseCumlArray(X)

    output = X_m.to_output(output_type, output_format=output_format)

    if output_type == "scipy":
        if output_format is None:
            assert isinstance(output, scipy.sparse.csr_matrix)
        elif output_format == "coo":
            assert isinstance(output, scipy.sparse.coo_matrix)
        elif output_format == "csc":
            assert isinstance(output, scipy.sparse.csc_matrix)
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
