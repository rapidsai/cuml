#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

import pytest

from cuml.common.array import CumlArray
from cuml.common.array_sparse import SparseCumlArray

import scipy.sparse
import cupy as cp
import cupyx

test_input_types = [
    'cupy', 'scipy'
]


@pytest.mark.parametrize('input_type', test_input_types)
@pytest.mark.parametrize('dtype', [cp.float32, cp.float64])
def test_input(input_type, dtype):

    rand_func = cupyx.scipy.sparse if input_type == 'cupy' else scipy.sparse

    X = rand_func.random(100, 100, format='csr', density=0.5, dtype=dtype)

    X_m = SparseCumlArray(X)

    assert X.shape == X_m.shape
    assert X.nnz == X_m.nnz

    # Just a sanity check
    assert isinstance(X_m.indptr, CumlArray)
    assert isinstance(X_m.indices, CumlArray)
    assert isinstance(X_m.data, CumlArray)

    assert X_m.indptr.dtype == cp.int32
    assert X_m.indices.dtype == cp.int32
    assert X_m.data.dtype == dtype


@pytest.mark.parametrize('input_type', test_input_types)
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('output_type', test_input_types)
def test_output(input_type, output_type, dtype):

    rand_func = cupyx.scipy.sparse if input_type == 'cupy' else scipy.sparse

    X = rand_func.random(100, 100, format='csr', density=0.5, dtype=dtype)

    X_m = SparseCumlArray(X)

    output = X_m.to_output(output_type)

    if output_type == 'scipy':
        assert isinstance(output, scipy.sparse.csr_matrix)
    else:
        assert isinstance(output, cupyx.scipy.sparse.csr_matrix)

    cp.testing.assert_array_equal(X.todense(), output.todense())
