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

import cudf
import cuml
import cupy as cp
import numpy as np

from numba.cuda import is_cuda_array, as_cuda_array

global_input_types = [
    'numpy', 'numba', 'cupy', 'cudf'
]

test_output_types = {
    'numpy': np.ndarray,
    'cupy': cp.ndarray,
    'cudf': cudf.Series
}


@pytest.mark.parametrize('input_type', global_input_types)
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_default_global_output_type(input_type):
    dataset = get_small_dataset(input_type)

    dbscan_float = cuml.DBSCAN(eps=1.0, min_samples=1)
    dbscan_float.fit(dataset)

    res = dbscan_float.labels_

    if input_type == 'numba':
        assert is_cuda_array(res)
    else:
        assert isinstance(res, test_output_types[input_type])


@pytest.mark.parametrize('global_type', global_input_types)
@pytest.mark.parametrize('input_type', global_input_types)
def test_global_output_type(global_type, input_type):
    dataset = get_small_dataset(input_type)

    cuml.set_global_output_type(global_type)

    dbscan_float = cuml.DBSCAN(eps=1.0, min_samples=1)
    dbscan_float.fit(dataset)

    res = dbscan_float.labels_

    if global_type == 'numba':
        assert is_cuda_array(res)
    else:
        assert isinstance(res, test_output_types[global_type])


@pytest.mark.parametrize('global_type', global_input_types)
@pytest.mark.parametrize('context_type', global_input_types)
def test_output_type_context_mgr(global_type, context_type):
    dataset = get_small_dataset('numba')

    test_type = 'cupy' if global_type != 'cupy' else 'numpy'
    cuml.set_global_output_type(test_type)

    # use cuml context manager
    with cuml.using_output_type(context_type):
        dbscan_float = cuml.DBSCAN(eps=1.0, min_samples=1)
        dbscan_float.fit(dataset)

        res = dbscan_float.labels_

        if context_type == 'numba':
            assert is_cuda_array(res)
        else:
            assert isinstance(res, test_output_types[context_type])

    # use cuml again outside the context manager

    dbscan_float = cuml.DBSCAN(eps=1.0, min_samples=1)
    dbscan_float.fit(dataset)

    res = dbscan_float.labels_
    assert isinstance(res, test_output_types[test_type])

    # reset cuml global output type to 'input' for further tests
    cuml.set_global_output_type('input')


def get_small_dataset(output_type):
    ary = [[1.0, 4.0, 4.0], [2.0, 2.0, 2.0], [5.0, 1.0, 1.0]]
    ary = cp.asarray(ary)

    if output_type == 'numba':
        return as_cuda_array(ary)

    elif output_type == 'cupy':
        return ary

    elif output_type == 'numpy':
        return cp.asnumpy(ary)

    else:
        return cudf.DataFrame.from_gpu_matrix(as_cuda_array(ary))
