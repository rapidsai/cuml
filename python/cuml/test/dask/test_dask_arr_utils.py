# Copyright (c) 2019, NVIDIA CORPORATION.
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

from cuml.test.utils import array_equal

import cudf
import cupy as cp

from dask.distributed import Client, wait

@pytest.mark.parametrize("input_type", ["dataframe",
                                        "scipysparse",
                                        "cupysparse",
                                        "numpy",
                                        "cupy"
                                        ])
@pytest.mark.parametrize("nrows", [1000])
@pytest.mark.parametrize("ncols", [10])
def test_to_sp_dask_array(input_type, nrows, ncols, cluster):

    c = Client(cluster)

    try:

        from cuml.dask.common import to_sp_dask_array

        a = cp.sparse.random(nrows, ncols, format='csr', dtype=cp.float32)
        if input_type == "dataframe":
            inp = cudf.DataFrame.from_gpu_matrix(a.todense())
        elif input_type == "scipysparse":
            inp = a.get()
        elif input_type == "cupysparse":
            inp = a
        elif input_type == "numpy":
            inp = a.get().todense()
        elif input_type == "cupy":
            inp = a.todense()

        arr = to_sp_dask_array(inp, c)

        b = arr.compute().get()

        assert array_equal(a.todense().get(), b.todense())



    finally:
        c.close()

