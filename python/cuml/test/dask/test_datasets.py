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

import numpy as np

from dask.distributed import Client

from cuml.test.utils import unit_param, quality_param, stress_param


@pytest.mark.parametrize('nrows', [unit_param(1e3), quality_param(1e5),
                                   stress_param(1e6)])
@pytest.mark.parametrize('ncols', [unit_param(10), quality_param(100),
                                   stress_param(1000)])
@pytest.mark.parametrize('centers', [10])
@pytest.mark.parametrize("cluster_std", [0.1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nparts", [unit_param(1), unit_param(7),
                                    quality_param(100),
                                    stress_param(1000)])
@pytest.mark.parametrize("order", ['F', 'C'])
@pytest.mark.parametrize("output", ['array', 'dataframe'])
def test_make_blobs(nrows,
                    ncols,
                    centers,
                    cluster_std,
                    dtype,
                    nparts,
                    cluster,
                    order,
                    output):

    c = Client(cluster)
    try:
        from cuml.dask.datasets import make_blobs

        X, y = make_blobs(nrows, ncols,
                          centers=centers,
                          cluster_std=cluster_std,
                          dtype=dtype,
                          n_parts=nparts,
                          output=output,
                          order=order)

        assert X.npartitions == nparts
        assert y.npartitions == nparts

        X = X.compute()
        y = y.compute()

        assert X.shape == (nrows, ncols)
        assert y.shape == (nrows, 1)

        if output == 'dataframe':
            assert len(y[0].unique()) == centers
            assert X.dtypes.unique() == [dtype]

        elif output == 'array':
            import cupy as cp
            assert len(cp.unique(y)) == centers
            assert y.dtype == dtype

    finally:
        c.close()
