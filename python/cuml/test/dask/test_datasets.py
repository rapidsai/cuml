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

from dask_cuda import LocalCUDACluster

from dask.distributed import Client, wait


@pytest.mark.parametrize('nrows', [1e3, 1e4])
@pytest.mark.parametrize('ncols', [1, 10, 100])
@pytest.mark.parametrize('centers', [10, 50, 1000])
@pytest.mark.parametrize("cluster_std", [0.1, 0.001, 0.0001])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nparts", [1, 5, 10])
def test_make_blobs(nrows, ncols, centers, cluster_std, dtype, nparts):

    cluster = LocalCUDACluster()
    c = Client(cluster)

    from cuml.dask.datasets import make_blobs

    X, y = make_blobs(nrows, ncols, n_parts=nparts, centers=centers,
                      cluster_std=cluster_std, dtype=dtype)

    assert X.npartitions == nparts
    assert y.npartitions == nparts

    X = X.compute()
    y = y.compute()

    assert X.shape == (nrows, ncols)
    assert y.shape == (nrows, 1)

    assert len(y[0].unique()) == centers

    assert X.dtypes.unique() == [dtype]

    c.close()
    cluster.close()
