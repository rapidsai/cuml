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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nparts", [1, 5, 7])
def test_to_dask_df(dtype, nparts, cluster):

    c = Client(cluster)

    try:

        from cuml.dask.common.dask_df_utils import to_dask_df
        from cuml.dask.datasets import make_blobs

        X, y = make_blobs(1e3, 25, n_parts=nparts, dtype=dtype)

        X_df = to_dask_df(X)
        y_df = to_dask_df(y)

        X_df_local = X_df.compute()
        y_df_local = y_df.compute()

        X_local = X.compute()
        y_local = y.compute()

        assert X_local.shape == X_df_local.shape
        assert y_local.shape == y_df_local.shape

        assert X_local.dtypes.unique() == X_df_local.dtypes.unique()
        assert y_local.dtypes.unique() == y_df_local.dtypes.unique()

    finally:
        c.close()
