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

from dask.distributed import Client, wait

import numpy as np


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [1e3, 1e5, 5e5])
@pytest.mark.parametrize("ncols", [10, 30])
@pytest.mark.parametrize("n_parts", [None, 50])
def test_end_to_end(nrows, ncols, n_parts, cluster):

    print("Getting cluster")
    client = Client(cluster)

    from cuml.dask.decomposition import PCA

    from cuml.dask.datasets import make_blobs

    print("Making blobs!")

    X_cudf, _ = make_blobs(nrows, ncols, 1, n_parts,
                           cluster_std=0.01, verbose=True,
                           random_state=10, dtype=np.float32)

    wait(X_cudf)

    print("DONE!")

    print("Creating PCA")

    cumlModel = PCA(verbose=True)

    print("Calling FIT!")
    cumlModel.fit(X_cudf)


    print("DONE!")

    # xformed = cumlModel.transform(X_cudf)
    #
    # print(str(xformed))

    client.close()
