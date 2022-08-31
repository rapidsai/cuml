# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
import cupy as cp
import cupyx

from cuml.prims.stats import cov
from cuml.testing.utils import array_equal


@pytest.mark.parametrize("nrows", [1000])
@pytest.mark.parametrize("ncols", [500, 1500])
@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_cov(nrows, ncols, sparse, dtype):
    if sparse:
        x = cupyx.scipy.sparse.random(nrows, ncols, density=0.07,
                                      format='csr', dtype=dtype)
    else:
        x = cp.random.random((nrows, ncols), dtype=dtype)

    cov_result = cov(x, x)

    assert cov_result.shape == (ncols, ncols)

    if sparse:
        x = x.todense()
    local_cov = cp.cov(x, rowvar=False, ddof=0)

    assert array_equal(cov_result, local_cov, 1e-6, with_sign=True)
