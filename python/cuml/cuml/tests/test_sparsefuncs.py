# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
from cuml.internals.safe_imports import gpu_only_import
from cuml.common.sparsefuncs import csr_row_normalize_l1
from cuml.common.sparsefuncs import csr_row_normalize_l2
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l1
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l2

import pytest
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
sp = cpu_only_import("scipy.sparse")
cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")


@pytest.mark.parametrize(
    "norm, ref_norm",
    [
        (csr_row_normalize_l1, inplace_csr_row_normalize_l1),
        (csr_row_normalize_l2, inplace_csr_row_normalize_l2),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed, shape", [(10, (10, 5)), (123, (500, 12))])
def test_csr_norms(norm, ref_norm, dtype, seed, shape):
    X = np.random.RandomState(seed).randn(*shape).astype(dtype)
    X_csr = sp.csr_matrix(X)
    X_csr_gpu = cupyx.scipy.sparse.csr_matrix(X_csr)

    norm(X_csr_gpu)
    ref_norm(X_csr)

    # checks that array have been changed inplace
    assert cp.any(cp.not_equal(X_csr_gpu.todense(), cp.array(X)))

    cp.testing.assert_array_almost_equal(X_csr_gpu.todense(), X_csr.todense())
