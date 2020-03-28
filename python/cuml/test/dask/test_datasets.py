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

import dask.array as da
import numpy as np
import cupy as cp

from dask.distributed import Client

from cuml.dask.datasets import make_blobs

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
        X, y = make_blobs(nrows, ncols,
                          centers=centers,
                          cluster_std=cluster_std,
                          dtype=dtype,
                          n_parts=nparts,
                          output=output,
                          order=order)

        assert X.npartitions == nparts
        assert y.npartitions == nparts

        X_local = X.compute()
        y_local = y.compute()

        assert X_local.shape == (nrows, ncols)

        if output == 'dataframe':
            assert len(y_local[0].unique()) == centers
            assert X_local.dtypes.unique() == [dtype]
            assert y_local.shape == (nrows, 1)

        elif output == 'array':
            import cupy as cp
            assert len(cp.unique(y_local)) == centers
            assert y_local.dtype == dtype
            assert y_local.shape == (nrows, )

    finally:
        c.close()


@pytest.mark.parametrize('n_samples', [unit_param(int(1e3)),
                         stress_param(int(1e6))])
@pytest.mark.parametrize('n_features', [unit_param(int(1e2)),
                         stress_param(int(1e3))])
@pytest.mark.parametrize('n_informative', [7])
@pytest.mark.parametrize('n_targets', [1, 3])
@pytest.mark.parametrize('bias', [-4.0])
@pytest.mark.parametrize('effective_rank', [None, 6])
@pytest.mark.parametrize('tail_strength', [0.5])
@pytest.mark.parametrize('noise', [1.0])
@pytest.mark.parametrize('shuffle', [True, False])
@pytest.mark.parametrize('coef', [True, False])
@pytest.mark.parametrize('random_state', [None, 1234])
@pytest.mark.parametrize('n_parts', [unit_param(1),
                         stress_param(3)])
def test_make_regression(n_samples, n_features, n_informative,
                         n_targets, bias, effective_rank,
                         tail_strength, noise, shuffle,
                         coef, random_state, n_parts,
                         cluster):
    c = Client(cluster)
    try:
        from cuml.dask.datasets import make_regression

        result = make_regression(n_samples=n_samples, n_features=n_features,
                                 n_informative=n_informative,
                                 n_targets=n_targets, bias=bias,
                                 effective_rank=effective_rank, noise=noise,
                                 shuffle=shuffle, coef=coef,
                                 random_state=random_state, n_parts=n_parts)

        if coef:
            out, values, coefs = result
        else:
            out, values = result

        assert out.shape == (n_samples, n_features), "out shape mismatch"

        if n_targets > 1:
            assert values.shape == (n_samples, n_targets), \
                   "values shape mismatch"
        else:
            assert values.shape == (n_samples,), "values shape mismatch"

        assert len(out.chunks[0]) == n_parts
        assert len(out.chunks[1]) == 1

        if coef:
            if n_targets > 1:
                assert coefs.shape == (n_features, n_targets), \
                       "coefs shape mismatch"
                assert len(coefs.chunks[1]) == 1
            else:
                assert coefs.shape == (n_features,), "coefs shape mismatch"
                assert len(coefs.chunks[0]) == 1

            test1 = da.all(da.sum(coefs != 0.0, axis=0) == n_informative)

            std_test2 = da.std(values - (da.dot(out, coefs) + bias), axis=0)

            test1, std_test2 = da.compute(test1, std_test2)

            diff = cp.abs(1.0 - std_test2)
            test2 = cp.all(diff < 1.5 * 10**(-1.))

            assert test1, \
                "Unexpected number of informative features"

            assert test2, "Unexpectedly incongruent outputs"

    finally:
        c.close()
