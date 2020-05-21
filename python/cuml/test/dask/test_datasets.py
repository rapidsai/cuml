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

from cuml.dask.datasets.blobs import make_blobs
from cuml.dask.common.input_utils import DistributedDataHandler

from cuml.test.utils import unit_param, quality_param, stress_param

from cuml.dask.common.part_utils import _extract_partitions


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
def test_make_blobs(nrows,
                    ncols,
                    centers,
                    cluster_std,
                    dtype,
                    nparts,
                    order,
                    cluster):

    c = Client(cluster)
    try:
        nrows = int(nrows)
        X, y = make_blobs(nrows, ncols,
                          centers=centers,
                          cluster_std=cluster_std,
                          dtype=dtype,
                          n_parts=nparts,
                          order=order,
                          client=c)

        assert len(X.chunks[0]) == nparts
        assert len(y.chunks[0]) == nparts

        assert X.shape == (nrows, ncols)
        assert y.shape == (nrows, )

        y_local = y.compute()
        assert len(cp.unique(y_local)) == centers

        X_ddh = DistributedDataHandler.create(data=X, client=c)
        X_first = X_ddh.gpu_futures[0][1].result()

        if order == 'F':
            assert X_first.flags['F_CONTIGUOUS']
        elif order == 'C':
            assert X_first.flags['C_CONTIGUOUS']

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
@pytest.mark.parametrize('n_parts', [4, 23])
@pytest.mark.parametrize('order', ['F', 'C'])
@pytest.mark.parametrize('use_full_low_rank', [True, False])
def test_make_regression(n_samples, n_features, n_informative,
                         n_targets, bias, effective_rank,
                         tail_strength, noise, shuffle,
                         coef, n_parts, order,
                         use_full_low_rank, cluster):
    c = Client(cluster)
    try:
        from cuml.dask.datasets import make_regression

        result = make_regression(n_samples=n_samples, n_features=n_features,
                                 n_informative=n_informative,
                                 n_targets=n_targets, bias=bias,
                                 effective_rank=effective_rank, noise=noise,
                                 shuffle=shuffle, coef=coef,
                                 n_parts=n_parts,
                                 use_full_low_rank=use_full_low_rank,
                                 order=order)

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

        data_ddh = DistributedDataHandler.create(data=(out, values),
                                                 client=c)
        out_part, value_part = data_ddh.gpu_futures[0][1].result()

        if coef:
            coefs_ddh = DistributedDataHandler.create(data=coefs,
                                                      client=c)
            coefs_part = coefs_ddh.gpu_futures[0][1].result()
        if order == 'F':
            assert out_part.flags['F_CONTIGUOUS']
            if n_targets > 1:
                assert value_part.flags['F_CONTIGUOUS']
                if coef:
                    assert coefs_part.flags['F_CONTIGUOUS']
        elif order == 'C':
            assert out_part.flags['C_CONTIGUOUS']
            if n_targets > 1:
                assert value_part.flags['C_CONTIGUOUS']
                if coef:
                    assert coefs_part.flags['C_CONTIGUOUS']

    finally:
        c.close()


@pytest.mark.parametrize('n_samples', [500, 1000])
@pytest.mark.parametrize('n_features', [50, 100])
@pytest.mark.parametrize('hypercube', [True, False])
@pytest.mark.parametrize('n_classes', [2, 4])
@pytest.mark.parametrize('n_clusters_per_class', [2, 4])
@pytest.mark.parametrize('n_informative', [7, 20])
@pytest.mark.parametrize('random_state', [None, 1234])
@pytest.mark.parametrize('n_parts', [2, 23])
@pytest.mark.parametrize('order', ['C', 'F'])
def test_make_classification(n_samples, n_features, hypercube, n_classes,
                             n_clusters_per_class, n_informative,
                             random_state, n_parts, order, cluster):
    client = Client(cluster)
    try:
        from cuml.dask.datasets.classification import make_classification

        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                   n_classes=n_classes, hypercube=hypercube,
                                   n_clusters_per_class=n_clusters_per_class,
                                   n_informative=n_informative,
                                   random_state=random_state, n_parts=n_parts,
                                   order=order)
        assert(len(X.chunks[0])) == n_parts
        assert(len(X.chunks[1])) == 1

        assert X.shape == (n_samples, n_features)
        assert y.shape == (n_samples, )

        assert len(X.chunks[0]) == n_parts
        assert len(y.chunks[0]) == n_parts

        import cupy as cp
        y_local = y.compute()
        assert len(cp.unique(y_local)) == n_classes

        X_parts = client.sync(_extract_partitions, X)
        X_first = X_parts[0][1].result()

        if order == 'F':
            assert X_first.flags['F_CONTIGUOUS']
        elif order == 'C':
            assert X_first.flags['C_CONTIGUOUS']

    finally:
        client.close()
