# Copyright (c) 2024, NVIDIA CORPORATION.
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
import pytest
from cuml.dask.common import utils as dask_utils
from functools import partial
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as skLR
from cuml.internals.safe_imports import cpu_only_import
from cuml.testing.utils import array_equal
from scipy.sparse import csr_matrix

pd = cpu_only_import("pandas")
np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")
dask_cudf = gpu_only_import("dask_cudf")
cudf = gpu_only_import("cudf")


def make_classification_dataset(
    datatype, nrows, ncols, n_info, n_redundant=2, n_classes=2
):
    X, y = make_classification(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=0,
    )
    X = X.astype(datatype)
    y = y.astype(datatype)

    return X, y


def _prep_training_data_sparse(c, X_train, y_train, partitions_per_worker):
    "The implementation follows test_dask_tfidf.create_cp_sparse_dask_array"
    import dask.array as da

    workers = c.has_what().keys()
    target_n_partitions = partitions_per_worker * len(workers)

    def cal_chunks(dataset, n_partitions):

        n_samples = dataset.shape[0]
        n_samples_per_part = int(n_samples / n_partitions)
        chunk_sizes = [n_samples_per_part] * n_partitions
        samples_last_row = n_samples - (
            (n_partitions - 1) * n_samples_per_part
        )
        chunk_sizes[-1] = samples_last_row
        return tuple(chunk_sizes)

    assert (
        X_train.shape[0] == y_train.shape[0]
    ), "the number of data records is not equal to the number of labels"
    target_chunk_sizes = cal_chunks(X_train, target_n_partitions)

    X_da = da.from_array(X_train, chunks=(target_chunk_sizes, -1))
    y_da = da.from_array(y_train, chunks=target_chunk_sizes)

    X_da, y_da = dask_utils.persist_across_workers(
        c, [X_da, y_da], workers=workers
    )
    return X_da, y_da


pytestmark = pytest.mark.mg


@pytest.mark.mg
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize(
    "regularization",
    [
        ("none", 1.0, None),
        ("l2", 2.0, None),
        ("l1", 2.0, None),
        ("elasticnet", 2.0, 0.2),
    ],
)
def test_standardization_sparse(fit_intercept, regularization):
    from dask_cuda import LocalCUDACluster
    from dask_cuda.utils_test import IncreasedCloseTimeoutNanny
    from dask.distributed import Client

    cluster = LocalCUDACluster(
        protocol="tcp",
        scheduler_port=0,
        worker_class=IncreasedCloseTimeoutNanny,
    )
    client = Client(cluster)

    import subprocess

    print("debug: reporting gpu status")
    gpu_status = subprocess.run("nvidia-smi")
    print(gpu_status.stdout)
    print(gpu_status.stderr)

    n_rows = int(1e5)
    n_cols = 25
    n_info = 15
    n_classes = 4

    nnz = int(n_rows * n_cols * 0.3)  # number of non-zero values
    # tolerance = 0.005

    datatype = np.float32
    n_parts = 2
    max_iter = 5  # cannot set this too large. Observed GPU-specific coefficients when objective converges at 0.

    penalty = regularization[0]
    C = regularization[1]
    l1_ratio = regularization[2]

    est_params = {
        "penalty": penalty,
        "C": C,
        "l1_ratio": l1_ratio,
        "fit_intercept": fit_intercept,
        "max_iter": max_iter,
    }

    def make_classification_with_nnz(
        datatype, n_rows, n_cols, n_info, n_classes, nnz
    ):
        assert n_rows * n_cols >= nnz

        X, y = make_classification_dataset(
            datatype, n_rows, n_cols, n_info, n_classes=n_classes
        )
        X = X.flatten()
        num_zero = len(X) - nnz
        zero_indices = np.random.choice(
            a=range(len(X)), size=num_zero, replace=False
        )
        X[zero_indices] = 0
        X_res = X.reshape(n_rows, n_cols)
        return X_res, y

    X_origin, y = make_classification_with_nnz(
        datatype, n_rows, n_cols, n_info, n_classes, nnz
    )
    X = csr_matrix(X_origin)
    assert X.nnz == nnz and X.shape == (n_rows, n_cols)

    # from sklearn.preprocessing import StandardScaler

    # scaler = StandardScaler(with_mean=fit_intercept, with_std=True)
    # scaler.fit(X_origin)
    # scaler.scale_ = np.sqrt(scaler.var_ * len(X_origin) / (len(X_origin) - 1))
    # X_scaled = scaler.transform(X_origin)

    X_da, y_da = _prep_training_data_sparse(
        client, X, y, partitions_per_worker=n_parts
    )
    from cuml.dask.linear_model import LogisticRegression as cumlLBFGS_dask

    lr_on = cumlLBFGS_dask(standardization=True, verbose=True, **est_params)
    lr_on.fit(X_da, y_da)

    # lron_coef_origin = lr_on.coef_ * scaler.scale_
    # if fit_intercept is True:
    #     lron_intercept_origin = lr_on.intercept_ + np.dot(
    #         lr_on.coef_, scaler.mean_
    #     )
    # else:
    #     lron_intercept_origin = lr_on.intercept_

    # from cuml.linear_model import LogisticRegression as SG

    # sg = SG(**est_params)
    # sg.fit(X_scaled, y)

    # assert array_equal(lron_coef_origin, sg.coef_, tolerance)
    # assert array_equal(lron_intercept_origin, sg.intercept_, tolerance)

    client.close()
    cluster.close()
