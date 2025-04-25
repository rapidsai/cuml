# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import numpy as np
from dask.distributed import get_worker
from raft_dask.common.comms import Comms, get_raft_comm_state

from cuml.dask.common.base import (
    BaseEstimator,
    DelayedPredictionMixin,
    DelayedTransformMixin,
    mnmg_import,
)
from cuml.dask.common.utils import wait_and_raise_from_futures
from cuml.internals.memory_utils import with_cupy_rmm


class DBSCAN(BaseEstimator, DelayedPredictionMixin, DelayedTransformMixin):
    """
    Multi-Node Multi-GPU implementation of DBSCAN.

    The whole dataset is copied to all the workers but the work is then
    divided by giving "ownership" of a subset to each worker: each worker
    computes a clustering by considering the relationships between those
    points and the rest of the dataset, and partial results are merged at
    the end to obtain the final clustering.

    Parameters
    ----------
    client : dask.distributed.Client
        Dask client to use
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    min_samples : int (default = 5)
        The number of samples in a neighborhood such that this group can be
        considered as an important core point (including the point itself).
    max_mbytes_per_batch : (optional) int64
        Calculate batch size using no more than this number of megabytes for
        the pairwise distance computation. This enables the trade-off between
        runtime and memory usage for making the N^2 pairwise distance
        computations more tractable for large numbers of samples.
        If you are experiencing out of memory errors when running DBSCAN, you
        can set this value based on the memory size of your device.
        Note: this option does not set the maximum total memory used in the
        DBSCAN computation and so this value will not be able to be set to
        the total memory available on the device.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    calc_core_sample_indices : (optional) boolean (default = True)
        Indicates whether the indices of the core samples should be calculated.
        The the attribute `core_sample_indices_` will not be used, setting this
        to False will avoid unnecessary kernel launches

    Notes
    -----
    For additional docs, see the documentation of the single-GPU DBSCAN model
    """

    def __init__(self, *, client=None, verbose=False, **kwargs):
        super().__init__(client=client, verbose=verbose, **kwargs)

    @staticmethod
    @mnmg_import
    def _func_fit(out_dtype):
        def _func(sessionId, data, **kwargs):
            from cuml.cluster.dbscan_mg import DBSCANMG as cumlDBSCAN

            handle = get_raft_comm_state(sessionId, get_worker())["handle"]

            return cumlDBSCAN(handle=handle, **kwargs).fit(
                data, out_dtype=out_dtype
            )

        return _func

    @with_cupy_rmm
    def fit(self, X, out_dtype="int32"):
        """
        Fit a multi-node multi-GPU DBSCAN model

        Parameters
        ----------
        X : array-like (device or host)
            Dense matrix containing floats or doubles.
            Acceptable formats: CUDA array interface compliant objects like
            CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas
            DataFrame/Series.
        out_dtype: dtype Determines the precision of the output labels array.
            default: "int32". Valid values are { "int32", np.int32,
            "int64", np.int64}.
        """
        if out_dtype not in ["int32", np.int32, "int64", np.int64]:
            raise ValueError(
                "Invalid value for out_dtype. "
                "Valid values are {'int32', 'int64', "
                "np.int32, np.int64}"
            )

        data = self.client.scatter(X, broadcast=True)

        comms = Comms(comms_p2p=True)
        comms.init()

        dbscan_fit = [
            self.client.submit(
                DBSCAN._func_fit(out_dtype),
                comms.sessionId,
                data,
                **self.kwargs,
                workers=[worker],
                pure=False,
            )
            for worker in comms.worker_addresses
        ]

        wait_and_raise_from_futures(dbscan_fit)

        comms.destroy()

        self._set_internal_model(dbscan_fit[0])

        return self

    def fit_predict(self, X, out_dtype="int32"):
        """
        Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : array-like (device or host)
            Dense matrix containing floats or doubles.
            Acceptable formats: CUDA array interface compliant objects like
            CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas
            DataFrame/Series.
        out_dtype: dtype Determines the precision of the output labels array.
            default: "int32". Valid values are { "int32", np.int32,
            "int64", np.int64}.
        Returns
        -------
        labels: array-like (device or host)
            Integer array of labels
        """
        self.fit(X, out_dtype)
        return self.get_combined_model().labels_

    def _get_param_names(self):
        return list(self.kwargs.keys())
