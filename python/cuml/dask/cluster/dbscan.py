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

import numpy as np

from cuml.dask.common.base import BaseEstimator
from cuml.dask.common.base import DelayedPredictionMixin
from cuml.dask.common.base import DelayedTransformMixin
from cuml.dask.common.base import mnmg_import

from cuml.dask.common.input_utils import concatenate
from cuml.dask.common.input_utils import DistributedDataHandler

from cuml.raft.dask.common.comms import Comms
from cuml.raft.dask.common.comms import worker_state

from cuml.dask.common.input_utils import _get_datatype_from_inputs
from cuml.dask.common.utils import wait_and_raise_from_futures

from cuml.common.memory_utils import with_cupy_rmm

class DBSCAN(BaseEstimator, DelayedPredictionMixin, DelayedTransformMixin):
    """
    Multi-Node Multi-GPU implementation of DBSCAN.

    For more information on this implementation, refer to the
    documentation for single-GPU DBSCAN.

    TODO: complete docs
    """

    def __init__(self, client=None, verbose=False, **kwargs):
        super(DBSCAN, self).__init__(client=client,
                                     verbose=verbose,
                                     **kwargs)

    @staticmethod
    @mnmg_import
    def _func_fit(out_dtype):
        def _func(sessionId, data, verbose, **kwargs):
            from cuml.cluster.dbscan_mg import DBSCANMG as cumlDBSCAN
            handle = worker_state(sessionId)["handle"]

            return cumlDBSCAN(handle=handle, verbose=verbose, **kwargs
                              ).fit(data, out_dtype=out_dtype)
        return _func

    @with_cupy_rmm # TODO: is the decorator needed?
    def fit(self, X, out_dtype="int32"):
        """
        Fit a multi-node multi-GPU DBSCAN model

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            Training data to cluster.
            TODO: fix this docstring
        out_dtype: dtype Determines the precision of the output labels array.
            default: "int32". Valid values are { "int32", np.int32,
            "int64", np.int64}.
        """
        if out_dtype not in ["int32", np.int32, "int64", np.int64]:
            raise ValueError("Invalid value for out_dtype. "
                             "Valid values are {'int32', 'int64', "
                             "np.int32, np.int64}")

        data = self.client.scatter(X, broadcast=True)

        comms = Comms(comms_p2p=True)
        comms.init()

        dbscan_fit = [self.client.submit(DBSCAN._func_fit(out_dtype),
                                         comms.sessionId,
                                         data,
                                         self.verbose,
                                         **self.kwargs,
                                         workers=[worker],
                                         pure=False)
                      for worker in comms.worker_addresses]

        wait_and_raise_from_futures(dbscan_fit)

        comms.destroy()

        self._set_internal_model(dbscan_fit[0])

        return self

    def fit_predict(self, X, out_dtype="int32"):
        """
        Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            Data to predict
        TODO: fix this docstring

        Returns
        -------
        result: Dask cuDF DataFrame or CuPy backed Dask Array
            Distributed object containing predictions

        """
        self.fit(X, out_dtype)
        return self.get_combined_model().labels_

    def get_param_names(self):
        return list(self.kwargs.keys())
