#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

from cuml.dask.common.input_utils import DistributedDataHandler
from cuml.dask.common.input_utils import to_output
from cuml.dask.common import parts_to_ranks
from cuml.dask.common import flatten_grouped_results
from cuml.dask.common.utils import raise_mg_import_exception
from cuml.dask.common.utils import wait_and_raise_from_futures
from cuml.raft.dask.common.comms import get_raft_comm_state
from cuml.dask.neighbors import NearestNeighbors
import dask.array as da
from uuid import uuid1


class KNeighborsRegressor(NearestNeighbors):
    """
    Multi-node Multi-GPU K-Nearest Neighbors Regressor Model.

    K-Nearest Neighbors Regressor is an instance-based learning technique,
    that keeps training samples around for prediction, rather than trying
    to learn a generalizable set of model parameters.

    Parameters
    ----------
    n_neighbors : int (default=5)
        Default number of neighbors to query
    batch_size: int (optional, default 2000000)
        Maximum number of query rows processed at once. This parameter can
        greatly affect the throughput of the algorithm. The optimal setting
        of this value will vary for different layouts and index to query
        ratios, but it will require `batch_size * n_features * 4` bytes of
        additional memory on each worker hosting index partitions.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    """
    def __init__(self, client=None, streams_per_handle=0,
                 verbose=False, **kwargs):
        super(KNeighborsRegressor, self).__init__(client=client,
                                                  verbose=verbose,
                                                  **kwargs)
        self.streams_per_handle = streams_per_handle

    def fit(self, X, y):
        """
        Fit a multi-node multi-GPU K-Nearest Neighbors Regressor index

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Index data.
            Acceptable formats: dask CuPy/NumPy/Numba Array

        y : array-like (device or host) shape = (n_samples, n_features)
            Index output data.
            Acceptable formats: dask CuPy/NumPy/Numba Array

        Returns
        -------
        self : KNeighborsRegressor model
        """
        self.data_handler = \
            DistributedDataHandler.create(data=[X, y],
                                          client=self.client)
        self.n_outputs = y.shape[1]

        return self

    @staticmethod
    def _func_create_model(sessionId, **kwargs):
        try:
            from cuml.neighbors.kneighbors_regressor_mg import \
                KNeighborsRegressorMG as cumlKNN
        except ImportError:
            raise_mg_import_exception()

        handle = get_raft_comm_state(sessionId)["handle"]
        return cumlKNN(handle=handle, **kwargs)

    @staticmethod
    def _func_predict(model, index, index_parts_to_ranks, index_nrows,
                      query, query_parts_to_ranks, query_nrows,
                      ncols, rank, n_output, convert_dtype):
        return model.predict(
            index, index_parts_to_ranks, index_nrows,
            query, query_parts_to_ranks, query_nrows,
            ncols, rank, n_output, convert_dtype
        )

    def predict(self, X, convert_dtype=True):
        """
        Predict outputs for a query from previously stored index
        and outputs.
        The process is done in a multi-node multi-GPU fashion.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Query data.
            Acceptable formats: dask cuDF, dask CuPy/NumPy/Numba Array

        convert_dtype : bool, optional (default = True)
            When set to True, the predict method will automatically
            convert the data to the right formats.

        Returns
        -------
        predictions : Dask futures or Dask CuPy Arrays
        """
        query_handler = \
            DistributedDataHandler.create(data=X,
                                          client=self.client)
        self.datatype = query_handler.datatype

        comms = KNeighborsRegressor._build_comms(self.data_handler,
                                                 query_handler,
                                                 self.streams_per_handle)

        worker_info = comms.worker_info(comms.worker_addresses)

        """
        Build inputs and outputs
        """
        self.data_handler.calculate_parts_to_sizes(comms=comms)
        query_handler.calculate_parts_to_sizes(comms=comms)

        data_parts_to_ranks, data_nrows = \
            parts_to_ranks(self.client,
                           worker_info,
                           self.data_handler.gpu_futures)

        query_parts_to_ranks, query_nrows = \
            parts_to_ranks(self.client,
                           worker_info,
                           query_handler.gpu_futures)

        """
        Each Dask worker creates a single model
        """
        key = uuid1()
        models = dict([(worker, self.client.submit(
            self._func_create_model,
            comms.sessionId,
            **self.kwargs,
            workers=[worker],
            key="%s-%s" % (key, idx)))
            for idx, worker in enumerate(comms.worker_addresses)])

        """
        Invoke knn_classify on Dask workers to perform distributed query
        """
        key = uuid1()
        knn_reg_res = dict([(worker_info[worker]["rank"], self.client.submit(
                            self._func_predict,
                            models[worker],
                            self.data_handler.worker_to_parts[worker] if
                            worker in self.data_handler.workers else [],
                            data_parts_to_ranks,
                            data_nrows,
                            query_handler.worker_to_parts[worker] if
                            worker in query_handler.workers else [],
                            query_parts_to_ranks,
                            query_nrows,
                            X.shape[1],
                            self.n_outputs,
                            worker_info[worker]["rank"],
                            convert_dtype,
                            key="%s-%s" % (key, idx),
                            workers=[worker]))
                           for idx, worker in enumerate(comms.worker_addresses)
                            ])

        wait_and_raise_from_futures(list(knn_reg_res.values()))

        """
        Gather resulting partitions and return result
        """
        out_futures = flatten_grouped_results(self.client,
                                              query_parts_to_ranks,
                                              knn_reg_res)

        comms.destroy()

        return to_output(out_futures, self.datatype).squeeze()

    def score(self, X, y):
        """
        Provide score by comparing predictions and ground truth.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Query test data.
            Acceptable formats: dask CuPy/NumPy/Numba Array

        y : array-like (device or host) shape = (n_samples, n_features)
            Outputs test data.
            Acceptable formats: dask CuPy/NumPy/Numba Array

        Returns
        -------
        score
        """
        y_pred = self.predict(X, convert_dtype=True)
        if not isinstance(y_pred, da.Array):
            y_pred = y_pred.to_dask_array(lengths=True)
        if not isinstance(y, da.Array):
            y = y.to_dask_array(lengths=True)
        y_true = y.squeeze()
        y_mean = y_true.mean(axis=0)
        residual_sss = ((y_true - y_pred) ** 2).sum(axis=0, dtype='float64')
        total_sss = ((y_true - y_mean) ** 2).sum(axis=0, dtype='float64')
        r2_score = da.mean(1 - (residual_sss / total_sss))
        return r2_score.compute()
