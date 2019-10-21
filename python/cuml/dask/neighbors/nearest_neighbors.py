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

from cuml.dask.common import extract_ddf_partitions, to_dask_cudf, \
    raise_exception_from_futures
from dask.distributed import default_client
from cuml.dask.common.comms import worker_state, CommsContext
from dask.distributed import wait

from collections import OrderedDict

from functools import reduce

from uuid import uuid1


class NearestNeighbors(object):
    """
    Multi-node Multi-GPU NearestNeighbors Model.
    """
    def __init__(self, client=None, **kwargs):
        self.client = default_client() if client is None else client
        self.kwargs = kwargs

    def fit(self, X):
        """
        Fit a multi-node multi-GPU Nearest Neighbors index
        :param X : dask_cudf.Dataframe
        :return : NearestNeighbors model
        """
        self.X = X

        return self

    @staticmethod
    def _func_get_size(df):
        return df.shape[0]

    @staticmethod
    def _func_create_model(sessionId, **kwargs):
        try:
            from cuml.neighbors.nearest_neighbors_mg import \
                NearestNeighborsMG as cumlNN
        except ImportError:
            raise Exception("cuML has not been built with multiGPU support "
                            "enabled. Build with the --multigpu flag to"
                            " enable multiGPU support.")

        handle = worker_state(sessionId)["handle"]
        return cumlNN(handle=handle, **kwargs)

    @staticmethod
    def _func_kneighbors(model, local_idx_parts, idx_m, n, idx_partsToRanks,
                         local_query_parts, query_m, query_partsToRanks,
                         rank, k):
        return model.kneighbors(
            local_idx_parts, idx_m, n, idx_partsToRanks,
            local_query_parts, query_m, query_partsToRanks,
            rank, k
        )

    @staticmethod
    def _func_get_d(f, idx):
        print("f=" + str(f))
        i, d = f
        return d[idx]

    @staticmethod
    def _func_get_i(f, idx):
        print("f=" + str(f))
        i, d = f
        return i[idx]

    def kneighbors(self, X, k=None):
        """
        Query the NearestNeighbors index
        :param X : dask_cudf.Dataframe list of vectors to query
        :param k : Number of neighbors to query for each row in X
        :param replicate : bool, string, or int. If X is small enough, it
                can be replicated onto the workers containing the indices.
                If the indices are replicated, this means only a single
                worker needs to perform the predict, otherwise, the results
                of the query are able to be reduced to a single partition.
                Setting replicate to True or False explicitly turns it
                on or off. Setting it to a string, specifies the threshold,
                using a format like "2GB", for determining whether X should
                be replicated. If this value is an int, the number of
                elements is used as a threshold.
        :return : dask_cudf.Dataframe containing the results
        """
        if self.kwargs["n_neighbors"] is not None and k is None:
            k = self.kwargs["n_neighbors"]

        index_futures = self.client.sync(extract_ddf_partitions, self.X, agg=False)
        query_futures = self.client.sync(extract_ddf_partitions, X, agg=False)

        index_worker_to_parts = OrderedDict()
        for w, p in index_futures:
            if w not in index_worker_to_parts:
                index_worker_to_parts[w] = []
                index_worker_to_parts[w].append(p)

        print(str(index_worker_to_parts))

        query_worker_to_parts = OrderedDict()
        for w, p in query_futures:
            if w not in query_worker_to_parts:
                query_worker_to_parts[w] = []
                query_worker_to_parts[w].append(p)

        workers = set(map(lambda x: x[0], index_futures))
        workers.update(list(map(lambda x: x[0], query_futures)))

        comms = CommsContext(comms_p2p=True)
        comms.init(workers=workers)

        worker_info = comms.worker_info(comms.worker_addresses)

        """
        Build inputs and outputs
        """
        key = uuid1()
        idx_partsToRanks = [(worker_info[wf[0]]["r"], self.client.submit(
            NearestNeighbors._func_get_size,
            wf[1],
            workers=[wf[0]],
            key="%s-%s" % (key, idx)).result())
            for idx, wf in enumerate(index_futures)]

        query_partsToRanks = [(worker_info[wf[0]]["r"], self.client.submit(
            NearestNeighbors._func_get_size,
            wf[1],
            workers=[wf[0]],
            key="%s-%s" % (key, idx)).result())
            for idx, wf in enumerate(query_futures)]

        N = X.shape[1]
        idx_M = reduce(lambda a,b: a+b, map(lambda x: x[1], idx_partsToRanks))
        query_M = reduce(lambda a,b: a+b, map(lambda x: x[1], query_partsToRanks))

        """
        Each Dask worker creates a single model
        """
        key = uuid1()
        nn_models = dict([(worker, self.client.submit(
            NearestNeighbors._func_create_model,
            comms.sessionId,
            **self.kwargs,
            workers=[worker],
            key="%s-%s" % (key, idx)))
            for idx, worker in enumerate(workers)])

        wait(nn_models.values())

        raise_exception_from_futures(nn_models.values())

        """
        Invoke kneighbors on Dask workers to perform distributed query
        """
        key = uuid1()
        nn_fit = dict([(worker_info[worker]["r"], self.client.submit(
            NearestNeighbors._func_kneighbors,
            nn_models[worker],
            index_worker_to_parts[worker] if worker in
                                             index_worker_to_parts else [],
            idx_M,
            N,
            idx_partsToRanks,
            query_worker_to_parts[worker] if worker in
                                             query_worker_to_parts else [],
            query_M,
            query_partsToRanks,
            worker_info[worker]["r"],
            k,
            key="%s-%s" % (key, idx),
            workers=[worker]))
            for idx, worker in enumerate(workers)])

        wait(list(nn_fit.values()))
        raise_exception_from_futures(nn_fit.values())
        comms.destroy()


        """
        Gather resulting partitions and return dask_cudfs
        """
        out_i_futures = []
        out_d_futures = []
        completed_part_map = {}
        for rank, size in query_partsToRanks:
            if rank not in completed_part_map:
                completed_part_map[rank] = 0

            f = nn_fit[rank]

            out_d_futures.append(self.client.submit(
                NearestNeighbors._func_get_d, f, completed_part_map[rank]))

            out_i_futures.append(self.client.submit(
                NearestNeighbors._func_get_i, f, completed_part_map[rank]))

            completed_part_map[rank] += 1

        return to_dask_cudf(out_d_futures), \
               to_dask_cudf(out_i_futures)
