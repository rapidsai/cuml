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

from cuml.dask.common import extract_ddf_partitions, to_dask_cudf
from dask.distributed import default_client
from cuml.dask.common.comms import worker_state, CommsContext
from dask.distributed import wait
import numpy as np

from collections import OrderedDict

from functools import reduce

from uuid import uuid1

import cudf


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
        if self.n_neighbors is not None and k is None:
            k = self.n_neighbors

        index_futures = self.client.sync(extract_ddf_partitions, self.X, agg=False)
        query_futures = self.client.sync(extract_ddf_partitions, X, agg=False)

        index_worker_to_parts = OrderedDict()
        for w, p in index_futures:
            if w not in index_worker_to_parts:
                index_worker_to_parts[w] = []
                index_worker_to_parts[w].append(p)

        query_worker_to_parts = OrderedDict()
        for w, p in query_futures:
            if w not in query_worker_to_parts:
                query_worker_to_parts[w] = []
                query_worker_to_parts[w].append(p)

        workers = set(map(lambda x: x[0], index_futures))
        workers.extend(list(map(lambda x: x[0], query_futures)))

        comms = CommsContext(comms_p2p=False)
        comms.init(workers=workers)

        worker_info = comms.worker_info(comms.worker_addresses)

        key = uuid1()
        idx_partsToRanks = [(worker_info[wf[0]]["r"], self.client.submit(
            NearestNeighbors._func_get_size,
            wf[1],
            workers=[wf[0]],
            key="%s-%s" % (key, idx)).result())
            for idx, wf in enumerate(index_worker_to_parts)]

        query_partsToRanks = [(worker_info[wf[0]]["r"], self.client.submit(
            NearestNeighbors._func_get_size,
            wf[1],
            workers=[wf[0]],
            key="%s-%s" % (key, idx)).result())
            for idx, wf in enumerate(query_futures)]

        N = X.shape[1]
        idx_M = reduce(lambda a,b: a+b, map(lambda x: x[1], idx_partsToRanks))
        query_M = reduce(lambda a,b: a+b, map(lambda x: x[1], query_partsToRanks))

        key = uuid1()
        nn_models = [(worker, self.client.submit(
            NearestNeighbors._func_create_model,
            comms.sessionId,
            **self.kwargs,
            workers=[worker],
            key="%s-%s" % (key, idx)))
            for idx, worker in enumerate(workers)]

        print(str(nn_models))

        key = uuid1()
        nn_fit = [self.client.submit(
            NearestNeighbors._func_kneighbors,
            index_worker_to_parts[worker] if worker in index_worker_to_parts else [],
            idx_M,
            N,
            idx_partsToRanks,
            query_worker_to_parts[worker] if worker in query_worker_to_parts else [],
            query_M,
            query_partsToRanks,
            worker_info[worker]["r"],
            key="%s-%s" % (key, idx),
            workers=[worker])
            for idx, worker in enumerate(workers)]

        wait(nn_fit)

        print(str([f.exception() for f in nn_fit]))

        comms.destroy()

        self.local_model = self.client.submit(NearestNeighbors._func_get_first,
                                              nn_models[0][1]).result()

        # TODO: Need to build output DFs for `output_i` and `output_d`
