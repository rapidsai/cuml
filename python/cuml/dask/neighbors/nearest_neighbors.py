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

from cuml.dask.common.spmg_ipc import new_ipc_thread
from cuml.dask.common.utils import parse_host_port
from cuml.dask.common.utils import device_of_devicendarray, build_host_dict

from cuml.neighbors import NearestNeighbors as cumlKNN

import logging

import random

from cuml.utils import numba_utils


from dask import delayed
from collections import defaultdict
from dask.distributed import wait, default_client
import dask.dataframe as dd

from tornado import gen
import dask_cudf
import cudf

import numpy as np

from toolz import first

import numba.cuda


def to_gpu_matrix(df):
    """
    Turn input cudf into a Numba array. Returns device
    of current worker and the start/stop indexes
    of the cudf.
    :param df:
    :return:
    """

    try:
        start_idx = df.index[0]
        stop_idx = df.index[-1]
        gpu_matrix = numba_utils.row_matrix(df)
        dev = device_of_devicendarray(gpu_matrix)
        return dev, gpu_matrix, (start_idx, stop_idx)

    except Exception as e:
        import traceback
        logging.error("Error in to_gpu_matrix(" + str(e))
        traceback.print_exc()
        pass


def build_alloc_info(data):
    """
    Use the __cuda_array_interface__ to extract cpointer
    information for passing into cython.
    :param data:
    :return:
    """
    dev, gpu_matrix, _ = data
    return gpu_matrix.__cuda_array_interface__


def get_ipc_handle(data):
    """
    Extract IPC handles from input Numba array. Pass
    along the device of the current worker and the
    start/stop indices from the original cudf.
    :param data:
    :return:
    """
    dev, gpu_matrix, idx = data

    try:
        in_handle = gpu_matrix.get_ipc_handle()
        return dev, in_handle, idx
    except Exception as e:
        import traceback
        logging.error("Error in get_ipc_handles(dev=" +
                      str(dev) + "): " + str(e))
        traceback.print_exc()
        pass


# Run on a single worker on each unique host
def _fit_on_worker(data, params):
    """
    For SPMG kNN, this is the function that runs on a
    chosen worker, collects all the IPC handles from
    the other workers, and makes the cython call to
    the cuML kNN multiGPU fit() function.
    :param data:
    :param params:
    :return:
    """

    ipcs, raw_arrs = data

    # Separate threads to hold pointers to separate devices
    # The order in which we pass the list of IPCs to the thread matters and
    # the goal isto maximize reuse while minimizing the number of threads.
    # We want to limit the number of threads to O(len(devices)) and want to
    # avoid having if be O(len(ipcs)) at all costs!
    device_handle_map = defaultdict(list)
    [device_handle_map[dev].append((idx, ipc)) for dev, ipc, idx in ipcs]

    open_ipcs = [([i[0] for i in ipcs],
                  new_ipc_thread([i[1] for i in ipcs], dev))
                 for dev, ipcs in device_handle_map.items()]

    alloc_info = []
    for idxs, t in open_ipcs:
        inf = t.info()
        for i in range(len(idxs)):
            alloc_info.append([idxs[i], inf[i]])

    alloc_info.extend([[t[2], build_alloc_info(t)] for t in raw_arrs])
    alloc_info.sort(key=lambda x: x[0][0])

    final_allocs = [a for i, a in alloc_info]

    m = cumlKNN(should_downcast=params["should_downcast"])
    m._fit_mg(params["D"], final_allocs)
    #
    # [t[1].close() for t in open_ipcs]
    # [t[1].join() for t in open_ipcs]

    return m


def _kneighbors_on_worker(data, m, params):
    """
    For SPMG kNN, this is the function that runs on
    a chosen worker, collects all the IPC handles from
    the other workers, and makes the cython call to
    the cuML kNN multiGPU kneighbors() function.
    :param data:
    :param m:
    :param params:
    :return:
    """
    ipc_dev_list, devarrs_dev_list = data

    device_handle_map = defaultdict(list)

    ipc_dev_list = list(filter(None, ipc_dev_list))
    devarrs_dev_list = list(filter(None, devarrs_dev_list))

    # Each ipc contains X, I, D handles
    [device_handle_map[dev].append((idx, ipc))
        for ipc, dev, idx in ipc_dev_list]

    def collect_ipcs(ipcs):
        """
        A simple helper function to flat map a deeply
        nested list of ipc handles.
        :param ipcs:
        :return:
        """
        final = []
        for ipc in ipcs:
            for i in ipc:
                for j in i:
                    final.append(j)

        return final

    open_ipcs = [([i[0] for i in idx_ipcs],
                  new_ipc_thread(collect_ipcs([i[1] for i in idx_ipcs]), dev))
                 for dev, idx_ipcs in device_handle_map.items()]

    alloc_info = []
    for idxs, t in open_ipcs:
        inf = t.info()
        for i in range(len(idxs)):
            alloc_info.append((idxs[i], inf[i*3:(i*3)+3]))

    for p, dev, idx in devarrs_dev_list:
        for X, inds, dists in p:
            alloc_info.extend([(idx, [build_alloc_info((dev, X, idx)),
                                      build_alloc_info((dev, inds, idx)),
                                      build_alloc_info((dev, dists, idx))])])

    alloc_info.sort(key=lambda x: x[0][0])

    for idx, allocs in alloc_info:
        X, inds, dists = allocs
        m._kneighbors(X["data"][0], X["shape"][0],
                      inds["data"][0], dists["data"][0], params["k"])

    [t.close() for idx, t in open_ipcs]
    [t.join() for idx, t in open_ipcs]

    return data


def input_to_device_arrays(X, params):
    """
    Create output arrays and return them w/ the input array(s)
    :param arr:
        A tuple in the form of (X, y)
    :return:
    """

    if len(X[0]) == 0:
        return None

    start_idx = X[0].index[0]
    stop_idx = X[0].index[-1]

    X_mat = numba_utils.row_matrix(X[0])
    dev = device_of_devicendarray(X_mat)

    shape = X_mat.shape[0]*params["k"]

    # Create output numba arrays.
    I_ndarr = numba.cuda.to_device(np.zeros(shape, dtype=np.int64, order="C"))
    D_ndarr = numba.cuda.to_device(np.zeros(shape, dtype=np.float32,
                                            order="C"))

    # Return canonical device id as string
    return [(X_mat, I_ndarr, D_ndarr)], dev, (start_idx, stop_idx)


def get_input_ipc_handles(arr):
    """
    Used for kneighbors() to extract the IPC handles from
    the input Numba arrays. The device of the current worker
    and the start/stop indices of the original cudf are
    passed along as well.
    :param arr:
    :return:
    """
    if arr is None:
        return None

    arrs, dev, idx = arr
    mat = [(X.get_ipc_handle(), inds.get_ipc_handle(), dists.get_ipc_handle())
           for X, inds, dists in arrs]
    return mat, dev, idx


def build_dask_dfs(arrs, params):
    """
    Convert Numba arrays for kneighbors() resulting
    indices and distances into cudf Dataframes, using
    the start/stop indices from dataframe X.
    :param arrs:
    :param params:
    :return:
    """

    with numba.cuda.gpus[0]:
        if arrs is None:
            return None

        arr, dev, idx = arrs
        X, I_ndarr, D_ndarr = arr[0]

        I_ndarr = I_ndarr.reshape((X.shape[0], params["k"]))
        D_ndarr = D_ndarr.reshape((X.shape[0], params["k"]))

        Ix = cudf.DataFrame(index=cudf.dataframe.RangeIndex(idx[0], idx[1]+1))
        D = cudf.DataFrame(index=cudf.dataframe.RangeIndex(idx[0], idx[1]+1))

        for i in range(0, params["k"]):
            Ix[str(i)] = I_ndarr[:, i]

        for i in range(0, params["k"]):
            D[str(i)] = D_ndarr[:, i]

        return Ix, D, idx


def get_idx(arrs):
    """
    Extract and return the start/stop
    indices from original cudf.
    :param arrs:
    :return:
    """
    return arrs[2]


def get_I(arrs):
    """
    Extract and return the indices cudf
    :param arrs:
    :return:
    """
    return arrs[0]


def get_D(arrs):
    """
    Extract and return the dists cudf
    :param arrs:
    :return:
    """
    return arrs[1]


def get_I_meta(arrs):
    """
    Extract and return the metadata
    from the indices cudf.
    :param arrs:
    :return:
    """
    return arrs[0].iloc[:0]


def get_D_meta(arrs):
    """
    Extract and return the metadata
    from the dists cudf.
    :param arrs:
    :return:
    """
    return arrs[1].iloc[:0]


class NearestNeighbors(object):
    """
    Model-parallel Multi-GPU kNN Model.

    Data is spread across Dask workers using Dask cuDF. A single worker is
    chosen to create a series of kNN indices, one for each chunk of the Dask
    input, across the devices on that host. Results will reflect the global
    order, extracted from the Dask cuDF used for training.
    """
    def __init__(self, n_neighbors=5, should_downcast=False):
        self.model = None
        self.master_host = None
        self.should_downcast = should_downcast
        self.n_neighbors = n_neighbors
        self.gpu_futures = None

    def fit(self, ddf):
        """
        Fits a single-node multi-gpu knn model using single process-multiGPU
        technique.
        :param futures:
        :return:
        """

        client = default_client()

        # Keep the futures around so the GPU memory doesn't get
        # deallocated on the workers.
        gpu_futures, cols = client.sync(self._get_mg_info, ddf)
        self.gpu_futures = gpu_futures

        host_dict = self._build_host_dict(gpu_futures, client).items()
        if len(host_dict) > 1:
            raise Exception("Dask cluster appears to span hosts. Current "
                            "multi-GPU version is limited to single host")

        # Choose a random worker on each unique host to run dask-cuml's
        # kNN.fit() function on all the cuDFs living on that host.
        self.master_host = [(host, random.sample(ports, 1)[0])
                            for host, ports in host_dict][0]

        host, port = self.master_host

        gpu_futures_for_host = list(filter(lambda d: d[0][0] == host,
                                           gpu_futures))
        exec_node = (host, port)

        # build ipc handles
        gpu_data_excl_worker = list(filter(lambda d: d[0] != exec_node,
                                           gpu_futures_for_host))
        gpu_data_incl_worker = list(filter(lambda d: d[0] == exec_node,
                                           gpu_futures_for_host))

        ipc_handles = [client.submit(get_ipc_handle, future, workers=[worker])
                       for worker, future in gpu_data_excl_worker]

        raw_arrays = [future for worker, future in gpu_data_incl_worker]

        f = (exec_node, client.submit(_fit_on_worker, (ipc_handles,
                                                       raw_arrays),
                                      {"D": cols,
                                      "should_downcast": self.should_downcast},
                                      workers=[exec_node]))

        wait(f)

        # The model on each unique host is held for futures queries
        self.model = f

    @gen.coroutine
    def _kneighbors(self, X, k):
        """
        Internal function to query the kNN model.
        :param X:
        :param k:
        :return:
        """
        client = default_client()
        if k is None:
            k = self.n_neighbors

        # Break apart Dask.array/dataframe into chunks/parts
        data_parts = X.to_delayed()

        parts = list(map(delayed, data_parts))
        parts = client.compute(parts)  # Start computation in the background
        yield wait(parts)
        for part in parts:
            if part.status == 'error':
                yield part  # trigger error locally

        # A dict in the form of { part_key: part }
        key_to_part_dict = dict([(str(part.key), part) for part in parts])

        who_has = yield client.who_has(parts)

        worker_parts = {}
        for key, workers in who_has.items():
            worker = parse_host_port(first(workers))
            if worker not in worker_parts:
                worker_parts[worker] = []
            worker_parts[worker].append(key_to_part_dict[key])

        """
        Create IP Handles on each worker hosting input data
        """
        # Format of input_devarrays = ([(X, y)..], dev)
        input_devarrays = [(worker, client.submit(input_to_device_arrays, part,
                                                  {"k": k}, workers=[worker]))
                           for worker, part in worker_parts.items()]

        yield wait(input_devarrays)

        """
        Gather IPC handles for each worker and call _fit() on each worker
        containing data.
        """
        exec_node, model = self.model

        # Need to fetch coefficient parts on worker
        on_worker = list(filter(lambda x: x[0] == exec_node, input_devarrays))
        not_on_worker = list(filter(lambda x: x[0] != exec_node,
                                    input_devarrays))

        ipc_handles = [client.submit(get_input_ipc_handles, future,
                                     workers=[a_worker])
                       for a_worker, future in not_on_worker]

        raw_arrays = [future for a_worker, future in on_worker]

        # IPC Handles are loaded in separate threads on worker so they can be
        # used to make calls through cython

        run = client.submit(_kneighbors_on_worker, (ipc_handles, raw_arrays),
                            model, {"k": k}, workers=[exec_node])
        yield wait(run)

        dfs = [client.submit(build_dask_dfs, f, {"k": k}, workers=[worker])
               for worker, f in input_devarrays]
        yield wait(dfs)

        return gen.Return(dfs)

    def kneighbors(self, X, k=None):

        """
        Queries the multi-gpu knn model given a dask-cudf as the query

        1. Create 2 new Dask dataframes to hold output
        (1 chunk each per chunk of X), co-locate pieces w/ X.
        2. Get IPC handles for each dataframe. Use IPCThread to hold onto
        them while calling query.

        :param input:
            A dask-cudf for calculating the kneighbors
        :param k:
            The number of nearest neighbors to query for each input vector.
        :return:
            dists and indices of the k-nearest neighbors to the input vectors
        """

        if k is None:
            k = self.n_neighbors

        client = default_client()
        dfs = client.sync(self._kneighbors, X, k).value

        dfs = [d for d in dfs if d.type != type(None)]  # NOQA

        local_divs = [client.submit(get_idx, f).result() for f in dfs]
        indices = [client.submit(get_I, f) for f in dfs]
        dists = [client.submit(get_D, f) for f in dfs]

        dfs_divs = list(zip(local_divs, indices, dists))

        # Sort delayed dfs by their starting index
        dfs_divs.sort(key=lambda x: x[0][0])

        I_meta = client.submit(get_I_meta, dfs[0]).result()
        D_meta = client.submit(get_D_meta, dfs[0]).result()

        I_ddf = dask_cudf.from_delayed(indices, meta=I_meta)
        D_ddf = dask_cudf.from_delayed(dists, meta=D_meta)

        return D_ddf, I_ddf

    @staticmethod
    def _build_host_dict(gpu_futures, client):
        """
        Helper function to build a dictionary mapping workers to parts
        that currently hold the parts of given futures.
        :param gpu_futures:
        :param client:
        :return:
        """
        who_has = client.who_has(gpu_futures)

        key_to_host_dict = {}
        for key in who_has:
            key_to_host_dict[key] = parse_host_port(who_has[key][0])

        hosts_to_key_dict = {}
        for key, host in key_to_host_dict.items():
            if host not in hosts_to_key_dict:
                hosts_to_key_dict[host] = set([key])
            else:
                hosts_to_key_dict[host].add(key)

        workers = [key[0] for key in list(who_has.values())]
        return build_host_dict(workers)

    @gen.coroutine
    def _get_mg_info(self, ddf):
        """
        Given a Dask cuDF, extract number of dimensions and convert
        the pieces of the Dask cuDF into Numba arrays, which can
        be passed into the kNN algorithm.
        build a
        :param ddf:
        :return:
        """

        client = default_client()

        if isinstance(ddf, dd.DataFrame):
            cols = len(ddf.columns)
            parts = ddf.to_delayed()
            parts = client.compute(parts)
            yield wait(parts)
        else:
            raise Exception("Input should be a Dask DataFrame")

        key_to_part_dict = dict([(str(part.key), part) for part in parts])
        who_has = yield client.who_has(parts)

        worker_map = []
        for key, workers in who_has.items():
            worker = parse_host_port(first(workers))
            worker_map.append((worker, key_to_part_dict[key]))

        gpu_data = [(worker, client.submit(to_gpu_matrix, part,
                                           workers=[worker]))
                    for worker, part in worker_map]

        yield wait(gpu_data)

        raise gen.Return((gpu_data, cols))
