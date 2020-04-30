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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cuml.dask.common.input_utils import _extract_partitions
from cuml.dask.common import workers_to_parts, parts_to_ranks, \
    raise_exception_from_futures, flatten_grouped_results
from cuml.dask.common.comms import CommsContext, worker_state
from dask.distributed import default_client
from dask.distributed import wait
import dask.array as da
from uuid import uuid1
import numpy as np

from cuml.common.array import CumlArray
from cuml.common.handle cimport cumlHandle

import rmm
from libc.stdlib cimport calloc, malloc, free
from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t, int64_t
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr

cdef extern from "cumlprims/opg/matrix/data.hpp" namespace \
        "MLCommon::Matrix":

    cdef cppclass Data[T]:
        Data(T *ptr, size_t totalSize)

ctypedef Data[float] floatData_t
ctypedef Data[int64_t] int64Data_t
ctypedef Data[int] intData_t
ctypedef vector[int*] int_ptr_vector

cdef extern from "cumlprims/opg/matrix/part_descriptor.hpp" namespace \
        "MLCommon::Matrix":

    cdef cppclass RankSizePair:
        int rank
        size_t size

    cdef cppclass PartDescriptor:
        PartDescriptor(size_t M,
                       size_t N,
                       vector[RankSizePair*] &partsToRanks,
                       int myrank)

cdef extern from "cumlprims/opg/selection/knn.hpp" namespace \
        "MLCommon::Selection::opg":

    cdef void knn_classify(
        cumlHandle &handle,
        vector[intData_t*] &out,
        vector[int64Data_t*] &out_I,
        vector[floatData_t*] &out_D,
        vector[floatData_t*] &idx_data,
        PartDescriptor &idx_desc,
        vector[floatData_t*] &query_data,
        PartDescriptor &query_desc,
        vector[int_ptr_vector] &y,
        vector[int*] &uniq_labels,
        vector[int] &n_unique,
        bool rowMajorIndex,
        bool rowMajorQuery,
        int k,
        size_t batch_size,
        bool verbose
    ) except +


def _func_knn_classify(sessionID, idx_local_parts, idx_desc, idx_parts_to_ranks,
                        q_local_parts, q_desc, query_parts_to_ranks,
                        labels_local_parts, labels_desc, labels_parts_to_ranks,
                        y, uniq_labels, n_unique,
                        n_neighbors, rank, batch_size, verbose):

    handle = worker_state(sessionID)["handle"]
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    cdef vector[intData_t*] *out_vec \
        = new vector[intData_t*]()
    cdef vector[int64Data_t*] *out_i_vec \
        = new vector[int64Data_t*]()
    cdef vector[floatData_t*] *out_d_vec \
        = new vector[floatData_t*]()

    output = []
    output_i = []
    output_d = []

    for query_part in q_desc:
        n_rows = query_part["shape"][0]
        o_ary = CumlArray.zeros(shape=(n_rows,),
                                order="C", dtype=np.int32)
        i_ary = CumlArray.zeros(shape=(n_rows, n_neighbors),
                                order="C", dtype=np.int64)
        d_ary = CumlArray.zeros(shape=(n_rows, n_neighbors),
                                order="C", dtype=np.float32)

        output.append(o_ary)
        output_i.append(i_ary)
        output_d.append(d_ary)

        out_vec.push_back(new intData_t(
            <int*><uintptr_t>o_ary.ptr, n_rows * n_neighbors))

        out_i_vec.push_back(new int64Data_t(
            <int64_t*><uintptr_t>i_ary.ptr, n_rows * n_neighbors))

        out_d_vec.push_back(new floatData_t(
            <float*><uintptr_t>d_ary.ptr, n_rows * n_neighbors))

    knn_classify(
        handle_[0],
        deref(out_vec),
        deref(out_i_vec),
        deref(out_d_vec),
        deref(<vector[floatData_t*]*><uintptr_t>idx_local_parts),
        deref(<PartDescriptor*><uintptr_t>idx_desc),
        deref(<vector[floatData_t*]*><uintptr_t>q_local_parts),
        deref(<PartDescriptor*><uintptr_t>q_desc),
        deref(<vector[int_ptr_vector]*><uintptr_t> y),
        deref(<vector[int*]*><uintptr_t> uniq_labels),
        deref(<vector[int]*><uintptr_t> n_unique),
        False,  # column-major index
        False,  # column-major query
        <int>n_neighbors,
        <size_t>batch_size,
        <bool>verbose
    )

    handle.sync()

    """
    _free_mem(<size_t>idx_rsp,
                <size_t>idx_desc,
                <size_t>q_rsp,
                <size_t>q_desc,
                <size_t>out_i_vec,
                <size_t>out_d_vec,
                <size_t>idx_local_parts,
                <size_t>q_local_parts)
    """

    return output, output_i, output_d


class KNeighborsClassifier():
    def __init__(self, client=None, streams_per_handle=0, verbose=False,
                 n_neighbors=5, batch_size=1024):
        self.client = default_client() if client is None else client
        self.streams_per_handle = streams_per_handle
        self.verbose = verbose
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size

    @staticmethod
    def _build_comms(index_wtp, label_wtp, query_wtp,
                     streams_per_handle, verbose):
        # Communicator clique needs to include the union of workers hosting
        # query and index partitions
        workers = set(index_wtp.keys())
        workers.update(label_wtp.keys())
        workers.update(query_wtp.keys())

        comms = CommsContext(comms_p2p=True,
                             streams_per_handle=streams_per_handle,
                             verbose=verbose)
        comms.init(workers=workers)
        return comms

    def fit(self, X, y, convert_dtype=True):
        self.X = self.client.sync(_extract_partitions, X)
        self.y = self.client.sync(_extract_partitions, y)

        uniq_labels = []
        if y.ndim == 1:
            uniq_labels.append(da.unique(y))
        else:
            n_targets = y.shape[1]
            for i in range(n_targets):
                uniq_labels.append(da.unique(y[:,i]))
        self.uniq_labels = da.compute(uniq_labels)[0]
        self.n_unique = list(map(lambda x: len(x), self.uniq_labels))
        print(self.uniq_labels)
        print(self.n_unique)
        return self

    def predict(self, X, convert_dtype=True):
        index_futures = self.X
        label_futures = self.y
        query_futures = self.client.sync(_extract_partitions, X)

        index_worker_to_parts = workers_to_parts(index_futures)
        labels_worker_to_parts = workers_to_parts(label_futures)
        query_worker_to_parts = workers_to_parts(query_futures)

        comms = KNeighborsClassifier._build_comms(index_worker_to_parts,
                                                  labels_worker_to_parts,
                                                  query_worker_to_parts,
                                                  self.streams_per_handle,
                                                  self.verbose)

        worker_info = comms.worker_info(comms.worker_addresses)

        """
        Build inputs and outputs
        """
        idx_parts_to_ranks, idx_M = parts_to_ranks(self.client,
                                                   worker_info,
                                                   index_futures)

        labels_parts_to_ranks, labels_M = parts_to_ranks(self.client,
                                                         worker_info,
                                                         label_futures)

        query_parts_to_ranks, query_M = parts_to_ranks(self.client,
                                                       worker_info,
                                                       query_futures)

        """
        Invoke kneighbors on Dask workers to perform distributed query
        """

        key = uuid1()
        nn_fit = dict([(worker_info[worker]["rank"], self.client.submit(
                        _func_knn_classify,
                        comms.sessionId,
                        index_worker_to_parts[worker] if
                        worker in index_worker_to_parts else [],
                        idx_M,
                        idx_parts_to_ranks,
                        query_worker_to_parts[worker] if
                        worker in query_worker_to_parts else [],
                        query_M,
                        query_parts_to_ranks,
                        labels_worker_to_parts[worker] if
                        worker in labels_worker_to_parts else [],
                        labels_M,
                        labels_parts_to_ranks,
                        self.uniq_labels,
                        self.n_unique,
                        self.n_neighbors,
                        worker_info[worker]["rank"],
                        self.batch_size,
                        self.verbose,
                        key="%s-%s" % (key, idx),
                        workers=[worker]))
                       for idx, worker in enumerate(comms.worker_addresses)])

        wait(list(nn_fit.values()))
        raise_exception_from_futures(list(nn_fit.values()))

        comms.destroy()

        """
        Gather resulting partitions and return dask_cudfs
        """
        out_futures = flatten_grouped_results(self.client,
                                              query_parts_to_ranks,
                                              nn_fit,
                                              getter_func=lambda f,
                                              idx: f[0][idx])

        out_d_futures = flatten_grouped_results(self.client,
                                                query_parts_to_ranks,
                                                nn_fit,
                                                getter_func=lambda f,
                                                idx: f[1][idx])

        out_i_futures = flatten_grouped_results(self.client,
                                                query_parts_to_ranks,
                                                nn_fit,
                                                getter_func=lambda f,
                                                idx: f[2][idx])

        return nn_fit, out_futures, out_d_futures, out_i_futures
