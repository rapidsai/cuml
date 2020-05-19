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

from cuml.dask.common.input_utils import DistributedDataHandler, \
                                         to_output, \
                                         to_dask_cupy
from cuml.dask.common import workers_to_parts, \
                             parts_to_ranks, \
                             raise_exception_from_futures, \
                             flatten_grouped_results
from cuml.dask.common.comms import CommsContext, worker_state
from dask.distributed import default_client
from dask.distributed import wait
import dask.array as da
from uuid import uuid1
import numpy as np

from cuml.common.array import CumlArray
from cuml.common.handle cimport cumlHandle
from cuml.common import input_to_cuml_array

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

    cdef cppclass floatData_t:
        floatData_t(float *ptr, size_t totalSize)
        float *ptr
        size_t totalSize

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


def _func_get_o(f, idx):
    o, i, d = f
    return o[idx]


def _func_get_i(f, idx):
    o, i, d = f
    return i[idx]


def _func_get_d(f, idx):
    o, i, d = f
    return d[idx]


def _build_part_inputs(cuda_arr_ifaces,
                       parts_to_ranks,
                       m, n, local_rank,
                       convert_dtype):

    cuml_arr_ifaces = []
    for arr in cuda_arr_ifaces:
        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(arr, order="F",
                                convert_to_dtype=(np.float32
                                                  if convert_dtype
                                                  else None),
                                check_dtype=[np.float32])
        cuml_arr_ifaces.append(X_m)

    cdef vector[floatData_t*] *local_parts = new vector[floatData_t*]()
    for arr in cuml_arr_ifaces:
        data = <floatData_t*>malloc(sizeof(floatData_t))
        data.ptr = <float*><uintptr_t>arr.ptr
        data.totalSize = <size_t>arr.shape[0]*arr.shape[1]*sizeof(float)
        local_parts.push_back(data)

    cdef vector[RankSizePair*] partsToRanks
    for idx, rankToSize in enumerate(parts_to_ranks):
        rank, size = rankToSize
        rsp = <RankSizePair*>malloc(sizeof(RankSizePair))
        rsp.rank = <int>rank
        rsp.size = <size_t>size
        partsToRanks.push_back(rsp)

    cdef PartDescriptor *descriptor = \
        new PartDescriptor(<size_t>m,
                           <size_t>n,
                           <vector[RankSizePair*]>partsToRanks,
                           <int>local_rank)

    return cuml_arr_ifaces, <uintptr_t>local_parts, <uintptr_t>descriptor


def _free_mem(out_vec, out_i_vec, out_d_vec,
              idx_local_parts, idx_desc,
              q_local_parts, q_desc,
              lbls_local_parts,
              uniq_labels, n_unique):

    free(<void*><uintptr_t>out_vec)
    free(<void*><uintptr_t>out_i_vec)
    free(<void*><uintptr_t>out_d_vec)

    cdef floatData_t* ptr

    cdef vector[floatData_t*] *idx_local_parts_v = \
        <vector[floatData_t *]*><uintptr_t>idx_local_parts
    for i in range(idx_local_parts_v.size()):
        ptr = idx_local_parts_v.at(i)
        free(<void*>ptr)
    free(<void*><uintptr_t>idx_local_parts)

    cdef vector[floatData_t*] *q_local_parts_v = \
        <vector[floatData_t *]*><uintptr_t>q_local_parts
    for i in range(q_local_parts_v.size()):
        ptr = q_local_parts_v.at(i)
        free(<void*>ptr)
    free(<void*><uintptr_t>q_local_parts)

    free(<void*><uintptr_t>idx_desc)
    free(<void*><uintptr_t>q_desc)

    free(<void*><uintptr_t>lbls_local_parts)

    free(<void*><uintptr_t>uniq_labels)
    free(<void*><uintptr_t>n_unique)


def _func_knn_classify(sessionID,
                       data, data_parts_to_ranks, data_nrows,
                       query, query_parts_to_ranks, query_nrows,
                       uniq_labels, n_unique,
                       ncols, n_neighbors, rank,
                       batch_size, convert_dtype, verbose):

    handle = worker_state(sessionID)["handle"]
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    idx = [d[0] for d in data]
    lbls = [d[1] for d in data]

    idx_cai, idx_local_parts, idx_desc = \
        _build_part_inputs(idx, data_parts_to_ranks,
                           data_nrows, ncols, rank, convert_dtype)

    q_cai, q_local_parts, q_desc = \
        _build_part_inputs(query, query_parts_to_ranks,
                           query_nrows, ncols, rank, convert_dtype)

    cdef vector[int_ptr_vector] *lbls_local_parts = \
        new vector[int_ptr_vector](<int>len(lbls))
    lbls_dev_arr = []
    for i, arr in enumerate(lbls):
        for j in range(arr.shape[1]):
            lbls_arr, _, _, _ = \
                input_to_cuml_array(arr[:, j], order="F",
                                    convert_to_dtype=(np.int32
                                                      if convert_dtype
                                                      else None),
                                    check_dtype=[np.int32])
            lbls_dev_arr.append(lbls_arr)
            lbls_local_parts.at(i).push_back(<int*><uintptr_t>lbls_arr.ptr)

    uniq_labels_d, _, _, _ = \
        input_to_cuml_array(uniq_labels, order='C', check_dtype=np.int32,
                            convert_to_dtype=np.int32)
    cdef int* ptr = <int*><uintptr_t>uniq_labels_d.ptr
    cdef vector[int*] *uniq_labels_vec = new vector[int*]()
    for i in range(uniq_labels_d.shape[0]):
        uniq_labels_vec.push_back(<int*>ptr)
        ptr += <int>uniq_labels_d.shape[1]

    cdef vector[int] *n_unique_vec = \
        new vector[int]()
    for uniq_label in n_unique:
        n_unique_vec.push_back(uniq_label)

    n_outputs = len(n_unique)

    cdef vector[intData_t*] *out_vec \
        = new vector[intData_t*]()
    cdef vector[int64Data_t*] *out_i_vec \
        = new vector[int64Data_t*]()
    cdef vector[floatData_t*] *out_d_vec \
        = new vector[floatData_t*]()

    output = []
    output_i = []
    output_d = []

    for query_part in q_cai:
        n_rows = query_part.shape[0]
        o_ary = CumlArray.zeros(shape=(n_rows, n_outputs),
                                order="C", dtype=np.int32)
        i_ary = CumlArray.zeros(shape=(n_rows, n_neighbors),
                                order="C", dtype=np.int64)
        d_ary = CumlArray.zeros(shape=(n_rows, n_neighbors),
                                order="C", dtype=np.float32)

        output.append(o_ary)
        output_i.append(i_ary)
        output_d.append(d_ary)

        out_vec.push_back(new intData_t(
            <int*><uintptr_t>o_ary.ptr, n_rows * n_outputs))

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
        deref(<vector[int_ptr_vector]*><uintptr_t>lbls_local_parts),
        deref(<vector[int*]*><uintptr_t>uniq_labels_vec),
        deref(<vector[int]*><uintptr_t>n_unique_vec),
        <bool>False,  # column-major index
        <bool>False,  # column-major query
        <int>n_neighbors,
        <size_t>batch_size,
        <bool>verbose
    )

    handle.sync()

    _free_mem(<uintptr_t>out_vec,
              <uintptr_t>out_i_vec,
              <uintptr_t>out_d_vec,
              <uintptr_t>idx_local_parts,
              <uintptr_t>idx_desc,
              <uintptr_t>q_local_parts,
              <uintptr_t>q_desc,
              <uintptr_t>lbls_local_parts,
              <uintptr_t>uniq_labels_vec,
              <uintptr_t>n_unique_vec)

    output = list(map(lambda o: o.to_output('cupy'), output))
    output_i = list(map(lambda o: o.to_output('cupy'), output_i))
    output_d = list(map(lambda o: o.to_output('cupy'), output_d))

    return output, output_i, output_d


class KNeighborsClassifier():
    """
    Multi-node Multi-GPU K-Nearest Neighbors Classifier Model.

    K-Nearest Neighbors Classifier is an instance-based learning technique,
    that keeps training samples around for prediction, rather than trying
    to learn a generalizable set of model parameters.
    """
    def __init__(self, client=None, streams_per_handle=0, verbose=False,
                 n_neighbors=5, batch_size=1024):
        self.client = default_client() if client is None else client
        self.streams_per_handle = streams_per_handle
        self.verbose = verbose
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size

    @staticmethod
    def _build_comms(data_handler, query_handler, streams_per_handle, verbose):
        # Communicator clique needs to include the union of workers hosting
        # query and index partitions
        workers = set(data_handler.workers)
        workers.update(query_handler.workers)

        comms = CommsContext(comms_p2p=True,
                             streams_per_handle=streams_per_handle,
                             verbose=verbose)
        comms.init(workers=workers)
        return comms

    def fit(self, X, y):
        """
        Fit a multi-node multi-GPU K-Nearest Neighbors Classifier index

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Index data.
            Acceptable formats: dask CuPy/NumPy/Numba Array

        y : array-like (device or host) shape = (n_samples, n_features)
            Index labels data.
            Acceptable formats: dask CuPy/NumPy/Numba Array

        Returns
        -------
        self : KNeighborsClassifier model
        """
        self.data_handler = \
            DistributedDataHandler.create(data=[X, y],
                                          client=self.client)
        self.datatype = self.data_handler.datatype

        uniq_labels = []
        if y.ndim == 1:
            uniq_labels.append(da.unique(y))
        else:
            n_targets = y.shape[1]
            for i in range(n_targets):
                uniq_labels.append(da.unique(y[:, i]))
        self.uniq_labels = np.array(da.compute(uniq_labels)[0])
        self.n_unique = list(map(lambda x: len(x), self.uniq_labels))
        return self

    def predict(self, X, convert_dtype=True):
        """
        Predict labels for a query from previously stored index
        and index labels.
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

        comms = KNeighborsClassifier._build_comms(self.data_handler,
                                                  query_handler,
                                                  self.streams_per_handle,
                                                  self.verbose)

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
        Invoke knn_classify on Dask workers to perform distributed query
        """
        key = uuid1()
        knn_clf_res = dict([(worker_info[worker]["rank"], self.client.submit(
                            _func_knn_classify,
                            comms.sessionId,
                            self.data_handler.worker_to_parts[worker] if
                            worker in self.data_handler.workers else [],
                            data_parts_to_ranks,
                            data_nrows,
                            query_handler.worker_to_parts[worker] if
                            worker in query_handler.workers else [],
                            query_parts_to_ranks,
                            query_nrows,
                            self.uniq_labels,
                            self.n_unique,
                            X.shape[1],
                            self.n_neighbors,
                            worker_info[worker]["rank"],
                            self.batch_size,
                            convert_dtype,
                            self.verbose,
                            key="%s-%s" % (key, idx),
                            workers=[worker]))
                           for idx, worker in enumerate(comms.worker_addresses)
                            ])

        wait(list(knn_clf_res.values()))
        raise_exception_from_futures(list(knn_clf_res.values()))

        """
        Gather resulting partitions and return result
        """
        out_futures = flatten_grouped_results(self.client,
                                              query_parts_to_ranks,
                                              knn_clf_res,
                                              getter_func=_func_get_o)

        out_i_futures = flatten_grouped_results(self.client,
                                                query_parts_to_ranks,
                                                knn_clf_res,
                                                getter_func=_func_get_i)

        out_d_futures = flatten_grouped_results(self.client,
                                                query_parts_to_ranks,
                                                knn_clf_res,
                                                getter_func=_func_get_d)

        comms.destroy()

        out = to_output(out_futures, self.datatype)
        out_i = to_output(out_i_futures, self.datatype)
        out_d = to_output(out_d_futures, self.datatype)
        return out, out_i, out_d
