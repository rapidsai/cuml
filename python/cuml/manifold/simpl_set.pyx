#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

# distutils: language = c++

import numpy as np
import cupy as cp

import cuml.internals
from cuml.common.base import Base
from cuml.common.input_utils import input_to_cuml_array
from cuml.common.array import CumlArray
from cuml.manifold.umap import UMAP

from rmm._lib.cuda_stream_view cimport cuda_stream_view
from raft.common.handle cimport handle_t
from raft.common.handle import Handle

from libc.stdint cimport uintptr_t
from libc.stdint cimport uint64_t
from libc.stdint cimport int64_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr


cdef extern from "cuml/manifold/umapparams.h" namespace "ML::UMAPParams":

    enum MetricType:
        EUCLIDEAN = 0,
        CATEGORICAL = 1


cdef extern from "cuml/manifold/umapparams.h" namespace "ML":

    cdef cppclass UMAPParams:
        int n_neighbors,
        int n_components,
        int n_epochs,
        float learning_rate,
        float min_dist,
        float spread,
        float set_op_mix_ratio,
        float local_connectivity,
        float repulsion_strength,
        int negative_sample_rate,
        float transform_queue_size,
        int verbosity,
        float a,
        float b,
        float initial_alpha,
        int init,
        int target_n_neighbors,
        MetricType target_metric,
        float target_weight,
        uint64_t random_state,
        bool deterministic,
        int optim_batch_size


cdef extern from "raft/sparse/coo.hpp":
    cdef cppclass COO "raft::sparse::COO<float, int>":
        COO(cuda_stream_view stream)
        void allocate(int nnz, int size, bool init, cuda_stream_view stream)
        int nnz
        float* vals()
        int* rows()
        int* cols()


cdef extern from "cuml/manifold/umap.hpp" namespace "ML::UMAP":

    unique_ptr[COO] get_graph(handle_t &handle,
                              float* X,
                              float* y,
                              int n,
                              int d,
                              UMAPParams* params)

    void refine(handle_t &handle,
                float* X,
                int n,
                int d,
                COO* cgraph_coo,
                UMAPParams* params,
                float* embeddings)


def fuzzy_simplicial_set(X,
                         n_neighbors,
                         random_state=None,
                         metric="euclidean",
                         metric_kwds={},
                         set_op_mix_ratio=1.0,
                         local_connectivity=1.0,
                         verbose=False):
    """Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to be modelled as a fuzzy simplicial set.
    n_neighbors: int
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    metric: string or function (optional, default 'euclidean')
        unused
    metric_kwds: dict (optional, default {})
        unused
    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.
    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    fuzzy_simplicial_set: coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i,
        j) entry of the matrix represents the membership strength of the
        1-simplex between the ith and jth sample points.
    """

    deterministic = random_state is not None
    if not isinstance(random_state, int):
        if isinstance(random_state, np.random.RandomState):
            rs = random_state
        else:
            rs = np.random.RandomState(random_state)
        random_state = rs.randint(low=np.iinfo(np.int32).min,
                                  high=np.iinfo(np.int32).max,
                                  dtype=np.int32)

    cdef UMAPParams* umap_params = new UMAPParams()
    umap_params.n_neighbors = <int> n_neighbors
    umap_params.random_state = <uint64_t> random_state
    umap_params.deterministic = <bool> deterministic
    umap_params.set_op_mix_ratio = <float> set_op_mix_ratio
    umap_params.local_connectivity = <float> local_connectivity
    umap_params.verbosity = <int> verbose

    X_m, n_rows, n_cols, dtype = \
        input_to_cuml_array(X, order='C', check_dtype=np.float32)

    handle = Handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
    cdef unique_ptr[COO] fss_graph = get_graph(handle_[0],
                                               <float*><uintptr_t> X_m.ptr,
                                               <float*><uintptr_t> NULL,
                                               <int> X_m.shape[0],
                                               <int> X_m.shape[1],
                                               <UMAPParams*> umap_params)

    cdef int nnz = fss_graph.get().nnz
    coo_arrays = []
    for ptr, dtype in [(<uintptr_t>fss_graph.get().vals(), np.float32),
                       (<uintptr_t>fss_graph.get().rows(), np.int32),
                       (<uintptr_t>fss_graph.get().cols(), np.int32)]:
        mem = cp.cuda.UnownedMemory(ptr=ptr,
                                    size=nnz * 4,
                                    owner=None,
                                    device_id=-1)
        memptr = cp.cuda.memory.MemoryPointer(mem, 0)
        coo_arrays.append(cp.ndarray(nnz, dtype=dtype, memptr=memptr))

    result = cp.sparse.coo_matrix(((coo_arrays[0],
                                    (coo_arrays[1], coo_arrays[2]))))
    result = result.copy()

    return result


@cuml.internals.api_return_generic()
def simplicial_set_embedding(
    data,
    graph,
    n_components=2,
    initial_alpha=1.0,
    a=None,
    b=None,
    repulsion_strength=1.0,
    negative_sample_rate=5,
    n_epochs=None,
    init="spectral",
    random_state=None,
    metric="euclidean",
    metric_kwds={},
    output_metric="euclidean",
    output_metric_kwds={},
    verbose=False,
):
    """Perform a fuzzy simplicial set embedding, using a specified
    initialisation method and then minimizing the fuzzy set cross entropy
    between the 1-skeletons of the high and low dimensional fuzzy simplicial
    sets.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The source data to be embedded by UMAP.
    graph: sparse matrix
        The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.
    n_components: int
        The dimensionality of the euclidean space into which to embed the data.
    initial_alpha: float
        Initial learning rate for the SGD.
    a: float
        Parameter of differentiable approximation of right adjoint functor
    b: float
        Parameter of differentiable approximation of right adjoint functor
    repulsion_strength: float
        Weight to apply to negative samples.
    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.
    n_epochs: int (optional, default 0)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If 0 is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    init: string
        How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    metric: string or callable
        unused
    metric_kwds: dict
        unused
    output_metric: function
        Function returning the distance between two points in embedding space
        and the gradient of the distance wrt the first argument.
    output_metric_kwds: dict
        Key word arguments to be passed to the output_metric function.
    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.
    Returns
    -------
    embedding: array of shape (n_samples, n_components)
        The optimized of ``graph`` into an ``n_components`` dimensional
        euclidean space.
    """

    if init not in ['spectral', 'random']:
        raise Exception("Initialization strategy not supported: %d" % init)

    if output_metric not in ['euclidean', 'categorical']:
        raise Exception("Invalid output metric: {}" % output_metric)

    n_epochs = n_epochs if n_epochs else 0

    if a is None or b is None:
        spread = 1.0
        min_dist = 0.1
        a, b = UMAP.find_ab_params(spread, min_dist)

    deterministic = random_state is not None
    if not isinstance(random_state, int):
        if isinstance(random_state, np.random.RandomState):
            rs = random_state
        else:
            rs = np.random.RandomState(random_state)
        random_state = rs.randint(low=np.iinfo(np.int32).min,
                                  high=np.iinfo(np.int32).max,
                                  dtype=np.int32)

    cdef UMAPParams* umap_params = new UMAPParams()
    umap_params.n_components = <int> n_components
    umap_params.initial_alpha = <int> initial_alpha
    umap_params.a = <int> a
    umap_params.b = <int> b
    umap_params.repulsion_strength = <float> repulsion_strength
    umap_params.negative_sample_rate = <int> negative_sample_rate
    umap_params.n_epochs = <int> n_epochs
    if init == 'spectral':
        umap_params.init = <int> 1
    else:  # init == 'random'
        umap_params.init = <int> 0
    umap_params.random_state = <int> random_state
    umap_params.deterministic = <bool> deterministic
    if output_metric == 'euclidean':
        umap_params.target_metric = MetricType.EUCLIDEAN
    else:  # output_metric == 'categorical'
        umap_params.target_metric = MetricType.CATEGORICAL
    umap_params.target_weight = <float> output_metric_kwds['p'] \
        if 'p' in output_metric_kwds else 0
    umap_params.verbosity = <int> verbose

    X_m, n_rows, n_cols, dtype = \
        input_to_cuml_array(data, order='C', check_dtype=np.float32)

    graph = graph.tocoo()
    graph.sum_duplicates()
    if not isinstance(graph, cp.sparse.coo_matrix):
        graph = cp.sparse.coo_matrix(graph)

    handle = Handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
    cdef COO* fss_graph = new COO(handle_.get_stream())
    fss_graph.allocate(graph.nnz, graph.shape[0], False, handle_.get_stream())
    handle_.sync_stream()

    nnz_size = graph.nnz * 4  # sizeof(float) == sizeof(int) == 4
    for dst_ptr, src_ptr in \
        [(<uintptr_t>fss_graph.vals(), graph.data.data.ptr),
         (<uintptr_t>fss_graph.rows(), graph.row.data.ptr),
         (<uintptr_t>fss_graph.cols(), graph.col.data.ptr)]:
        dest_buff = cp.cuda.UnownedMemory(ptr=dst_ptr,
                                          size=nnz_size,
                                          owner=None,
                                          device_id=-1)
        dest_mptr = cp.cuda.memory.MemoryPointer(dest_buff, 0)
        src_buff = cp.cuda.UnownedMemory(ptr=src_ptr,
                                         size=nnz_size,
                                         owner=None,
                                         device_id=-1)
        src_mptr = cp.cuda.memory.MemoryPointer(src_buff, 0)
        dest_mptr.copy_from_device(src_mptr, nnz_size)

    embedding = CumlArray.zeros((X_m.shape[0], n_components),
                                order="C", dtype=np.float32,
                                index=X_m.index)

    refine(handle_[0],
           <float*><uintptr_t> X_m.ptr,
           <int> X_m.shape[0],
           <int> X_m.shape[1],
           <COO*> fss_graph,
           <UMAPParams*> umap_params,
           <float*><uintptr_t> embedding.ptr)

    cuml.internals.set_api_output_type("cupy")
    return embedding
