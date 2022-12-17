#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

import typing
import ctypes
import numpy as np
import pandas as pd
import warnings

import joblib

import cupy
import cupyx

import numba.cuda as cuda

from cuml.manifold.umap_utils cimport *
from cuml.manifold.umap_utils import GraphHolder, find_ab_params

from cuml.common.sparsefuncs import extract_knn_graph
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix,\
    coo_matrix as cp_coo_matrix, csc_matrix as cp_csc_matrix

import cuml.internals
from cuml.common import using_output_type
from cuml.internals.base import UniversalBase
from pylibraft.common.handle cimport handle_t
from cuml.common.doc_utils import generate_docstring
from cuml.internals import logger
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.memory_utils import using_output_type
from cuml.internals.import_utils import has_scipy
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.mixins import CMajorInputTagMixin
from cuml.common.sparse_utils import is_sparse
from cuml.metrics.distance_type cimport DistanceType

from cuml.manifold.simpl_set import fuzzy_simplicial_set, \
    simplicial_set_embedding

if has_scipy(True):
    import scipy.sparse

from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.api_decorators import device_interop_preparation
from cuml.internals.api_decorators import enable_device_interop

import rmm

from libc.stdint cimport uintptr_t
from libc.stdlib cimport free

from libcpp.memory cimport shared_ptr

cimport cuml.common.cuda


cdef extern from "cuml/manifold/umap.hpp" namespace "ML::UMAP":

    void fit(handle_t & handle,
             float * X,
             float * y,
             int n,
             int d,
             int64_t * knn_indices,
             float * knn_dists,
             UMAPParams * params,
             float * embeddings,
             COO * graph) except +

    void fit_sparse(handle_t &handle,
                    int *indptr,
                    int *indices,
                    float *data,
                    size_t nnz,
                    float *y,
                    int n,
                    int d,
                    UMAPParams *params,
                    float *embeddings,
                    COO * graph) except +

    void transform(handle_t & handle,
                   float * X,
                   int n,
                   int d,
                   int64_t * knn_indices,
                   float * knn_dists,
                   float * orig_X,
                   int orig_n,
                   float * embedding,
                   int embedding_n,
                   UMAPParams * params,
                   float * out) except +

    void transform_sparse(handle_t &handle,
                          int *indptr,
                          int *indices,
                          float *data,
                          size_t nnz,
                          int n,
                          int d,
                          int *orig_x_indptr,
                          int *orig_x_indices,
                          float *orig_x_data,
                          size_t orig_nnz,
                          int orig_n,
                          float *embedding,
                          int embedding_n,
                          UMAPParams *params,
                          float *transformed) except +


class UMAP(UniversalBase,
           CMajorInputTagMixin):
    """
    Uniform Manifold Approximation and Projection

    Finds a low dimensional embedding of the data that approximates
    an underlying manifold.

    Adapted from https://github.com/lmcinnes/umap/blob/master/umap/umap_.py

    The UMAP algorithm is outlined in [1]. This implementation follows the
    GPU-accelerated version as described in [2].

    Parameters
    ----------
    n_neighbors: float (optional, default 15)
        The size of local neighborhood (in terms of number of neighboring
        sample points) used for manifold approximation. Larger values
        result in more global views of the manifold, while smaller
        values result in more local data being preserved. In general
        values should be in the range 2 to 100.
    n_components: int (optional, default 2)
        The dimension of the space to embed into. This defaults to 2 to
        provide easy visualization, but can reasonably be set to any
    metric : string (default='euclidean').
        Distance metric to use. Supported distances are ['l1, 'cityblock',
        'taxicab', 'manhattan', 'euclidean', 'l2', 'sqeuclidean', 'canberra',
        'minkowski', 'chebyshev', 'linf', 'cosine', 'correlation', 'hellinger',
        'hamming', 'jaccard']
        Metrics that take arguments (such as minkowski) can have arguments
        passed via the metric_kwds dictionary.
    n_epochs: int (optional, default None)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    learning_rate: float (optional, default 1.0)
        The initial learning rate for the embedding optimization.
    init: string (optional, default 'spectral')
        How to initialize the low dimensional embedding. Options are:

        * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
        * 'random': assign initial embedding positions at random.

    min_dist: float (optional, default 0.1)
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points
        on the manifold are drawn closer together, while larger values will
        result on a more even dispersal of points. The value should be set
        relative to the ``spread`` value, which determines the scale at which
        embedded points will be spread out.
    spread: float (optional, default 1.0)
        The effective scale of embedded points. In combination with
        ``min_dist`` this determines how clustered/clumped the embedded
        points are.
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
    repulsion_strength: float (optional, default 1.0)
        Weighting applied to negative samples in low dimensional embedding
        optimization. Values higher than one will result in greater weight
        being given to negative samples.
    negative_sample_rate: int (optional, default 5)
        The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.
    transform_queue_size: float (optional, default 4.0)
        For transform operations (embedding new points using a trained model
        this will control how aggressively to search for nearest neighbors.
        Larger values will result in slower performance but more accurate
        nearest neighbor evaluation.
    a: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    b: float (optional, default None)
        More specific parameters controlling the embedding. If None these
        values are set automatically as determined by ``min_dist`` and
        ``spread``.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    hash_input: bool, optional (default = False)
        UMAP can hash the training input so that exact embeddings
        are returned when transform is called on the same data upon
        which the model was trained. This enables consistent
        behavior between calling ``model.fit_transform(X)`` and
        calling ``model.fit(X).transform(X)``. Not that the CPU-based
        UMAP reference implementation does this by default. This
        feature is made optional in the GPU version due to the
        significant overhead in copying memory to the host for
        computing the hash.
    random_state : int, RandomState instance or None, optional (default=None)
        random_state is the seed used by the random number generator during
        embedding initialization and during sampling used by the optimizer.
        Note: Unfortunately, achieving a high amount of parallelism during
        the optimization stage often comes at the expense of determinism,
        since many floating-point additions are being made in parallel
        without a deterministic ordering. This causes slightly different
        results across training sessions, even when the same seed is used
        for random number generation. Setting a random_state will enable
        consistency of trained embeddings, allowing for reproducible results
        to 3 digits of precision, but will do so at the expense of potentially
        slower training and increased memory usage.
    callback: An instance of GraphBasedDimRedCallback class
        Used to intercept the internal state of embeddings while they are being
        trained. Example of callback usage:

        .. code-block:: python

            from cuml.internals import GraphBasedDimRedCallback

            class CustomCallback(GraphBasedDimRedCallback):
                def on_preprocess_end(self, embeddings):
                    print(embeddings.copy_to_host())

                def on_epoch_end(self, embeddings):
                    print(embeddings.copy_to_host())

                def on_train_end(self, embeddings):
                    print(embeddings.copy_to_host())

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Notes
    -----
    This module is heavily based on Leland McInnes' reference UMAP package.
    However, there are a number of differences and features that are not yet
    implemented in `cuml.umap`:

    * Using a pre-computed pairwise distance matrix (under consideration
      for future releases)
    * Manual initialization of initial embedding positions

    In addition to these missing features, you should expect to see
    the final embeddings differing between cuml.umap and the reference
    UMAP. In particular, the reference UMAP uses an approximate kNN
    algorithm for large data sizes while cuml.umap always uses exact
    kNN.

    References
    ----------
    .. [1] `Leland McInnes, John Healy, James Melville
       UMAP: Uniform Manifold Approximation and Projection for Dimension
       Reduction <https://arxiv.org/abs/1802.03426>`_

    .. [2] `Corey Nolet, Victor Lafargue, Edward Raff, Thejaswi Nanditale,
       Tim Oates, John Zedlewski, Joshua Patterson
       Bringing UMAP Closer to the Speed of Light with GPU Acceleration
       <https://arxiv.org/abs/2008.00325>`_
    """

    _cpu_estimator_import_path = 'umap.UMAP'
    embedding_ = CumlArrayDescriptor(order='C')

    @device_interop_preparation
    def __init__(self, *,
                 n_neighbors=15,
                 n_components=2,
                 metric="euclidean",
                 metric_kwds=None,
                 n_epochs=None,
                 learning_rate=1.0,
                 min_dist=0.1,
                 spread=1.0,
                 set_op_mix_ratio=1.0,
                 local_connectivity=1.0,
                 repulsion_strength=1.0,
                 negative_sample_rate=5,
                 transform_queue_size=4.0,
                 init="spectral",
                 verbose=False,
                 a=None,
                 b=None,
                 target_n_neighbors=-1,
                 target_weight=0.5,
                 target_metric="categorical",
                 handle=None,
                 hash_input=False,
                 random_state=None,
                 callback=None,
                 output_type=None):

        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)

        self.hash_input = hash_input

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.n_epochs = n_epochs

        if init == "spectral" or init == "random":
            self.init = init
        else:
            raise Exception("Initialization strategy not supported: %d" % init)

        if a is None or b is None:
            a, b = self.find_ab_params(spread, min_dist)

        self.a = a
        self.b = b

        self.learning_rate = learning_rate
        self.min_dist = min_dist
        self.spread = spread
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.repulsion_strength = repulsion_strength
        self.negative_sample_rate = negative_sample_rate
        self.transform_queue_size = transform_queue_size
        self.target_n_neighbors = target_n_neighbors
        self.target_weight = target_weight

        self.deterministic = random_state is not None

        # Check to see if we are already a random_state (type==np.uint64).
        # Reuse this if already passed (can happen from get_params() of another
        # instance)
        if isinstance(random_state, np.uint64):
            self.random_state = random_state
        else:
            # Otherwise create a RandomState instance to generate a new
            # np.uint64
            if isinstance(random_state, np.random.RandomState):
                rs = random_state
            else:
                rs = np.random.RandomState(random_state)

            self.random_state = rs.randint(low=0,
                                           high=np.iinfo(np.uint32).max,
                                           dtype=np.uint32)

        if target_metric == "euclidean" or target_metric == "categorical":
            self.target_metric = target_metric
        else:
            raise Exception("Invalid target metric: {}" % target_metric)

        self.callback = callback  # prevent callback destruction
        self.embedding_ = None

        self.validate_hyperparams()

        self.sparse_fit = False
        self._input_hash = None
        self._small_data = False

    def validate_hyperparams(self):

        if self.min_dist > self.spread:
            raise ValueError("min_dist should be <= spread")

    @staticmethod
    def _build_umap_params(cls):
        cdef UMAPParams* umap_params = new UMAPParams()
        umap_params.n_neighbors = <int> cls.n_neighbors
        umap_params.n_components = <int> cls.n_components
        umap_params.n_epochs = <int> cls.n_epochs if cls.n_epochs else 0
        umap_params.learning_rate = <float> cls.learning_rate
        umap_params.min_dist = <float> cls.min_dist
        umap_params.spread = <float> cls.spread
        umap_params.set_op_mix_ratio = <float> cls.set_op_mix_ratio
        umap_params.local_connectivity = <float> cls.local_connectivity
        umap_params.repulsion_strength = <float> cls.repulsion_strength
        umap_params.negative_sample_rate = <int> cls.negative_sample_rate
        umap_params.transform_queue_size = <int> cls.transform_queue_size
        umap_params.verbosity = <int> cls.verbose
        umap_params.a = <float> cls.a
        umap_params.b = <float> cls.b
        if cls.init == "spectral":
            umap_params.init = <int> 1
        else:  # self.init == "random"
            umap_params.init = <int> 0
        umap_params.target_n_neighbors = <int> cls.target_n_neighbors
        if cls.target_metric == "euclidean":
            umap_params.target_metric = MetricType.EUCLIDEAN
        else:  # self.target_metric == "categorical"
            umap_params.target_metric = MetricType.CATEGORICAL
        umap_params.target_weight = <float> cls.target_weight
        umap_params.random_state = <uint64_t> cls.random_state
        umap_params.deterministic = <bool> cls.deterministic

        # metric
        metric_parsing = {
            "l2": DistanceType.L2SqrtUnexpanded,
            "euclidean": DistanceType.L2SqrtUnexpanded,
            "sqeuclidean": DistanceType.L2Unexpanded,
            "cityblock": DistanceType.L1,
            "l1": DistanceType.L1,
            "manhattan": DistanceType.L1,
            "taxicab": DistanceType.L1,
            "minkowski": DistanceType.LpUnexpanded,
            "chebyshev": DistanceType.Linf,
            "linf": DistanceType.Linf,
            "cosine": DistanceType.CosineExpanded,
            "correlation": DistanceType.CorrelationExpanded,
            "hellinger": DistanceType.HellingerExpanded,
            "hamming": DistanceType.HammingUnexpanded,
            "jaccard": DistanceType.JaccardExpanded,
            "canberra": DistanceType.Canberra
        }

        if cls.metric.lower() in metric_parsing:
            umap_params.metric = metric_parsing[cls.metric.lower()]
        else:
            raise ValueError("Invalid value for metric: {}"
                             .format(cls.metric))

        if cls.metric_kwds is None:
            umap_params.p = <float> 2.0
        else:
            umap_params.p = <float>cls.metric_kwds.get('p')

        cdef uintptr_t callback_ptr = 0
        if cls.callback:
            callback_ptr = cls.callback.get_native_callback()
            umap_params.callback = <GraphBasedDimRedCallback*>callback_ptr

        return <size_t>umap_params

    @staticmethod
    def _destroy_umap_params(ptr):
        cdef UMAPParams* umap_params = <UMAPParams*> <size_t> ptr
        free(umap_params)

    @staticmethod
    def find_ab_params(spread, min_dist):
        return find_ab_params(spread, min_dist)

    @generate_docstring(convert_dtype_cast='np.float32',
                        X='dense_sparse',
                        skip_parameters_heading=True)
    @enable_device_interop
    def fit(self, X, y=None, convert_dtype=True,
            knn_graph=None) -> "UMAP":
        """
        Fit X into an embedded space.

        Parameters
        ----------
        knn_graph : sparse array-like (device or host)
            shape=(n_samples, n_samples)
            A sparse array containing the k-nearest neighbors of X,
            where the columns are the nearest neighbor indices
            for each row and the values are their distances.
            It's important that `k>=n_neighbors`,
            so that UMAP can model the neighbors from this graph,
            instead of building its own internally.
            Users using the knn_graph parameter provide UMAP
            with their own run of the KNN algorithm. This allows the user
            to pick a custom distance function (sometimes useful
            on certain datasets) whereas UMAP uses euclidean by default.
            The custom distance function should match the metric used
            to train UMAP embeddings. Storing and reusing a knn_graph
            will also provide a speedup to the UMAP algorithm
            when performing a grid search.
            Acceptable formats: sparse SciPy ndarray, CuPy device ndarray,
            CSR/COO preferred other formats will go through conversion to CSR
        """
        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        if y is not None and knn_graph is not None\
                and self.target_metric != "categorical":
            raise ValueError("Cannot provide a KNN graph when in \
            semi-supervised mode with categorical target_metric for now.")

        # Handle sparse inputs
        if is_sparse(X):

            self._raw_data = SparseCumlArray(X, convert_to_dtype=cupy.float32,
                                             convert_format=False)
            self.n_rows, self.n_dims = self._raw_data.shape
            self.sparse_fit = True

        # Handle dense inputs
        else:
            self._raw_data, self.n_rows, self.n_dims, dtype = \
                input_to_cuml_array(X, order='C', check_dtype=np.float32,
                                    convert_to_dtype=(np.float32
                                                      if convert_dtype
                                                      else None))

        if self.n_rows <= 1:
            raise ValueError("There needs to be more than 1 sample to "
                             "build nearest the neighbors graph")

        (knn_indices_m, knn_indices_ctype), (knn_dists_m, knn_dists_ctype) =\
            extract_knn_graph(knn_graph, convert_dtype)

        cdef uintptr_t knn_indices_raw = knn_indices_ctype or 0
        cdef uintptr_t knn_dists_raw = knn_dists_ctype or 0

        self.n_neighbors = min(self.n_rows, self.n_neighbors)

        self.embedding_ = CumlArray.zeros((self.n_rows,
                                           self.n_components),
                                          order="C", dtype=np.float32,
                                          index=self._raw_data.index)

        if self.hash_input:
            self._input_hash = joblib.hash(self._raw_data.to_output('numpy'))

        cdef handle_t * handle_ = \
            <handle_t*> <size_t> self.handle.getHandle()

        cdef uintptr_t embed_raw = self.embedding_.ptr

        cdef UMAPParams* umap_params = \
            <UMAPParams*> <size_t> UMAP._build_umap_params(self)

        cdef uintptr_t y_raw = 0

        if y is not None:
            y_m, _, _, _ = \
                input_to_cuml_array(y, check_dtype=np.float32,
                                    convert_to_dtype=(np.float32
                                                      if convert_dtype
                                                      else None))
            y_raw = y_m.ptr

        fss_graph = GraphHolder.new_graph(handle_.get_stream())
        if self.sparse_fit:
            fit_sparse(handle_[0],
                       <int*><uintptr_t> self._raw_data.indptr.ptr,
                       <int*><uintptr_t> self._raw_data.indices.ptr,
                       <float*><uintptr_t> self._raw_data.data.ptr,
                       <size_t> self._raw_data.nnz,
                       <float*> y_raw,
                       <int> self.n_rows,
                       <int> self.n_dims,
                       <UMAPParams*> umap_params,
                       <float*> embed_raw,
                       <COO*> fss_graph.get())

        else:
            fit(handle_[0],
                <float*><uintptr_t> self._raw_data.ptr,
                <float*> y_raw,
                <int> self.n_rows,
                <int> self.n_dims,
                <int64_t*> knn_indices_raw,
                <float*> knn_dists_raw,
                <UMAPParams*>umap_params,
                <float*>embed_raw,
                <COO*> fss_graph.get())

        self.graph_ = fss_graph.get_cupy_coo()

        self.handle.sync()

        UMAP._destroy_umap_params(<size_t>umap_params)

        return self

    @generate_docstring(convert_dtype_cast='np.float32',
                        skip_parameters_heading=True,
                        return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Embedding of the \
                                                       data in \
                                                       low-dimensional space.',
                                       'shape': '(n_samples, n_components)'})
    @cuml.internals.api_base_fit_transform()
    @enable_device_interop
    def fit_transform(self, X, y=None, convert_dtype=True,
                      knn_graph=None) -> CumlArray:
        """
        Fit X into an embedded space and return that transformed
        output.

        There is a subtle difference between calling fit_transform(X)
        and calling fit().transform(). Calling fit_transform(X) will
        train the embeddings on X and return the embeddings. Calling
        fit(X).transform(X) will train the embeddings on X and then
        run a second optimization.

        Parameters
        ----------
        knn_graph : sparse array-like (device or host)
            shape=(n_samples, n_samples)
            A sparse array containing the k-nearest neighbors of X,
            where the columns are the nearest neighbor indices
            for each row and the values are their distances.
            It's important that `k>=n_neighbors`,
            so that UMAP can model the neighbors from this graph,
            instead of building its own internally.
            Users using the knn_graph parameter provide UMAP
            with their own run of the KNN algorithm. This allows the user
            to pick a custom distance function (sometimes useful
            on certain datasets) whereas UMAP uses euclidean by default.
            The custom distance function should match the metric used
            to train UMAP embeddings. Storing and reusing a knn_graph
            will also provide a speedup to the UMAP algorithm
            when performing a grid search.
            Acceptable formats: sparse SciPy ndarray, CuPy device ndarray,
            CSR/COO preferred other formats will go through conversion to CSR

        """
        self.fit(X, y, convert_dtype=convert_dtype, knn_graph=knn_graph)

        return self.embedding_

    @generate_docstring(convert_dtype_cast='np.float32',
                        skip_parameters_heading=True,
                        return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Embedding of the \
                                                       data in \
                                                       low-dimensional space.',
                                       'shape': '(n_samples, n_components)'})
    @enable_device_interop
    def transform(self, X, convert_dtype=True, knn_graph=None) -> CumlArray:
        """
        Transform X into the existing embedded space and return that
        transformed output.

        Please refer to the reference UMAP implementation for information
        on the differences between fit_transform() and running fit()
        transform().

        Specifically, the transform() function is stochastic:
        https://github.com/lmcinnes/umap/issues/158

        Parameters
        ----------
        knn_graph : sparse array-like (device or host)
            shape=(n_samples, n_samples)
            A sparse array containing the k-nearest neighbors of X,
            where the columns are the nearest neighbor indices
            for each row and the values are their distances.
            It's important that `k>=n_neighbors`,
            so that UMAP can model the neighbors from this graph,
            instead of building its own internally.
            Users using the knn_graph parameter provide UMAP
            with their own run of the KNN algorithm. This allows the user
            to pick a custom distance function (sometimes useful
            on certain datasets) whereas UMAP uses euclidean by default.
            The custom distance function should match the metric used
            to train UMAP embeddings. Storing and reusing a knn_graph
            will also provide a speedup to the UMAP algorithm
            when performing a grid search.
            Acceptable formats: sparse SciPy ndarray, CuPy device ndarray,
            CSR/COO preferred other formats will go through conversion to CSR

        """
        if len(X.shape) != 2:
            raise ValueError("X should be two dimensional")

        if is_sparse(X) and not self.sparse_fit:
            logger.warn("Model was trained on dense data but sparse "
                        "data was provided to transform(). Converting "
                        "to dense.")
            X = X.todense()

        elif not is_sparse(X) and self.sparse_fit:
            logger.warn("Model was trained on sparse data but dense "
                        "data was provided to transform(). Converting "
                        "to sparse.")
            X = cupyx.scipy.sparse.csr_matrix(X)

        if is_sparse(X):
            X_m = SparseCumlArray(X, convert_to_dtype=cupy.float32,
                                  convert_format=False)
            index = None
        else:
            X_m, n_rows, n_cols, dtype = \
                input_to_cuml_array(X, order='C', check_dtype=np.float32,
                                    convert_to_dtype=(np.float32
                                                      if convert_dtype
                                                      else None))
            index = X_m.index
        n_rows = X_m.shape[0]
        n_cols = X_m.shape[1]

        if n_cols != self._raw_data.shape[1]:
            raise ValueError("n_features of X must match n_features of "
                             "training data")

        if self.hash_input:
            if joblib.hash(X_m.to_output('numpy')) == self._input_hash:
                del X_m
                return self.embedding_

        embedding = CumlArray.zeros((X_m.shape[0],
                                    self.n_components),
                                    order="C", dtype=np.float32,
                                    index=index)
        cdef uintptr_t xformed_ptr = embedding.ptr

        (knn_indices_m, knn_indices_ctype), (knn_dists_m, knn_dists_ctype) =\
            extract_knn_graph(knn_graph, convert_dtype)

        cdef uintptr_t knn_indices_raw = knn_indices_ctype or 0
        cdef uintptr_t knn_dists_raw = knn_dists_ctype or 0

        cdef handle_t * handle_ = \
            <handle_t*> <size_t> self.handle.getHandle()

        cdef uintptr_t embed_ptr = self.embedding_.ptr

        cdef UMAPParams* umap_params = \
            <UMAPParams*> <size_t> UMAP._build_umap_params(self)

        if self.sparse_fit:
            transform_sparse(handle_[0],
                             <int*><uintptr_t> X_m.indptr.ptr,
                             <int*><uintptr_t> X_m.indices.ptr,
                             <float*><uintptr_t> X_m.data.ptr,
                             <size_t> X_m.nnz,
                             <int> X_m.shape[0],
                             <int> X_m.shape[1],
                             <int*><uintptr_t> self._raw_data.indptr.ptr,
                             <int*><uintptr_t> self._raw_data.indices.ptr,
                             <float*><uintptr_t> self._raw_data.data.ptr,
                             <size_t> self._raw_data.nnz,
                             <int> self._raw_data.shape[0],
                             <float*> embed_ptr,
                             <int> self._raw_data.shape[0],
                             <UMAPParams*> umap_params,
                             <float*> xformed_ptr)
        else:
            transform(handle_[0],
                      <float*><uintptr_t> X_m.ptr,
                      <int> X_m.shape[0],
                      <int> X_m.shape[1],
                      <int64_t*> knn_indices_raw,
                      <float*> knn_dists_raw,
                      <float*><uintptr_t>self._raw_data.ptr,
                      <int> self._raw_data.shape[0],
                      <float*> embed_ptr,
                      <int> self._raw_data.shape[0],
                      <UMAPParams*> umap_params,
                      <float*> xformed_ptr)
        self.handle.sync()

        UMAP._destroy_umap_params(<size_t>umap_params)

        del X_m
        return embedding

    def get_param_names(self):
        return super().get_param_names() + [
            "n_neighbors",
            "n_components",
            "n_epochs",
            "learning_rate",
            "min_dist",
            "spread",
            "set_op_mix_ratio",
            "local_connectivity",
            "repulsion_strength",
            "negative_sample_rate",
            "transform_queue_size",
            "init",
            "a",
            "b",
            "target_n_neighbors",
            "target_weight",
            "target_metric",
            "hash_input",
            "random_state",
            "callback",
            "metric",
            "metric_kwds"
        ]

    def get_attr_names(self):
        return ['_raw_data', 'embedding_', '_input_hash', '_small_data']
