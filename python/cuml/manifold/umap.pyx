#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import cudf
import cuml
import ctypes
import numpy as np
import pandas as pd
import warnings

import joblib

import cupy

import numba.cuda as cuda

from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix,\
    coo_matrix as cp_coo_matrix, csc_matrix as cp_csc_matrix

from cuml.common.base import Base
from cuml.raft.common.handle cimport handle_t
from cuml.common.doc_utils import generate_docstring
from cuml.common.input_utils import input_to_cuml_array
from cuml.common.memory_utils import with_cupy_rmm
from cuml.common.import_utils import has_scipy
from cuml.common.array import CumlArray

import rmm

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdint cimport uint64_t
from libc.stdint cimport int64_t
from libc.stdlib cimport calloc, malloc, free

from libcpp.memory cimport shared_ptr

cimport cuml.common.cuda


cdef extern from "cuml/manifold/umapparams.h" namespace "ML::UMAPParams":

    enum MetricType:
        EUCLIDEAN = 0,
        CATEGORICAL = 1

cdef extern from "cuml/common/callback.hpp" namespace "ML::Internals":

    cdef cppclass GraphBasedDimRedCallback

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
        float target_weights,
        uint64_t random_state,
        bool multicore_implem,
        int optim_batch_size,
        GraphBasedDimRedCallback * callback


cdef extern from "cuml/manifold/umap.hpp" namespace "ML":
    void fit(handle_t & handle,
             float * X,
             int n,
             int d,
             int64_t * knn_indices,
             float * knn_dists,
             UMAPParams * params,
             float * embeddings) except +

    void fit(handle_t & handle,
             float * X,
             float * y,
             int n,
             int d,
             int64_t * knn_indices,
             float * knn_dists,
             UMAPParams * params,
             float * embeddings) except +

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


class UMAP(Base):
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
    optim_batch_size: int (optional, default 100000 / n_components)
        Used to maintain the consistency of embeddings for large datasets.
        The optimization step will be processed with at most optim_batch_size
        edges at once preventing inconsistencies. A lower batch size will yield
        more consistently repeatable embeddings at the cost of speed.
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
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_output_type`.
        See :ref:`output-data-type-configuration` for more info.

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

    def __init__(self,
                 n_neighbors=15,
                 n_components=2,
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
                 target_weights=0.5,
                 target_metric="categorical",
                 handle=None,
                 hash_input=False,
                 random_state=None,
                 optim_batch_size=0,
                 callback=None,
                 output_type=None):

        super(UMAP, self).__init__(handle=handle, verbose=verbose,
                                   output_type=output_type)

        self.hash_input = hash_input

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.n_epochs = n_epochs if n_epochs else 0

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
        self.target_weights = target_weights

        self.multicore_implem = random_state is None

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
                                           high=np.iinfo(np.uint64).max,
                                           dtype=np.uint64)

        if target_metric == "euclidean" or target_metric == "categorical":
            self.target_metric = target_metric
        else:
            raise Exception("Invalid target metric: {}" % target_metric)

        self.optim_batch_size = <int> optim_batch_size

        self.callback = callback  # prevent callback destruction
        self._X_m = None  # accessed via X_m
        self._embedding_ = None  # accessed via embedding_

        self.validate_hyperparams()

    def validate_hyperparams(self):

        if self.min_dist > self.spread:
            raise ValueError("min_dist should be <= spread")

    @staticmethod
    def _build_umap_params(cls):
        cdef UMAPParams* umap_params = new UMAPParams()
        umap_params.n_neighbors = <int> cls.n_neighbors
        umap_params.n_components = <int> cls.n_components
        umap_params.n_epochs = <int> cls.n_epochs
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
        umap_params.target_weights = <float> cls.target_weights
        umap_params.random_state = <uint64_t> cls.random_state
        umap_params.multicore_implem = <bool> cls.multicore_implem
        umap_params.optim_batch_size = <int> cls.optim_batch_size

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
        """ Function taken from UMAP-learn : https://github.com/lmcinnes/umap
        Fit a, b params for the differentiable curve used in lower
        dimensional fuzzy simplicial complex construction. We want the
        smooth curve (from a pre-defined family with simple gradient) that
        best matches an offset exponential decay.
        """

        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))

        if has_scipy():
            from scipy.optimize import curve_fit
        else:
            raise RuntimeError('Scipy is needed to run find_ab_params')

        xv = np.linspace(0, spread * 3, 300)
        yv = np.zeros(xv.shape)
        yv[xv < min_dist] = 1.0
        yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
        params, covar = curve_fit(curve, xv, yv)
        return params[0], params[1]

    @with_cupy_rmm
    def _extract_knn_graph(self, knn_graph, convert_dtype=True):
        if has_scipy():
            from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
        else:
            from cuml.common.import_utils import DummyClass
            csr_matrix = DummyClass
            coo_matrix = DummyClass
            csc_matrix = DummyClass

        if isinstance(knn_graph, (csc_matrix, cp_csc_matrix)):
            knn_graph = cp_csr_matrix(knn_graph)
            n_samples = knn_graph.shape[0]
            reordering = knn_graph.data.reshape((n_samples, -1))
            reordering = reordering.argsort()
            n_neighbors = reordering.shape[1]
            reordering += (cupy.arange(n_samples) * n_neighbors)[:, np.newaxis]
            reordering = reordering.flatten()
            knn_graph.indices = knn_graph.indices[reordering]
            knn_graph.data = knn_graph.data[reordering]

        knn_indices = None
        if isinstance(knn_graph, (csr_matrix, cp_csr_matrix)):
            knn_indices = knn_graph.indices
        elif isinstance(knn_graph, (coo_matrix, cp_coo_matrix)):
            knn_indices = knn_graph.col

        knn_indices_ptr, knn_dists_ptr = None, None
        if knn_indices is not None:
            knn_dists = knn_graph.data
            knn_indices_m, _, _, _ = \
                input_to_cuml_array(knn_indices, order='C',
                                    deepcopy=True,
                                    check_dtype=np.int64,
                                    convert_to_dtype=(np.int64
                                                      if convert_dtype
                                                      else None))

            knn_dists_m, _, _, _ = \
                input_to_cuml_array(knn_dists, order='C',
                                    deepcopy=True,
                                    check_dtype=np.float32,
                                    convert_to_dtype=(np.float32
                                                      if convert_dtype
                                                      else None))

            return (knn_indices_m, knn_indices_m.ptr),\
                   (knn_dists_m, knn_dists_m.ptr)
        return (None, None), (None, None)

    @generate_docstring(convert_dtype_cast='np.float32',
                        skip_parameters_heading=True)
    @with_cupy_rmm
    def fit(self, X, y=None, convert_dtype=True,
            knn_graph=None):
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
            to train UMAP embeedings. Storing and reusing a knn_graph
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

        self._X_m, self.n_rows, self.n_dims, dtype = \
            input_to_cuml_array(X, order='C', check_dtype=np.float32,
                                convert_to_dtype=(np.float32
                                                  if convert_dtype
                                                  else None))

        if self.n_rows <= 1:
            raise ValueError("There needs to be more than 1 sample to "
                             "build nearest the neighbors graph")

        self._set_base_attributes(output_type=X, n_features=X)

        (knn_indices_m, knn_indices_ctype), (knn_dists_m, knn_dists_ctype) =\
            self._extract_knn_graph(knn_graph, convert_dtype)

        cdef uintptr_t knn_indices_raw = knn_indices_ctype or 0
        cdef uintptr_t knn_dists_raw = knn_dists_ctype or 0

        self.n_neighbors = min(self.n_rows, self.n_neighbors)

        self._embedding_ = CumlArray.zeros((self.n_rows,
                                           self.n_components),
                                           order="C", dtype=np.float32)

        if self.hash_input:
            self.input_hash = joblib.hash(self._X_m.to_output('numpy'))

        cdef handle_t * handle_ = \
            <handle_t*> <size_t> self.handle.getHandle()

        cdef uintptr_t x_raw = self._X_m.ptr
        cdef uintptr_t embed_raw = self._embedding_.ptr

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

            fit(handle_[0],
                <float*> x_raw,
                <float*> y_raw,
                <int> self.n_rows,
                <int> self.n_dims,
                <int64_t*> knn_indices_raw,
                <float*> knn_dists_raw,
                <UMAPParams*>umap_params,
                <float*>embed_raw)

        else:
            fit(handle_[0],
                <float*> x_raw,
                <int> self.n_rows,
                <int> self.n_dims,
                <int64_t*> knn_indices_raw,
                <float*> knn_dists_raw,
                <UMAPParams*>umap_params,
                <float*>embed_raw)
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
    def fit_transform(self, X, y=None, convert_dtype=True,
                      knn_graph=None):
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
            to train UMAP embeedings. Storing and reusing a knn_graph
            will also provide a speedup to the UMAP algorithm
            when performing a grid search.
            Acceptable formats: sparse SciPy ndarray, CuPy device ndarray,
            CSR/COO preferred other formats will go through conversion to CSR

        """
        self.fit(X, y, convert_dtype=convert_dtype,
                 knn_graph=knn_graph)
        out_type = self._get_output_type(X)
        return self._embedding_.to_output(out_type)

    @generate_docstring(convert_dtype_cast='np.float32',
                        skip_parameters_heading=True,
                        return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Embedding of the \
                                                       data in \
                                                       low-dimensional space.',
                                       'shape': '(n_samples, n_components)'})
    @with_cupy_rmm
    def transform(self, X, convert_dtype=True,
                  knn_graph=None):
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
            to train UMAP embeedings. Storing and reusing a knn_graph
            will also provide a speedup to the UMAP algorithm
            when performing a grid search.
            Acceptable formats: sparse SciPy ndarray, CuPy device ndarray,
            CSR/COO preferred other formats will go through conversion to CSR

        """
        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        cdef uintptr_t x_ptr = 0
        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, order='C', check_dtype=np.float32,
                                convert_to_dtype=(np.float32 if convert_dtype
                                                  else None))
        x_ptr = X_m.ptr

        if n_rows <= 1:
            raise ValueError("There needs to be more than 1 sample to "
                             "build nearest the neighbors graph")

        if n_cols != self.n_dims:
            raise ValueError("n_features of X must match n_features of "
                             "training data")

        out_type = self._get_output_type(X)

        if self.hash_input and joblib.hash(X_m.to_output('numpy')) == \
                self.input_hash:
            ret = self._embedding_.to_output(out_type)
            del X_m
            return ret

        embedding = CumlArray.zeros((X_m.shape[0],
                                     self.n_components),
                                    order="C", dtype=np.float32)
        cdef uintptr_t xformed_ptr = embedding.ptr

        (knn_indices_m, knn_indices_ctype), (knn_dists_m, knn_dists_ctype) =\
            self._extract_knn_graph(knn_graph, convert_dtype)

        cdef uintptr_t knn_indices_raw = knn_indices_ctype or 0
        cdef uintptr_t knn_dists_raw = knn_dists_ctype or 0

        cdef handle_t * handle_ = \
            <handle_t*> <size_t> self.handle.getHandle()

        cdef uintptr_t orig_x_raw = self._X_m.ptr
        cdef uintptr_t embed_ptr = self._embedding_.ptr

        cdef UMAPParams* umap_params = \
            <UMAPParams*> <size_t> UMAP._build_umap_params(self)

        transform(handle_[0],
                  <float*>x_ptr,
                  <int> X_m.shape[0],
                  <int> X_m.shape[1],
                  <int64_t*> knn_indices_raw,
                  <float*> knn_dists_raw,
                  <float*>orig_x_raw,
                  <int> self.n_rows,
                  <float*> embed_ptr,
                  <int> self.n_rows,
                  <UMAPParams*> umap_params,
                  <float*> xformed_ptr)
        self.handle.sync()

        UMAP._destroy_umap_params(<size_t>umap_params)

        ret = embedding.to_output(out_type)
        del X_m
        return ret

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
            "target_weights",
            "target_metric",
            "hash_input",
            "random_state",
            "optim_batch_size",
            "callback",
        ]
