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
# distutils: extra_compile_args = -Ofast
# cython: boundscheck = False
# cython: wraparound = False

import ctypes
import numpy as np
import inspect
import pandas as pd
import warnings
import cupy

import cuml.internals
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.base import Base
from pylibraft.common.handle cimport handle_t
import cuml.internals.logger as logger

from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.common.sparse_utils import is_sparse
from cuml.common.doc_utils import generate_docstring
from cuml.common import input_to_cuml_array
from cuml.internals.mixins import CMajorInputTagMixin
from cuml.common.sparsefuncs import extract_knn_graph
from cuml.metrics.distance_type cimport DistanceType
import rmm

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdint cimport int64_t
from libc.stdlib cimport free
from libcpp.memory cimport shared_ptr
from cython.operator cimport dereference as deref

cimport cuml.common.cuda

cdef extern from "cuml/manifold/tsne.h" namespace "ML":

    enum TSNE_ALGORITHM:
        EXACT = 0,
        BARNES_HUT = 1,
        FFT = 2

    cdef cppclass TSNEParams:
        int dim,
        int n_neighbors,
        float theta,
        float epssq,
        float perplexity,
        int perplexity_max_iter,
        float perplexity_tol,
        float early_exaggeration,
        float late_exaggeration,
        int exaggeration_iter,
        float min_gain,
        float pre_learning_rate,
        float post_learning_rate,
        int max_iter,
        float min_grad_norm,
        float pre_momentum,
        float post_momentum,
        long long random_state,
        int verbosity,
        bool initialize_embeddings,
        bool square_distances,
        DistanceType metric,
        float p,
        TSNE_ALGORITHM algorithm


cdef extern from "cuml/manifold/tsne.h" namespace "ML":

    cdef void TSNE_fit(
        handle_t &handle,
        float *X,
        float *Y,
        int n,
        int p,
        int64_t* knn_indices,
        float* knn_dists,
        TSNEParams &params,
        float* kl_div) except +

    cdef void TSNE_fit_sparse(
        const handle_t &handle,
        int *indptr,
        int *indices,
        float *data,
        float *Y,
        int nnz,
        int n,
        int p,
        int* knn_indices,
        float* knn_dists,
        TSNEParams &params,
        float* kl_div) except +


class TSNE(Base,
           CMajorInputTagMixin):
    """
    t-SNE (T-Distributed Stochastic Neighbor Embedding) is an extremely
    powerful dimensionality reduction technique that aims to maintain
    local distances between data points. It is extremely robust to whatever
    dataset you give it, and is used in many areas including cancer research,
    music analysis and neural network weight visualizations.

    cuML's t-SNE supports three algorithms: the original exact algorithm, the
    Barnes-Hut approximation and the fast Fourier transform interpolation
    approximation. The latter two are derived from CannyLabs' open-source CUDA
    code and produce extremely fast embeddings when n_components = 2. The exact
    algorithm is more accurate, but too slow to use on large datasets.

    Parameters
    ----------
    n_components : int (default 2)
        The output dimensionality size. Currently only 2 is supported.
    perplexity : float (default 30.0)
        Larger datasets require a larger value. Consider choosing different
        perplexity values from 5 to 50 and see the output differences.
    early_exaggeration : float (default 12.0)
        Controls the space between clusters. Not critical to tune this.
    late_exaggeration : float (default 1.0)
        Controls the space between clusters. It may be beneficial to increase
        this slightly to improve cluster separation. This will be applied
        after `exaggeration_iter` iterations (FFT only).
    learning_rate : float (default 200.0)
        The learning rate usually between (10, 1000). If this is too high,
        t-SNE could look like a cloud / ball of points.
    n_iter : int (default 1000)
        The more epochs, the more stable/accurate the final embedding.
    n_iter_without_progress : int (default 300)
        Currently unused. When the KL Divergence becomes too small after some
        iterations, terminate t-SNE early.
    min_grad_norm : float (default 1e-07)
        The minimum gradient norm for when t-SNE will terminate early.
        Used in the 'exact' and 'fft' algorithms. Consider reducing if
        the embeddings are unsatisfactory. It's recommended to use a
        smaller value for smaller datasets.
    metric : str (default='euclidean').
        Distance metric to use. Supported distances are ['l1, 'cityblock',
        'manhattan', 'euclidean', 'l2', 'sqeuclidean', 'minkowski',
        'chebyshev', 'cosine', 'correlation']
    init : str 'random' (default 'random')
        Currently supports random initialization.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    random_state : int (default None)
        Setting this can make repeated runs look more similar. Note, however,
        that this highly parallelized t-SNE implementation is not completely
        deterministic between runs, even with the same `random_state`.
    method : str 'fft', 'barnes_hut' or 'exact' (default 'fft')
        'barnes_hut' and 'fft' are fast approximations. 'exact' is more
        accurate but slower.
    angle : float (default 0.5)
        Valid values are between 0.0 and 1.0, which trade off speed and
        accuracy, respectively. Generally, these values are set between 0.2 and
        0.8. (Barnes-Hut only.)
    learning_rate_method : str 'adaptive', 'none' or None (default 'adaptive')
        Either adaptive or None. 'adaptive' tunes the learning rate, early
        exaggeration, perplexity and n_neighbors automatically based on
        input size.
    n_neighbors : int (default 90)
        The number of datapoints you want to use in the
        attractive forces. Smaller values are better for preserving
        local structure, whilst larger values can improve global structure
        preservation. Default is 3 * 30 (perplexity)
    perplexity_max_iter : int (default 100)
        The number of epochs the best gaussian bands are found for.
    exaggeration_iter : int (default 250)
        To promote the growth of clusters, set this higher.
    pre_momentum : float (default 0.5)
        During the exaggeration iteration, more forcefully apply gradients.
    post_momentum : float (default 0.8)
        During the late phases, less forcefully apply gradients.
    square_distances : boolean, default=True
        Whether TSNE should square the distance values.
        Internally, this will be used to compute a kNN graph using the provided
        metric and then squaring it when True. If a `knn_graph` is passed
        to `fit` or `fit_transform` methods, all the distances will be
        squared when True. For example, if a `knn_graph` was obtained using
        'sqeuclidean' metric, the distances will still be squared when True.
        Note: This argument should likely be set to False for distance metrics
        other than 'euclidean' and 'l2'.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    kl_divergence_ : float
        Kullback-Leibler divergence after optimization. An experimental
        feature at this time.

    References
    ----------
    .. [1] `van der Maaten, L.J.P.
       t-Distributed Stochastic Neighbor Embedding
       <https://lvdmaaten.github.io/tsne/>`_

    .. [2] van der Maaten, L.J.P.; Hinton, G.E.
       Visualizing High-Dimensional Data
       Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.

    .. [3] George C. Linderman, Manas Rachh, Jeremy G. Hoskins,
        Stefan Steinerberger, Yuval Kluger Efficient Algorithms for
        t-distributed Stochastic Neighborhood Embedding

    .. tip::
        Maaten and Linderman showcased how t-SNE can be very sensitive to both
        the starting conditions (i.e. random initialization), and how parallel
        versions of t-SNE can generate vastly different results between runs.
        You can run t-SNE multiple times to settle on the best configuration.
        Note that using the same random_state across runs does not guarantee
        similar results each time.

    .. note::
        The CUDA implementation is derived from the excellent CannyLabs open
        source implementation here: https://github.com/CannyLab/tsne-cuda/. The
        CannyLabs code is licensed according to the conditions in
        cuml/cpp/src/tsne/cannylabs_tsne_license.txt. A full description of
        their approach is available in their article t-SNE-CUDA:
        GPU-Accelerated t-SNE and its Applications to Modern Data
        (https://arxiv.org/abs/1807.11824).

    """

    X_m = CumlArrayDescriptor()
    embedding_ = CumlArrayDescriptor()

    def __init__(self, *,
                 n_components=2,
                 perplexity=30.0,
                 early_exaggeration=12.0,
                 late_exaggeration=1.0,
                 learning_rate=200.0,
                 n_iter=1000,
                 n_iter_without_progress=300,
                 min_grad_norm=1e-07,
                 metric='euclidean',
                 metric_params=None,
                 init='random',
                 verbose=False,
                 random_state=None,
                 method='fft',
                 angle=0.5,
                 learning_rate_method='adaptive',
                 n_neighbors=90,
                 perplexity_max_iter=100,
                 exaggeration_iter=250,
                 pre_momentum=0.5,
                 post_momentum=0.8,
                 square_distances=True,
                 handle=None,
                 output_type=None):

        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)

        if n_components < 0:
            raise ValueError("n_components = {} should be more "
                             "than 0.".format(n_components))
        if n_components != 2:
            raise ValueError("Currently TSNE supports n_components = 2; "
                             "but got n_components = {}".format(n_components))
        if perplexity < 0:
            raise ValueError("perplexity = {} should be more than 0.".format(
                             perplexity))
        if early_exaggeration < 0:
            raise ValueError("early_exaggeration = {} should be more "
                             "than 0.".format(early_exaggeration))
        if late_exaggeration < 0:
            raise ValueError("late_exaggeration = {} should be more "
                             "than 0.".format(late_exaggeration))
        if learning_rate < 0:
            raise ValueError("learning_rate = {} should be more "
                             "than 0.".format(learning_rate))
        if n_iter < 0:
            raise ValueError("n_iter = {} should be more than 0.".format(
                             n_iter))
        if n_iter <= 100:
            warnings.warn("n_iter = {} might cause TSNE to output wrong "
                          "results. Set it higher.".format(n_iter))
        if init.lower() != 'random':
            # TODO https://github.com/rapidsai/cuml/issues/3458
            warnings.warn("TSNE does not support {} but only random "
                          "initialization.".format(init))
            init = 'random'
        if angle < 0 or angle > 1:
            raise ValueError("angle = {} should be ≥ 0 and ≤ 1".format(angle))
        if n_neighbors < 0:
            raise ValueError("n_neighbors = {} should be more "
                             "than 0.".format(n_neighbors))
        if n_neighbors > 1023:
            warnings.warn("n_neighbors = {} should be less than 1024")
            n_neighbors = 1023
        if perplexity_max_iter < 0:
            raise ValueError("perplexity_max_iter = {} should be more "
                             "than 0.".format(perplexity_max_iter))
        if exaggeration_iter < 0:
            raise ValueError("exaggeration_iter = {} should be more "
                             "than 0.".format(exaggeration_iter))
        if exaggeration_iter > n_iter:
            raise ValueError("exaggeration_iter = {} should be more less "
                             "than n_iter = {}.".format(exaggeration_iter,
                                                        n_iter))
        if pre_momentum < 0 or pre_momentum > 1:
            raise ValueError("pre_momentum = {} should be more than 0 "
                             "and less than 1.".format(pre_momentum))
        if post_momentum < 0 or post_momentum > 1:
            raise ValueError("post_momentum = {} should be more than 0 "
                             "and less than 1.".format(post_momentum))
        if pre_momentum > post_momentum:
            raise ValueError("post_momentum = {} should be more than "
                             "pre_momentum = {}".format(post_momentum,
                                                        pre_momentum))
        if method == "fft":
            warnings.warn("Starting from version 22.04, the default method "
                          "of TSNE is 'fft'.")

        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.late_exaggeration = late_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_neighbors = n_neighbors
        self.perplexity_max_iter = perplexity_max_iter
        self.exaggeration_iter = exaggeration_iter
        self.pre_momentum = pre_momentum
        self.post_momentum = post_momentum
        if learning_rate_method is None:
            self.learning_rate_method = 'none'
        else:
            # To support `sklearn.base.clone()`, we must minimize altering
            # argument references unless absolutely necessary. Check to see if
            # lowering the string results in the same value, and if so, keep
            # the same reference that was passed in. This may seem redundant,
            # but it allows `clone()` to function without raising an error
            if (learning_rate_method.lower() != learning_rate_method):
                learning_rate_method = learning_rate_method.lower()

            self.learning_rate_method = learning_rate_method
        self.epssq = 0.0025
        self.perplexity_tol = 1e-5
        self.min_gain = 0.01
        self.pre_learning_rate = learning_rate
        self.post_learning_rate = learning_rate * 2
        self.square_distances = square_distances

        self.X_m = None
        self.embedding_ = None

        self.sparse_fit = False

    @generate_docstring(skip_parameters_heading=True,
                        X='dense_sparse',
                        convert_dtype_cast='np.float32')
    def fit(self, X, convert_dtype=True, knn_graph=None) -> "TSNE":
        """
        Fit X into an embedded space.

        Parameters
        ----------
        knn_graph : sparse array-like (device or host), \
                shape=(n_samples, n_samples)
            A sparse array containing the k-nearest neighbors of X,
            where the columns are the nearest neighbor indices
            for each row and the values are their distances.
            Users using the knn_graph parameter provide t-SNE
            with their own run of the KNN algorithm. This allows the user
            to pick a custom distance function (sometimes useful
            on certain datasets) whereas t-SNE uses euclidean by default.
            The custom distance function should match the metric used
            to train t-SNE embeddings. Storing and reusing a knn_graph
            will also provide a speedup to the t-SNE algorithm
            when performing a grid search.
            Acceptable formats: sparse SciPy ndarray, CuPy device ndarray,
            CSR/COO preferred other formats will go through conversion to CSR

        """
        cdef int n, p
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        if handle_ == NULL:
            raise ValueError("cuML Handle is Null! Terminating TSNE.")

        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        if is_sparse(X):

            self.X_m = SparseCumlArray(X, convert_to_dtype=cupy.float32,
                                       convert_format=False)
            n, p = self.X_m.shape
            self.sparse_fit = True

        # Handle dense inputs
        else:
            self.X_m, n, p, _ = \
                input_to_cuml_array(X, order='C', check_dtype=np.float32,
                                    convert_to_dtype=(np.float32
                                                      if convert_dtype
                                                      else None))

        if n <= 1:
            raise ValueError("There needs to be more than 1 sample to build "
                             "nearest the neighbors graph")

        self.n_neighbors = min(n, self.n_neighbors)
        if self.perplexity > n:
            warnings.warn("Perplexity = {} should be less than the "
                          "# of datapoints = {}.".format(self.perplexity, n))
            self.perplexity = n

        (knn_indices_m, knn_indices_ctype), (knn_dists_m, knn_dists_ctype) =\
            extract_knn_graph(knn_graph, convert_dtype, self.sparse_fit)

        cdef uintptr_t knn_indices_raw = knn_indices_ctype or 0
        cdef uintptr_t knn_dists_raw = knn_dists_ctype or 0

        # Prepare output embeddings
        self.embedding_ = CumlArray.zeros(
            (n, self.n_components),
            order="F",
            dtype=np.float32,
            index=self.X_m.index)

        cdef uintptr_t embed_ptr = self.embedding_.ptr

        # Find best params if learning rate method is adaptive
        if self.learning_rate_method=='adaptive' and (self.method=="barnes_hut"
                                                      or self.method=='fft'):
            logger.debug("Learning rate is adaptive. In TSNE paper, "
                         "it has been shown that as n->inf, "
                         "Barnes Hut works well if n_neighbors->30, "
                         "learning_rate->20000, early_exaggeration->24.")
            logger.debug("cuML uses an adpative method."
                         "n_neighbors decreases to 30 as n->inf. "
                         "Likewise for the other params.")
            if n <= 2000:
                self.n_neighbors = min(max(self.n_neighbors, 90), n)
            else:
                # A linear trend from (n=2000, neigh=100) to (n=60000,neigh=30)
                self.n_neighbors = max(int(102 - 0.0012 * n), 30)
            self.pre_learning_rate = max(n / 3.0, 1)
            self.post_learning_rate = self.pre_learning_rate
            self.early_exaggeration = 24.0 if n > 10000 else 12.0
            if logger.should_log_for(logger.level_debug):
                logger.debug("New n_neighbors = {}, learning_rate = {}, "
                             "exaggeration = {}"
                             .format(self.n_neighbors, self.pre_learning_rate,
                                     self.early_exaggeration))

        if self.method == 'barnes_hut':
            algo = TSNE_ALGORITHM.BARNES_HUT
        elif self.method == 'fft':
            algo = TSNE_ALGORITHM.FFT
        elif self.method == 'exact':
            algo = TSNE_ALGORITHM.EXACT
        else:
            raise ValueError("Allowed methods are 'exact', 'barnes_hut' and "
                             "'fft'.")

        cdef TSNEParams* params = <TSNEParams*> <size_t> \
            self._build_tsne_params(algo)

        cdef float kl_divergence = 0

        if self.sparse_fit:
            TSNE_fit_sparse(handle_[0],
                            <int*><uintptr_t>
                            self.X_m.indptr.ptr,
                            <int*><uintptr_t>
                            self.X_m.indices.ptr,
                            <float*><uintptr_t>
                            self.X_m.data.ptr,
                            <float*> embed_ptr,
                            <int> self.X_m.nnz,
                            <int> n,
                            <int> p,
                            <int*> knn_indices_raw,
                            <float*> knn_dists_raw,
                            <TSNEParams&> deref(params),
                            &kl_divergence)
        else:
            TSNE_fit(handle_[0],
                     <float*><uintptr_t> self.X_m.ptr,
                     <float*> embed_ptr,
                     <int> n,
                     <int> p,
                     <int64_t*> knn_indices_raw,
                     <float*> knn_dists_raw,
                     <TSNEParams&> deref(params),
                     &kl_divergence)

        self.handle.sync()
        free(params)

        self._kl_divergence_ = kl_divergence
        logger.debug("[t-SNE] KL divergence: {}".format(kl_divergence))
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
    def fit_transform(self, X, convert_dtype=True,
                      knn_graph=None) -> CumlArray:
        """
        Fit X into an embedded space and return that transformed output.
        """
        return self.fit(X, convert_dtype=convert_dtype,
                        knn_graph=knn_graph)._transform(X)

    def _transform(self, X) -> CumlArray:
        """
        Internal transform function to allow base wrappers default
        functionality to work
        """
        return self.embedding_

    def _build_tsne_params(self, algo):
        cdef long long seed = -1
        if self.random_state is not None:
            seed = self.random_state

        cdef TSNEParams* params = new TSNEParams()
        params.dim = <int> self.n_components
        params.n_neighbors = <int> self.n_neighbors
        params.theta = <float> self.angle
        params.epssq = <float> self.epssq
        params.perplexity = <float> self.perplexity
        params.perplexity_max_iter = <int> self.perplexity_max_iter
        params.perplexity_tol = <float> self.perplexity_tol
        params.early_exaggeration = <float> self.early_exaggeration
        params.late_exaggeration = <float> self.late_exaggeration
        params.exaggeration_iter = <int> self.exaggeration_iter
        params.min_gain = <float> self.min_gain
        params.pre_learning_rate = <float> self.pre_learning_rate
        params.post_learning_rate = <float> self.post_learning_rate
        params.max_iter = <int> self.n_iter
        params.min_grad_norm = <float> self.min_grad_norm
        params.pre_momentum = <float> self.pre_momentum
        params.post_momentum = <float> self.post_momentum
        params.random_state = <long long> seed
        params.verbosity = <int> self.verbose
        params.initialize_embeddings = <bool> True
        params.square_distances = <bool> self.square_distances
        params.algorithm = algo

        # metric
        metric_parsing = {
            "l2": DistanceType.L2SqrtExpanded,
            "euclidean": DistanceType.L2SqrtExpanded,
            "sqeuclidean": DistanceType.L2Expanded,
            "cityblock": DistanceType.L1,
            "l1": DistanceType.L1,
            "manhattan": DistanceType.L1,
            "minkowski": DistanceType.LpUnexpanded,
            "chebyshev": DistanceType.Linf,
            "cosine": DistanceType.CosineExpanded,
            "correlation": DistanceType.CorrelationExpanded
        }

        if self.metric.lower() in metric_parsing:
            params.metric = metric_parsing[self.metric.lower()]
        else:
            raise ValueError("Invalid value for metric: {}"
                             .format(self.metric))

        if self.metric_params is None:
            params.p = <float> 2.0
        else:
            params.p = <float>self.metric_params.get('p')

        return <size_t> params

    @property
    def kl_divergence_(self):
        if self.method == 'barnes_hut':
            warnings.warn("The calculation of the Kullback-Leibler "
                          "divergence is still an experimental feature "
                          "while using the Barnes Hut algorithm.")
        return self._kl_divergence_

    @kl_divergence_.setter
    def kl_divergence_(self, value):
        self._kl_divergence_ = value

    def __del__(self):

        if hasattr(self, "embedding_"):
            del self.embedding_

    def __getstate__(self):
        state = self.__dict__.copy()
        if "handle" in state:
            del state["handle"]
        return state

    def __setstate__(self, state):
        super(TSNE, self).__init__(handle=None,
                                   verbose=state['verbose'])
        self.__dict__.update(state)
        return state

    def get_param_names(self):
        return super().get_param_names() + [
            "n_components",
            "perplexity",
            "early_exaggeration",
            "late_exaggeration",
            "learning_rate",
            "n_iter",
            "n_iter_without_progress",
            "min_grad_norm",
            "metric",
            "metric_params",
            "init",
            "random_state",
            "method",
            "angle",
            "learning_rate_method",
            "n_neighbors",
            "perplexity_max_iter",
            "exaggeration_iter",
            "pre_momentum",
            "post_momentum",
            "square_distances"
        ]
