# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import warnings

import cupy
import numpy as np
import sklearn
from packaging.version import Version

import cuml.internals
from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.common.sparse_utils import is_sparse
from cuml.common.sparsefuncs import extract_knn_graph
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.base import Base
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import CMajorInputTagMixin, SparseInputTagMixin
from cuml.internals.utils import check_random_seed

from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.internals cimport logger
from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/manifold/tsne.h" namespace "ML" nogil:

    enum TSNE_ALGORITHM:
        EXACT = 0,
        BARNES_HUT = 1,
        FFT = 2

    enum TSNE_INIT:
        RANDOM = 0,
        PCA = 1

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
        logger.level_enum verbosity,
        TSNE_INIT init,
        bool square_distances,
        DistanceType metric,
        float p,
        TSNE_ALGORITHM algorithm


cdef extern from "cuml/manifold/tsne.h" namespace "ML" nogil:

    cdef void TSNE_fit(
        handle_t &handle,
        float *X,
        float *Y,
        int n,
        int p,
        int64_t* knn_indices,
        float* knn_dists,
        TSNEParams &params,
        float* kl_div,
        int* n_iter) except +

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
        float* kl_div,
        int* n_iter) except +


# Changed in scikit-learn version 1.5: Parameter name changed from n_iter to max_iter.
if Version(sklearn.__version__) >= Version("1.5.0"):
    _SKLEARN_N_ITER_PARAM = "max_iter"
else:
    _SKLEARN_N_ITER_PARAM = "n_iter"

_SUPPORTED_METRICS = {
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

_SUPPORTED_METHODS = {
    "barnes_hut": TSNE_ALGORITHM.BARNES_HUT,
    "exact": TSNE_ALGORITHM.EXACT,
    "fft": TSNE_ALGORITHM.FFT,
}

_SUPPORTED_INITS = {
    "random": TSNE_INIT.RANDOM,
    "pca": TSNE_INIT.PCA,
}


def _check_numeric(estimator, name, gt=None, ge=None, lt=None, le=None):
    """Check that a numeric parameter `name` is within valid bounds"""
    value = getattr(estimator, name)
    cls_name = type(estimator).__name__
    if gt is not None and value <= gt:
        raise ValueError(f"{cls_name} requires `{name} > {gt}`, got {value}")
    if ge is not None and value < ge:
        raise ValueError(f"{cls_name} requires `{name} >= {ge}`, got {value}")
    if lt is not None and value >= lt:
        raise ValueError(f"{cls_name} requires `{name} < {lt}`, got {value}")
    if le is not None and value > le:
        raise ValueError(f"{cls_name} requires `{name} <= {le}`, got {value}")
    return value


def _check_mapping(estimator, name, mapping):
    """Check that a parameter `name` contained within a valid mapping"""
    value = getattr(estimator, name)
    cls_name = type(estimator).__name__
    try:
        return mapping[value]
    except KeyError:
        raise ValueError(
            f"{cls_name} expects `{name}` to be one of {sorted(mapping)}, got {value}"
        ) from None


cdef _init_params(self, int n_samples, TSNEParams &params):
    """Validate TSNE parameters and initialize a TSNEParams instance."""
    if (n_components := self.n_components) != 2:
        raise ValueError(
            f"Currently TSNE only supports n_components = 2, got {self.n_components}"
        )
    perplexity = _check_numeric(self, "perplexity", gt=0)
    early_exaggeration = _check_numeric(self, "early_exaggeration", ge=1.0)
    late_exaggeration = _check_numeric(self, "late_exaggeration", ge=1.0)
    learning_rate = _check_numeric(self, "learning_rate", gt=0)
    adaptive_learning = _check_mapping(
        self, "learning_rate_method", {"adaptive": True, "none": False, None: False}
    )
    n_iter = _check_numeric(self, "n_iter", gt=0)
    min_grad_norm = _check_numeric(self, "min_grad_norm", ge=0)
    angle = _check_numeric(self, "angle", ge=0, le=1)
    n_neighbors = _check_numeric(self, "n_neighbors", gt=0)
    perplexity_max_iter = _check_numeric(self, "perplexity_max_iter", ge=0)
    exaggeration_iter = _check_numeric(self, "exaggeration_iter", ge=0)
    pre_momentum = _check_numeric(self, "pre_momentum", gt=0, lt=1)
    post_momentum = _check_numeric(self, "post_momentum", gt=0, lt=1)
    init = _check_mapping(self, "init", _SUPPORTED_INITS)
    algo = _check_mapping(self, "method", _SUPPORTED_METHODS)
    metric = _check_mapping(self, "metric", _SUPPORTED_METRICS)

    if n_samples < 2:
        raise ValueError("TSNE requires >= 2 samples")

    exaggeration_iter = min(exaggeration_iter, self.n_iter)
    if n_neighbors > 1023:
        warnings.warn(
            f"n_neighbors ({n_neighbors}) should be < 1024, "
            "thresholding n_neighbors to 1023"
        )
        n_neighbors = 1023
    n_neighbors = min(n_neighbors, n_samples)

    if perplexity > n_samples:
        warnings.warn(
            f"perplexity ({perplexity}) should be less than n_samples, "
            f"thresholding perplexity to {n_samples}"
        )
        perplexity = n_samples

    if adaptive_learning and algo is not TSNE_ALGORITHM.EXACT:
        # Adjust parameters when using adaptive learning
        if n_samples <= 2000:
            n_neighbors = min(max(n_neighbors, 90), n_samples)
        else:
            # A linear trend from (n=2000, neigh=100) to (n=60000,neigh=30)
            n_neighbors = max(int(102 - 0.0012 * n_samples), 30)

        pre_learning_rate = max(n_samples / 3.0, 1)
        post_learning_rate = pre_learning_rate
        early_exaggeration = 24.0 if n_samples > 10000 else 12.0
    else:
        pre_learning_rate = learning_rate
        post_learning_rate = learning_rate * 2

    cdef long long seed = (
        -1 if self.random_state is None
        else check_random_seed(self.random_state)
    )

    params.dim = n_components
    params.n_neighbors = n_neighbors
    params.theta = angle
    params.epssq = 0.0025
    params.perplexity = perplexity
    params.perplexity_max_iter = perplexity_max_iter
    params.perplexity_tol = 1e-5
    params.early_exaggeration = early_exaggeration
    params.late_exaggeration = late_exaggeration
    params.exaggeration_iter = exaggeration_iter
    params.min_gain = 0.01
    params.pre_learning_rate = pre_learning_rate
    params.post_learning_rate = post_learning_rate
    params.max_iter = n_iter
    params.min_grad_norm = min_grad_norm
    params.pre_momentum = pre_momentum
    params.post_momentum = post_momentum
    params.random_state = seed
    params.verbosity = self.verbose
    params.square_distances = self.square_distances
    params.algorithm = algo
    params.init = init
    params.metric = metric
    params.p = (self.metric_params or {}).get("p", 2.0)


class TSNE(Base,
           InteropMixin,
           CMajorInputTagMixin,
           SparseInputTagMixin):
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
    init : str 'random' or 'pca' (default 'random')
        Currently supports random or pca initialization.
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
    precomputed_knn : array / sparse array / tuple, optional (device or host)
        Either one of a tuple (indices, distances) of
        arrays of shape (n_samples, n_neighbors), a pairwise distances
        dense array of shape (n_samples, n_samples) or a KNN graph
        sparse array (preferably CSR/COO). This feature allows
        the precomputation of the KNN outside of TSNE
        and also allows the use of a custom distance function. This function
        should match the metric used to train the TSNE embeedings.
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
    embedding_ : array
        Stores the embedding vectors.
    kl_divergence_ : float
        Kullback-Leibler divergence after optimization. An experimental
        feature at this time.
    learning_rate_ : float
        Effective learning rate.
    n_iter_ : int
        Number of iterations run.

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
    embedding_ = CumlArrayDescriptor(order="F")

    _cpu_class_path = "sklearn.manifold.TSNE"

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
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
            "square_distances",
            "precomputed_knn"
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.n_components != 2:
            raise UnsupportedOnGPU("Only `n_components=2` is supported")

        # Our barnes_hut implementation can sometimes hang, see #3865 and #3360.
        # fft should be at least as good, and doesn't have this issue.
        method = {"exact": "exact", "barnes_hut": "fft"}.get(model.method, None)
        if method is None:
            raise UnsupportedOnGPU(f"`method={model.method!r}` is not supported")

        if not (isinstance(model.init, str) and model.init in _SUPPORTED_INITS):
            raise UnsupportedOnGPU(f"`init={model.init!r}` is not supported")

        if not (isinstance(model.metric, str) and model.metric in _SUPPORTED_METRICS):
            raise UnsupportedOnGPU(f"`metric={model.metric!r}` is not supported")

        params = {
            "n_components": model.n_components,
            "perplexity": model.perplexity,
            "early_exaggeration": model.early_exaggeration,
            "n_iter_without_progress": model.n_iter_without_progress,
            "min_grad_norm": model.min_grad_norm,
            "metric": model.metric,
            "metric_params": model.metric_params,
            "init": model.init,
            "random_state": model.random_state,
            "method": method,
        }
        if model.learning_rate != "auto":
            # For now have `learning_rate="auto"` just use cuml's default
            params["learning_rate"]: model.learning_rate

        if (max_iter := getattr(model, _SKLEARN_N_ITER_PARAM, None)) is not None:
            params["n_iter"] = max_iter

        return params

    def _params_to_cpu(self):
        method = "exact" if self.method == "Exact" else "barnes_hut"

        params = {
            "n_components": self.n_components,
            "perplexity": self.perplexity,
            "early_exaggeration": self.early_exaggeration,
            "learning_rate": self.learning_rate,
            "n_iter_without_progress": self.n_iter_without_progress,
            "min_grad_norm": self.min_grad_norm,
            "metric": self.metric,
            "metric_params": self.metric_params,
            "init": self.init,
            "random_state": self.random_state,
            "method": method,
            _SKLEARN_N_ITER_PARAM: self.n_iter,
        }
        return params

    def _attrs_from_cpu(self, model):
        return {
            "embedding_": to_gpu(model.embedding_),
            "kl_divergence_": to_gpu(model.kl_divergence_),
            "learning_rate_": model.learning_rate_,
            "n_iter_": model.n_iter_,
            **super()._attrs_from_cpu(model)
        }

    def _attrs_to_cpu(self, model):
        return {
            "embedding_": to_cpu(self.embedding_),
            "kl_divergence_": to_cpu(self.kl_divergence_),
            "learning_rate_": self.learning_rate_,
            "n_iter_": self.n_iter_,
            **super()._attrs_to_cpu(model)
        }

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
                 random_state=None,
                 method='fft',
                 angle=0.5,
                 n_neighbors=90,
                 perplexity_max_iter=100,
                 exaggeration_iter=250,
                 pre_momentum=0.5,
                 post_momentum=0.8,
                 learning_rate_method='adaptive',
                 square_distances=True,
                 precomputed_knn=None,
                 verbose=False,
                 handle=None,
                 output_type=None):

        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)

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
        self.learning_rate_method = learning_rate_method
        self.square_distances = square_distances
        self.precomputed_knn = precomputed_knn

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        # Exposed to support sklearn's `get_feature_names_out`
        return self.embedding_.shape[1]

    @generate_docstring(skip_parameters_heading=True,
                        X='dense_sparse',
                        convert_dtype_cast='np.float32')
    def fit(self, X, y=None, *, convert_dtype=True, knn_graph=None) -> "TSNE":
        """
        Fit X into an embedded space.

        Parameters
        ----------
        knn_graph : array / sparse array / tuple, optional (device or host)
        Either one of a tuple (indices, distances) of
        arrays of shape (n_samples, n_neighbors), a pairwise distances
        dense array of shape (n_samples, n_samples) or a KNN graph
        sparse array (preferably CSR/COO). This feature allows
        the precomputation of the KNN outside of TSNE
        and also allows the use of a custom distance function. This function
        should match the metric used to train the TSNE embeedings.
        Takes precedence over the precomputed_knn parameter.
        """
        cdef int n_samples, n_features
        cdef uintptr_t X_ptr = 0
        cdef uintptr_t X_indptr_ptr = 0
        cdef uintptr_t X_indices_ptr = 0
        cdef int X_nnz = 0
        cdef bool sparse_fit = is_sparse(X)

        # Normalize input X
        if sparse_fit:
            X_m = SparseCumlArray(
                X, convert_to_dtype=cupy.float32, convert_format=False
            )
            n_samples, n_features = X_m.shape
            X_ptr = <uintptr_t>X_m.data.ptr
            X_indptr_ptr = <uintptr_t>X_m.indptr.ptr
            X_indices_ptr = <uintptr_t>X_m.indices.ptr
            X_nnz = X_m.nnz
        else:
            X_m, n_samples, n_features, _ = input_to_cuml_array(
                X, order='F', check_dtype=np.float32,
                convert_to_dtype=(np.float32 if convert_dtype else None)
            )
            X_ptr = X_m.ptr

        # Initialize TSNEParams
        cdef TSNEParams params
        _init_params(self, n_samples, params)

        # Normalize precomputed knn graph if provided
        cdef uintptr_t knn_dists_ptr = 0
        cdef uintptr_t knn_indices_ptr = 0
        if knn_graph is None:
            knn_graph = self.precomputed_knn
        if knn_graph is not None:
            knn_indices, knn_dists = extract_knn_graph(knn_graph, params.n_neighbors)

            if sparse_fit:
                # Sparse fitting requires the indices to be int32
                knn_indices = input_to_cuml_array(
                    knn_indices, convert_to_dtype=np.int32
                ).array

            knn_dists_ptr = knn_dists.ptr
            knn_indices_ptr = knn_indices.ptr

        # Allocate output array
        embedding = CumlArray.zeros(
            (n_samples, self.n_components),
            order="F",
            dtype=np.float32,
            index=X_m.index,
        )
        cdef uintptr_t embed_ptr = embedding.ptr

        # Execute fit
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef float kl_divergence = 0
        cdef int n_iter = 0

        with nogil:
            if sparse_fit:
                TSNE_fit_sparse(
                    handle_[0],
                    <int*>X_indptr_ptr,
                    <int*>X_indices_ptr,
                    <float*>X_ptr,
                    <float*>embed_ptr,
                    X_nnz,
                    n_samples,
                    n_features,
                    <int*>knn_indices_ptr,
                    <float*>knn_dists_ptr,
                    params,
                    &kl_divergence,
                    &n_iter,
                )
            else:
                TSNE_fit(
                    handle_[0],
                    <float*>X_ptr,
                    <float*>embed_ptr,
                    n_samples,
                    n_features,
                    <int64_t*> knn_indices_ptr,
                    <float*> knn_dists_ptr,
                    params,
                    &kl_divergence,
                    &n_iter,
                )
        self.handle.sync()

        # Store fitted attributes
        self._kl_divergence_ = kl_divergence
        self.n_iter_ = n_iter
        self.learning_rate_ = params.pre_learning_rate
        self.embedding_ = embedding

        return self

    @generate_docstring(convert_dtype_cast='np.float32',
                        return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Embedding of the \
                                                       data in \
                                                       low-dimensional space.',
                                       'shape': '(n_samples, n_components)'})
    @cuml.internals.api_base_fit_transform()
    def fit_transform(self, X, y=None, *, convert_dtype=True, knn_graph=None) -> CumlArray:
        """
        Fit X into an embedded space and return that transformed output.
        """
        self.fit(X, convert_dtype=convert_dtype, knn_graph=knn_graph)
        return self.embedding_

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
