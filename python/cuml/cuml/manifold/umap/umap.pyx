#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import ctypes
import warnings

import cupy as cp
import cupyx.scipy.sparse
import joblib
import numpy as np
import scipy.sparse

from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.common.sparse_utils import is_sparse
from cuml.common.sparsefuncs import extract_knn_graph
from cuml.internals import logger, reflect
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.base import Base, get_handle
from cuml.internals.input_utils import input_to_cuml_array, is_array_like
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mem_type import MemoryType
from cuml.internals.mixins import CMajorInputTagMixin, SparseInputTagMixin
from cuml.internals.utils import check_random_seed

from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibraft.common.handle cimport handle_t
from rmm.librmm.device_buffer cimport device_buffer
from rmm.librmm.per_device_resource cimport get_current_device_resource
from rmm.pylibrmm.device_buffer cimport DeviceBuffer

cimport cuml.manifold.umap.lib as lib
from cuml.metrics.distance_type cimport DistanceType


def _joblib_hash(X):
    """A thin shim around joblib.hash"""
    if scipy.sparse.issparse(X):
        # XXX: joblib.hash doesn't special case sparse inputs, meaning that
        # it's sensitive to what should be irrelevant internal state. For now
        # we trigger a cached attribute to ensure state is always fully filled
        # in so hashing is consistent. This is a relatively cheap operation and
        # has no measurable impact on performance.
        X.has_sorted_indices
    return joblib.hash(X)


def find_ab_params(spread=1.0, min_dist=0.1):
    """Fit a & b parameters for UMAP.

    Selects `a` and `b` for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.

    Borrowed from upstream umap: https://github.com/lmcinnes/umap.

    Parameters
    ----------
    spread: float (optional, default 1.0)
        The effective scale of embedded points.
    min_dist: float (optional, default 0.1)
        The effective minimum distance between embedded points.

    Returns
    -------
    a, b : float
        The `a` and `b` parameters for `UMAP`.
    """
    from scipy.optimize import curve_fit

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, _ = curve_fit(curve, xv, yv)
    return params[0], params[1]


cdef class RaftCOO:
    """A wrapper around a `raft::sparse::COO`"""
    cdef unique_ptr[lib.COO] ptr

    def __dealloc__(self):
        self.ptr.reset(NULL)

    @staticmethod
    cdef RaftCOO from_ptr(unique_ptr[lib.COO]& ptr):
        """Create a new instance, taking ownership of an existing `unique_ptr[COO]`"""
        cdef RaftCOO self = RaftCOO.__new__(RaftCOO)
        self.ptr = move(ptr)
        return self

    @staticmethod
    cdef RaftCOO from_cupy_coo(handle, arr):
        """Create a new instance as a copy of a `cupyx.scipy.sparse.coo_matrix"""
        def copy_from_cupy(dst_ptr, src, dtype):
            src = src.astype(dtype, copy=False)
            size = src.size * src.dtype.itemsize
            dest_mem = cp.cuda.UnownedMemory(dst_ptr, size, owner=None)
            dest_mptr = cp.cuda.memory.MemoryPointer(dest_mem, 0)
            src_mem = cp.cuda.UnownedMemory(src.data.ptr, size, owner=None)
            src_mptr = cp.cuda.memory.MemoryPointer(src_mem, 0)
            dest_mptr.copy_from_device(src_mptr, size)

        cdef RaftCOO self = RaftCOO.__new__(RaftCOO)
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef lib.COO* coo = new lib.COO(handle_.get_stream())
        self.ptr.reset(coo)
        coo.allocate(arr.nnz, arr.shape[0], False, handle_.get_stream())
        handle_.sync_stream()

        copy_from_cupy(<uintptr_t>coo.vals(), arr.data, np.float32)
        copy_from_cupy(<uintptr_t>coo.rows(), arr.row, np.int32)
        copy_from_cupy(<uintptr_t>coo.cols(), arr.col, np.int32)

        return self

    def view_cupy_coo(self):
        """Create a new `cupyx.scipy.sparse.coo_matrix` as a view of this COO"""
        cdef lib.COO* coo = self.get()

        def view_as_cupy(ptr, dtype):
            dtype = np.dtype(dtype)
            mem = cp.cuda.UnownedMemory(ptr, (coo.nnz * dtype.itemsize), owner=self)
            memptr = cp.cuda.memory.MemoryPointer(mem, 0)
            return cp.ndarray(coo.nnz, dtype=dtype, memptr=memptr)

        vals = view_as_cupy(<uintptr_t>coo.vals(), np.float32)
        rows = view_as_cupy(<uintptr_t>coo.rows(), np.int32)
        cols = view_as_cupy(<uintptr_t>coo.cols(), np.int32)

        return cupyx.scipy.sparse.coo_matrix(((vals, (rows, cols))))

    cdef inline lib.COO* get(self) noexcept nogil:
        return self.ptr.get()


cdef copy_raft_host_coo_to_scipy_coo(lib.HostCOO &coo):
    """Copy a `raft::host_coo_matrix` to a `scipy.sparse.coo_matrix`"""
    nnz = coo.get_nnz()
    i32 = ctypes.POINTER(ctypes.c_int)
    f32 = ctypes.POINTER(ctypes.c_float)
    vals = np.ctypeslib.as_array(ctypes.cast(<uintptr_t>coo.vals(), f32), shape=(nnz,))
    rows = np.ctypeslib.as_array(ctypes.cast(<uintptr_t>coo.rows(), i32), shape=(nnz,))
    cols = np.ctypeslib.as_array(ctypes.cast(<uintptr_t>coo.cols(), i32), shape=(nnz,))
    return scipy.sparse.coo_matrix((vals.copy(), (rows.copy(), cols.copy())))


_BUILD_ALGOS = {"auto", "brute_force_knn", "nn_descent"}

_INITS = {"random": 0, "spectral": 1}

_TARGET_METRICS = {
    "euclidean": lib.MetricType.EUCLIDEAN,
    "l2": lib.MetricType.EUCLIDEAN,
    "categorical": lib.MetricType.CATEGORICAL,
}

_METRICS = {
    "l2": DistanceType.L2SqrtExpanded,
    "euclidean": DistanceType.L2SqrtExpanded,
    "sqeuclidean": DistanceType.L2Expanded,
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

_SUPPORTED_METRICS = {
    "nn_descent": {
        "sparse": frozenset(),
        "dense": frozenset((
            DistanceType.L2SqrtExpanded,
            DistanceType.L2Expanded,
            DistanceType.CosineExpanded,
        ))
    },
    "brute_force_knn": {
        "sparse": frozenset((
            DistanceType.Canberra,
            DistanceType.CorrelationExpanded,
            DistanceType.CosineExpanded,
            DistanceType.HammingUnexpanded,
            DistanceType.HellingerExpanded,
            DistanceType.JaccardExpanded,
            DistanceType.L1,
            DistanceType.L2SqrtExpanded,
            DistanceType.L2Expanded,
            DistanceType.Linf,
            DistanceType.LpUnexpanded,
        )),
        "dense": frozenset((
            DistanceType.Canberra,
            DistanceType.CorrelationExpanded,
            DistanceType.CosineExpanded,
            DistanceType.HammingUnexpanded,
            DistanceType.HellingerExpanded,
            # DistanceType.JaccardExpanded,  # not supported
            DistanceType.L1,
            DistanceType.L2SqrtExpanded,
            DistanceType.L2Expanded,
            DistanceType.Linf,
            DistanceType.LpUnexpanded,
        ))
    }
}


def coerce_metric(metric, sparse=False, build_algo="brute_force_knn"):
    """Coerce a metric string to a `DistanceType`.

    Also checks that the metric is valid and supported.
    """
    if not isinstance(metric, str):
        raise TypeError(f"Expected `metric` to be a str, got {type(metric).__name__}")

    try:
        out = _METRICS[metric.lower()]
    except KeyError:
        raise ValueError(f"Invalid value for metric: {metric!r}")

    kind = "sparse" if sparse else "dense"
    supported = _SUPPORTED_METRICS[build_algo][kind]
    if out not in supported:
        raise NotImplementedError(
            f"Metric {metric!r} not supported for {kind} inputs with {build_algo=}"
        )

    return out


cdef init_params(self, lib.UMAPParams &params, n_rows, is_sparse=False, is_fit=True):
    """Initialize a UMAPParams instance from a UMAP model.

    This would be a method, except cdef methods aren't allowed on non cdef classes.

    Parameters
    ----------
    self : UMAP
        The UMAP model.
    params : lib.UMAPParams
        The params to instantiate.
    n_rows : int
        The number of rows in X for either fit or transform.
    is_sparse : bool
        Whether X is sparse for either fit or transform.
    is_fit : bool
        Whether these parameters are for a fit call.
    """
    if self.n_components < 1:
        raise ValueError(f"Expected `n_components >= 1`, got {self.n_components}")
    if self.n_neighbors < 1:
        raise ValueError(f"Expected `n_neighbors >= 1`, got {self.n_neighbors}")
    if self.min_dist > self.spread:
        raise ValueError(f"Expected min_dist ({self.min_dist}) <= spread ({self.spread})")

    if is_fit:
        build_algo = self.build_algo

        # Compute and stash some inferred params when fitting
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a, self._b = self.a, self.b
        self._n_neighbors = min(n_rows, self.n_neighbors)
        if self._n_neighbors != self.n_neighbors:
            warnings.warn(
                f"n_neighbors ({self.n_neighbors}) is larger than n_samples ({n_rows}) "
                f"truncating to {self._n_neighbors}"
            )
    else:
        # Only brute_force_knn supported for transform
        build_algo = "brute_force_knn"
        # Use the larger of the input shapes when inferring deterministic behavior
        n_rows = max(self._raw_data.shape[0], n_rows)

    if build_algo == "auto":
        if self.random_state is not None:
            # TODO: for now, users should be able to see the same results
            # as previous version (i.e. running brute force knn) when they
            # explicitly pass random_state
            # https://github.com/rapidsai/cuml/issues/5985
            build_algo ="brute_force_knn"
        elif n_rows <= 50_000 or is_sparse:
            # brute force is faster for small datasets
            build_algo = "brute_force_knn"
        else:
            build_algo = "nn_descent"
        logger.debug(f"Building knn graph using build_algo={build_algo!r}")
    elif build_algo not in _BUILD_ALGOS:
        raise ValueError(
            f"Expected `build_algo` to be one of {list(_BUILD_ALGOS)}, "
            f"got {build_algo!r}"
        )

    if build_algo == "nn_descent" and n_rows < 150:
        # https://github.com/rapidsai/cuvs/issues/184
        warnings.warn(
            "using build_algo='nn_descent' on a small dataset (< 150 samples) "
            "is unstable"
        )

    if build_algo == "nn_descent" and self.random_state is not None:
        warnings.warn("build_algo='nn_descent' is not deterministic. Please use "
                      "build_algo='brute_force_knn' instead with random_state set.")

    params.n_neighbors = self._n_neighbors
    params.n_components = self.n_components
    params.n_epochs = self.n_epochs or 0
    params.learning_rate = self.learning_rate
    params.initial_alpha = self.learning_rate
    params.min_dist = self.min_dist
    params.spread = self.spread
    params.set_op_mix_ratio = self.set_op_mix_ratio
    params.local_connectivity = self.local_connectivity
    params.repulsion_strength = self.repulsion_strength
    params.negative_sample_rate = self.negative_sample_rate
    params.transform_queue_size = self.transform_queue_size
    params.verbosity = self._verbose_level
    params.a = self._a
    params.b = self._b
    params.target_n_neighbors = self.target_n_neighbors
    params.target_weight = self.target_weight
    params.metric = coerce_metric(self.metric, sparse=is_sparse, build_algo=build_algo)
    params.p = (self.metric_kwds or {}).get("p", 2.0)
    params.random_state = check_random_seed(self.random_state)

    # deterministic if a random_state provided or when run on very small inputs
    params.deterministic = self.random_state is not None or n_rows < 300

    if is_array_like(self.init):
        params.init = 2
    elif self.init in _INITS:
        params.init = _INITS[self.init]
    else:
        raise ValueError(
            f"Expected `init` to be an array or one of {list(_INITS)}, "
            f"got {self.init!r}"
        )

    if self.target_metric in _TARGET_METRICS:
        params.target_metric = _TARGET_METRICS[self.target_metric]
    else:
        raise ValueError(
            f"Expected `target_metric` to be one of {list(_TARGET_METRICS)}, "
            f"got {self.target_metric!r}"
        )

    if self.callback is not None:
        params.callback = (
            <lib.GraphBasedDimRedCallback*><uintptr_t>self.callback.get_native_callback()
        )

    if build_algo == "brute_force_knn":
        params.build_algo = lib.graph_build_algo.BRUTE_FORCE_KNN
    else:
        params.build_algo = lib.graph_build_algo.NN_DESCENT

        build_kwds = self.build_kwds or {}
        n_clusters = build_kwds.get("nnd_n_clusters", 1)
        overlap_factor = build_kwds.get("nnd_overlap_factor", 2)
        max_iterations = build_kwds.get("nnd_max_iterations", 20)
        termination_threshold = build_kwds.get("nnd_termination_threshold", 0.0001)
        graph_degree = build_kwds.get("nnd_graph_degree", 64)
        intermediate_graph_degree = build_kwds.get("nnd_intermediate_graph_degree", 128)

        if n_clusters < 1:
            raise ValueError(f"Expected `nnd_n_clusters >= 1`, got {n_clusters}")
        elif n_clusters > 1 and overlap_factor >= n_clusters:
            raise ValueError(
                f"`nnd_n_clusters > 1` requires `nnd_n_clusters ({n_clusters}) > "
                f"nnd_overlap_factor ({overlap_factor})`"
            )

        if graph_degree < self._n_neighbors:
            logger.warn(
                f"build_algo='nn_descent' requires `nnd_graph_degree >= n_neighbors`. "
                f"Setting nnd_graph_degree to {self._n_neighbors}"
            )
            graph_degree = self._n_neighbors

        if intermediate_graph_degree < graph_degree:
            logger.warn(
                f"build_algo='nn_descent' requires `nnd_intermediate_graph_degree >= "
                f"nnd_graph_degree`. Setting nnd_intermediate_graph_degree to "
                f"{graph_degree}"
            )
            intermediate_graph_degree = graph_degree

        params.build_params.n_clusters = n_clusters
        params.build_params.overlap_factor = overlap_factor
        params.build_params.nnd.max_iterations = max_iterations
        params.build_params.nnd.termination_threshold = termination_threshold
        params.build_params.nnd.graph_degree = graph_degree
        params.build_params.nnd.intermediate_graph_degree = intermediate_graph_degree


class UMAP(Base, InteropMixin, CMajorInputTagMixin, SparseInputTagMixin):
    """Uniform Manifold Approximation and Projection

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
    metric: string (default='euclidean').
        Distance metric to use. Supported distances are ['l1, 'cityblock',
        'taxicab', 'manhattan', 'euclidean', 'l2', 'sqeuclidean', 'canberra',
        'minkowski', 'chebyshev', 'linf', 'cosine', 'correlation', 'hellinger',
        'hamming', 'jaccard']
        Metrics that take arguments (such as minkowski) can have arguments
        passed via the metric_kwds dictionary.
        Note: The 'jaccard' distance metric is only supported for sparse
        inputs.
    metric_kwds: dict (optional, default=None)
        Metric argument
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
        * An array-like with initial embedding positions.

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
    target_n_neighbors: int (optional, default=-1)
        The number of nearest neighbors to use to construct the target
        simplicial set. If set to -1 use the ``n_neighbors`` value.
    target_metric: string or callable (optional, default='categorical')
        The metric used to measure distance for a target array when using
        supervised dimension reduction. By default this is 'categorical'
        which will measure distance in terms of whether categories match
        or are different. Furthermore, if semi-supervised is required
        target values of -1 will be treated as unlabelled under the
        'categorical' metric. If the target array takes continuous values
        (e.g. for a regression problem) then metric of 'l1' or 'l2' is
        probably more appropriate.
    target_weight: float (optional, default=0.5)
        Weighting factor between data topology and target topology. A
        value of 0.0 weights predominantly on data, a value of 1.0 places
        a strong emphasis on target. The default of 0.5 balances the
        weighting equally between data and target.
    hash_input: bool, optional (default = False)
        UMAP can hash the training input so that exact embeddings
        are returned when transform is called on the same data upon
        which the model was trained. This enables consistent
        behavior between calling ``model.fit_transform(X)`` and
        calling ``model.fit(X).transform(X)``. Note that the CPU-based
        UMAP reference implementation does this by default. This
        feature is made optional in the GPU version due to the
        significant overhead in copying memory to the host for
        computing the hash.
    precomputed_knn : array / sparse array / tuple, optional (device or host)
        Either one of a tuple (indices, distances) of
        arrays of shape (n_samples, n_neighbors), a pairwise distances
        dense array of shape (n_samples, n_samples) or a KNN graph
        sparse array (preferably CSR/COO). This feature allows
        the precomputation of the KNN outside of UMAP
        and also allows the use of a custom distance function. This function
        should match the metric used to train the UMAP embeedings.
    random_state : int, RandomState instance or None, optional (default=None)
        Seed used by the random number generator for embedding initialization
        and optimizer sampling. Setting a random_state enables reproducible
        embeddings, but at the cost of slower training and increased memory
        usage. This is because high parallelism during optimization involves
        non-deterministic floating-point addition ordering.

        Note: Explicitly setting ``build_algo='nn_descent'`` will break
        reproducibility, as NN Descent produces non-deterministic KNN graphs.
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

    handle : cuml.Handle or None, default=None

        .. deprecated:: 26.02
            The `handle` argument was deprecated in 26.02 and will be removed
            in 26.04. There's no need to pass in a handle, cuml now manages
            this resource automatically. To configure multi-device execution,
            please use the `device_ids` parameter instead.

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    build_algo: string (default='auto')
        How to build the knn graph. Supported build algorithms are ['auto', 'brute_force_knn',
        'nn_descent']. 'auto' chooses to run with brute force knn if number of data rows is
        smaller than or equal to 50K. Otherwise, runs with nn descent.
    build_kwds: dict (optional, default=None)
        Dictionary of parameters to configure the build algorithm. Default values:

        - `nnd_graph_degree` (int, default=64): Graph degree used for NN Descent.
          Must be ≥ `n_neighbors`.

        - `nnd_intermediate_graph_degree` (int, default=128): Intermediate graph degree for
          NN Descent. Must be > `nnd_graph_degree`.

        - `nnd_max_iterations` (int, default=20): Max NN Descent iterations.

        - `nnd_termination_threshold` (float, default=0.0001): Stricter threshold leads to
          better convergence but longer runtime.

        - `nnd_n_clusters` (int, default=1): Number of clusters for data partitioning.
          Higher values reduce memory usage at the cost of accuracy. When `nnd_n_clusters > 1`,
          UMAP can process data larger than device memory.

        - `nnd_overlap_factor` (int, default=2): Number of clusters each data point belongs to.
          Valid only when `nnd_n_clusters > 1`. Must be < 'nnd_n_clusters'.

        Hints:

        - Increasing `nnd_graph_degree` and `nnd_max_iterations` may improve accuracy.

        - The ratio `nnd_overlap_factor / nnd_n_clusters` impacts memory usage.
          Approximately `(nnd_overlap_factor / nnd_n_clusters) * num_rows_in_entire_data`
          rows will be loaded onto device memory at once.  E.g., 2/20 uses less device
          memory than 2/10.

        - Larger `nnd_overlap_factor` results in better accuracy of the final knn graph.
          E.g. While using similar amount of device memory,
          `(nnd_overlap_factor / nnd_n_clusters)` = 4/20 will have better accuracy
          than 2/10 at the cost of performance.

        - Start with `nnd_overlap_factor = 2` and gradually increase (2->3->4 ...)
          for better accuracy.

        - Start with `nnd_n_clusters = 4` and increase (4 → 8 → 16...) for less GPU
          memory usage. This is independent from nnd_overlap_factor as long as
          'nnd_overlap_factor' < 'nnd_n_clusters'.
    device_ids : list[int], "all", or None, default=None
        The device IDs to use during fitting (only used when
        `build_algo=nn_descent` and `nnd_n_clusters > 1`). May be a list of
        ids, ``"all"`` (to use all available devices), or ``None`` (to fit
        using a single GPU only). Default is None.

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
    UMAP.

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
    embedding_ = CumlArrayDescriptor(order="C")

    _cpu_class_path = "umap.UMAP"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
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
            "metric_kwds",
            "precomputed_knn",
            "build_algo",
            "build_kwds",
            "device_ids",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if not ((isinstance(model.init, str) and model.init in _INITS) or
                is_array_like(model.init)):
            raise UnsupportedOnGPU(f"`init={model.init!r}` is not supported")

        try:
            coerce_metric(model.metric)
        except (ValueError, TypeError, NotImplementedError):
            raise UnsupportedOnGPU(f"`metric={model.metric!r}` is not supported")

        if model.target_metric not in _TARGET_METRICS:
            raise UnsupportedOnGPU(f"`target_metric={model.target_metric!r}` is not supported")

        if model.unique:
            raise UnsupportedOnGPU("`unique=True` is not supported")

        if model.densmap:
            raise UnsupportedOnGPU("`densmap=True` is not supported")

        precomputed_knn = model.precomputed_knn[:2]
        if all(item is None for item in precomputed_knn):
            precomputed_knn = None

        return {
            "n_neighbors": model.n_neighbors,
            "n_components": model.n_components,
            "metric": model.metric,
            "metric_kwds": model.metric_kwds,
            "n_epochs": model.n_epochs,
            "learning_rate": model.learning_rate,
            "min_dist": model.min_dist,
            "spread": model.spread,
            "set_op_mix_ratio": model.set_op_mix_ratio,
            "local_connectivity": model.local_connectivity,
            "repulsion_strength": model.repulsion_strength,
            "negative_sample_rate": model.negative_sample_rate,
            "transform_queue_size": model.transform_queue_size,
            "init": model.init,
            "a": model.a,
            "b": model.b,
            "target_n_neighbors": model.target_n_neighbors,
            "target_weight": model.target_weight,
            "target_metric": model.target_metric,
            "hash_input": True,
            "random_state": model.random_state,
            "precomputed_knn": precomputed_knn,
        }

    def _params_to_cpu(self):
        if (precomputed_knn := self.precomputed_knn) is None:
            precomputed_knn = (None, None, None)

        init = self.init
        if is_array_like(init):
            init = cp.asnumpy(init)

        return {
            "n_neighbors": self.n_neighbors,
            "n_components": self.n_components,
            "metric": self.metric,
            "metric_kwds": self.metric_kwds,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "min_dist": self.min_dist,
            "spread": self.spread,
            "set_op_mix_ratio": self.set_op_mix_ratio,
            "local_connectivity": self.local_connectivity,
            "repulsion_strength": self.repulsion_strength,
            "negative_sample_rate": self.negative_sample_rate,
            "transform_queue_size": self.transform_queue_size,
            "init": init,
            "a": self.a,
            "b": self.b,
            "target_n_neighbors": self.target_n_neighbors,
            "target_weight": self.target_weight,
            "target_metric": self.target_metric,
            "random_state": self.random_state,
            "precomputed_knn": precomputed_knn,
        }

    def _attrs_from_cpu(self, model):
        if scipy.sparse.issparse(model._raw_data):
            raw_data = SparseCumlArray(model._raw_data, convert_to_dtype=cp.float32)
        else:
            raw_data = to_gpu(model._raw_data)

        return {
            "embedding_": to_gpu(model.embedding_, order="C"),
            "graph_": model.graph_.tocoo(),
            "_raw_data": raw_data,
            "_input_hash": model._input_hash,
            "_sparse_data": model._sparse_data,
            "_a": model._a,
            "_b": model._b,
            "_n_neighbors": model._n_neighbors,
            "n_features_in_": model._raw_data.shape[1],
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        from umap.umap_ import DISCONNECTION_DISTANCES

        disconnection_distance = DISCONNECTION_DISTANCES.get(self.metric, np.inf)

        raw_data = self._raw_data.to_output("numpy")

        if (input_hash := getattr(self, "_input_hash", None)) is None:
            input_hash = _joblib_hash(raw_data)

        if (knn_dists := getattr(self, "_knn_dists", None)) is not None:
            knn_dists = to_cpu(knn_dists)

        if (knn_indices := getattr(self, "_knn_indices", None)) is not None:
            knn_indices = to_cpu(knn_indices)

        return {
            "embedding_": to_cpu(self.embedding_),
            "graph_": self.graph_.tocsr(),
            "graph_dists_": None,
            "_raw_data": raw_data,
            "_input_hash": input_hash,
            "_sparse_data": self._sparse_data,
            "_a": self._a,
            "_b": self._b,
            "_disconnection_distance": disconnection_distance,
            "_initial_alpha": self.learning_rate,
            "_n_neighbors": self._n_neighbors,
            "_supervised": self._supervised,
            "_small_data": False,
            "_knn_dists": knn_dists,
            "_knn_indices": knn_indices,
            # XXX: umap.UMAP requires _knn_search_index to transform on new data.
            # This is an instance of pynndescent.NNDescent, which can _currently_
            # only be recreated by rerunning the nndescent algorithm, effectively
            # repeating any work already done on GPU. Instead, we opt to set
            # _knn_search_index to None (which is allowed). In this case
            # calling umap.UMAP.transform on _new_ data will result in
            # a nicer NotImplementedError being raised informing the user
            # that that's not supported on the instance.
            "_knn_search_index": None,
            **super()._attrs_to_cpu(model),
        }

    def _sync_attrs_to_cpu(self, model):
        super()._sync_attrs_to_cpu(model)
        # _validate_parameters constructs the rest of umap.UMAP's internal state
        model._validate_parameters()

    def __init__(
        self,
        *,
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
        a=None,
        b=None,
        target_n_neighbors=-1,
        target_weight=0.5,
        target_metric="categorical",
        hash_input=False,
        random_state=None,
        precomputed_knn=None,
        callback=None,
        build_algo="auto",
        build_kwds=None,
        device_ids=None,
        handle=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)

        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.metric_kwds = metric_kwds
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.min_dist = min_dist
        self.spread = spread
        self.set_op_mix_ratio = set_op_mix_ratio
        self.local_connectivity = local_connectivity
        self.repulsion_strength = repulsion_strength
        self.negative_sample_rate = negative_sample_rate
        self.transform_queue_size = transform_queue_size
        self.init = init
        self.a = a
        self.b = b
        self.target_n_neighbors = target_n_neighbors
        self.target_weight = target_weight
        self.target_metric = target_metric
        self.hash_input = hash_input
        self.random_state = random_state
        self.precomputed_knn = precomputed_knn
        self.callback = callback
        self.build_algo = build_algo
        self.build_kwds = build_kwds
        self.device_ids = device_ids

    @generate_docstring(
        convert_dtype_cast="np.float32",
        X="dense_sparse",
        skip_parameters_heading=True,
    )
    @reflect(reset=True)
    def fit(self, X, y=None, *, convert_dtype=True, knn_graph=None) -> "UMAP":
        """
        Fit X into an embedded space.

        Parameters
        ----------
        knn_graph : array / sparse array / tuple, optional (device or host)
            Either one of a tuple (indices, distances) of
            arrays of shape (n_samples, n_neighbors), a pairwise distances
            dense array of shape (n_samples, n_samples) or a KNN graph
            sparse array (preferably CSR/COO). This feature allows
            the precomputation of the KNN outside of UMAP
            and also allows the use of a custom distance function. This function
            should match the metric used to train the UMAP embeedings.
            Takes precedence over the precomputed_knn parameter.
        """
        if len(X.shape) != 2:
            raise ValueError("Reshape your data: data should be two dimensional")

        cdef int n_rows = X.shape[0]
        cdef int n_dims = X.shape[1]

        if n_rows < 2:
            raise ValueError(
                f"Found an array with {n_rows} sample(s) (shape={X.shape}) "
                f"while a minimum of 2 is required."
            )
        if n_dims < 1:
            raise ValueError(
                f"Found an array with 0 feature(s) (shape={X.shape}) "
                f"while a minimum of 1 is required."
            )

        cdef bool X_is_sparse = is_sparse(X)

        cdef lib.UMAPParams params
        init_params(self, params, n_rows=n_rows, is_sparse=X_is_sparse)

        cdef uintptr_t X_ptr = 0, X_indices_ptr = 0, X_indptr_ptr = 0
        cdef size_t X_nnz = 0

        # Don't coerce to device memory when using a precomputed KNN, so
        # that X may be dropped earlier if passed on host.
        mem_type = (
            MemoryType.device
            if knn_graph is None and self.precomputed_knn is None
            else False
        )

        if X_is_sparse:
            X_m = SparseCumlArray(X, convert_to_dtype=cp.float32, convert_to_mem_type=mem_type)
            X_ptr = X_m.data.ptr
            X_indices_ptr = X_m.indices.ptr
            X_indptr_ptr = X_m.indptr.ptr
            X_nnz = X_m.nnz
        else:
            X_m = input_to_cuml_array(
                X,
                order="C",
                check_dtype=np.float32,
                convert_to_dtype=(np.float32 if convert_dtype else None),
                convert_to_mem_type=(
                    MemoryType.host
                    if params.build_algo == lib.graph_build_algo.NN_DESCENT
                    else mem_type
                )
            ).array
            X_ptr = X_m.ptr

        cdef uintptr_t y_ptr = 0
        if y is not None:
            y_m = input_to_cuml_array(
                y,
                check_dtype=np.float32,
                convert_to_dtype=(np.float32 if convert_dtype else None),
                check_rows=n_rows,
                check_cols=1,
            ).array
            y_ptr = y_m.ptr

        cdef uintptr_t knn_dists_ptr = 0, knn_indices_ptr = 0
        if knn_graph is not None or self.precomputed_knn is not None:
            if y is not None and self.target_metric != "categorical":
                raise ValueError(
                    "Cannot provide a KNN graph when in semi-supervised mode "
                    "with categorical target_metric for now."
                )

            knn_indices, knn_dists = extract_knn_graph(
                (knn_graph if knn_graph is not None else self.precomputed_knn),
                self._n_neighbors,
            )
            if X_is_sparse:
                knn_indices = input_to_cuml_array(
                    knn_indices, convert_to_dtype=np.int32
                ).array
            knn_indices_ptr = knn_indices.ptr
            knn_dists_ptr = knn_dists.ptr
        else:
            knn_indices = knn_dists = None

        handle = get_handle(model=self, device_ids=self.device_ids)
        cdef handle_t * handle_ = <handle_t*> <size_t> handle.getHandle()
        cdef unique_ptr[device_buffer] embeddings_buffer
        cdef lib.HostCOO fss_graph = lib.HostCOO()
        handle_ = <handle_t*> <size_t> handle.getHandle()

        if is_array_like(self.init):
            init_m = input_to_cuml_array(
                self.init,
                order="C",
                check_dtype=np.float32,
                convert_to_dtype=np.float32,
                convert_to_mem_type=False,
                check_rows=n_rows,
                check_cols=self.n_components,
            ).array

            embeddings_buffer.reset(
                new device_buffer(
                    <const void*><uintptr_t>init_m.ptr,
                    init_m.size,
                    handle_.get_stream(),
                    get_current_device_resource()
                )
            )

        with nogil:
            if X_is_sparse:
                lib.fit_sparse(
                    handle_[0],
                    <int*><uintptr_t> X_indptr_ptr,
                    <int*><uintptr_t> X_indices_ptr,
                    <float*><uintptr_t> X_ptr,
                    X_nnz,
                    <float*> y_ptr,
                    n_rows,
                    n_dims,
                    <int*> knn_indices_ptr,
                    <float*> knn_dists_ptr,
                    &params,
                    embeddings_buffer,
                    fss_graph,
                )
            else:
                lib.fit(
                    handle_[0],
                    <float*> X_ptr,
                    <float*> y_ptr,
                    n_rows,
                    n_dims,
                    <int64_t*> knn_indices_ptr,
                    <float*> knn_dists_ptr,
                    &params,
                    embeddings_buffer,
                    fss_graph,
                )
        handle.sync()

        buffer = DeviceBuffer.c_from_unique_ptr(move(embeddings_buffer))
        embedding = cp.ndarray(
            shape=(n_rows, self.n_components),
            dtype=np.float32,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(buffer.ptr, buffer.size, owner=buffer), 0
            ),
            order="C"
        )
        self.embedding_ = CumlArray(data=embedding, index=X_m.index)
        self.graph_ = copy_raft_host_coo_to_scipy_coo(fss_graph)
        self._raw_data = X_m
        self._sparse_data = X_is_sparse
        self._supervised = y is not None
        self._knn_indices = knn_indices
        self._knn_dists = knn_dists

        if self.hash_input:
            self._input_hash = _joblib_hash(X_m.to_output("numpy"))
        else:
            self._input_hash = None

        return self

    @generate_docstring(
        convert_dtype_cast="np.float32",
        skip_parameters_heading=True,
        return_values={
            "name": "X_new",
            "type": "dense",
            "description": "Embedding of the data in low-dimensional space.",
            "shape": "(n_samples, n_components)"
        }
    )
    @reflect
    def fit_transform(
        self, X, y=None, *, convert_dtype=True, knn_graph=None
    ) -> CumlArray:
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
        knn_graph : array / sparse array / tuple, optional (device or host)
            Either one of a tuple (indices, distances) of
            arrays of shape (n_samples, n_neighbors), a pairwise distances
            dense array of shape (n_samples, n_samples) or a KNN graph
            sparse array (preferably CSR/COO). This feature allows
            the precomputation of the KNN outside of UMAP
            and also allows the use of a custom distance function. This function
            should match the metric used to train the UMAP embeedings.
            Takes precedence over the precomputed_knn parameter.
        """
        self.fit(X, y, convert_dtype=convert_dtype, knn_graph=knn_graph)
        return self.embedding_

    @generate_docstring(
        convert_dtype_cast="np.float32",
        return_values={
            "name": "X_new",
            "type": "dense",
            "description": "Embedding of the data in low-dimensional space.",
            "shape": "(n_samples, n_components)"
        }
    )
    @reflect
    def transform(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Transform X into the existing embedded space and return that
        transformed output.

        Please refer to the reference UMAP implementation for information
        on the differences between fit_transform() and running fit()
        transform().

        Specifically, the transform() function is stochastic:
        https://github.com/lmcinnes/umap/issues/158
        """
        if len(X.shape) != 2:
            raise ValueError("Reshape your data: X should be two dimensional")

        if is_sparse(X):
            X = SparseCumlArray(X, convert_to_dtype=cp.float32)
            index = None
        else:
            X = input_to_cuml_array(
                X,
                order="C",
                check_dtype=np.float32,
                convert_to_dtype=(np.float32 if convert_dtype else None),
            ).array
            index = X.index

        if self._sparse_data and not isinstance(X, SparseCumlArray):
            logger.warn(
                "Model was trained on sparse data but dense data was provided to "
                "transform(). Converting to sparse."
            )
            X = SparseCumlArray(
                cupyx.scipy.sparse.csr_matrix(X.to_output("cupy")),
                convert_to_dtype=cp.float32
            )
        elif not self._sparse_data and isinstance(X, SparseCumlArray):
            logger.warn(
                "Model was trained on dense data but sparse data was provided to "
                "transform(). Converting to dense."
            )
            X = input_to_cuml_array(
                X.to_output("cupy").todense(),
                order="C",
                check_dtype=np.float32,
                convert_to_dtype=(np.float32 if convert_dtype else None),
            ).array

        cdef bool X_is_sparse = self._sparse_data
        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]
        cdef int orig_n_rows = self._raw_data.shape[0]

        if n_cols != self.n_features_in_:
            raise ValueError(
                f"X has {n_cols} features, but UMAP is expecting "
                f"{self.n_features_in_} features as input"
            )

        if self.hash_input:
            if _joblib_hash(X.to_output("numpy")) == self._input_hash:
                return self.embedding_

        cdef lib.UMAPParams params
        init_params(self, params, n_rows=n_rows, is_sparse=X_is_sparse, is_fit=False)

        out = CumlArray.zeros(
            (n_rows, self.n_components),
            order="C",
            dtype=np.float32,
            index=index
        )

        cdef uintptr_t X_ptr, X_indptr_ptr, X_indices_ptr
        cdef uintptr_t orig_ptr, orig_indptr_ptr, orig_indices_ptr
        cdef size_t X_nnz, orig_nnz
        if X_is_sparse:
            X_indptr_ptr = X.indptr.ptr
            X_indices_ptr = X.indices.ptr
            X_ptr = X.data.ptr
            X_nnz = X.nnz
            orig_indptr_ptr = self._raw_data.indptr.ptr
            orig_indices_ptr = self._raw_data.indices.ptr
            orig_ptr = self._raw_data.data.ptr
            orig_nnz = self._raw_data.nnz
        else:
            X_ptr = X.ptr
            orig_ptr = self._raw_data.ptr

        cdef uintptr_t out_ptr = out.ptr
        cdef uintptr_t embedding_ptr = self.embedding_.ptr
        handle = get_handle(model=self, device_ids=self.device_ids)
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()

        with nogil:
            if X_is_sparse:
                lib.transform_sparse(
                    handle_[0],
                    <int*> X_indptr_ptr,
                    <int*> X_indices_ptr,
                    <float*> X_ptr,
                    X_nnz,
                    n_rows,
                    n_cols,
                    <int*> orig_indptr_ptr,
                    <int*> orig_indices_ptr,
                    <float*> orig_ptr,
                    orig_nnz,
                    orig_n_rows,
                    <float*> embedding_ptr,
                    orig_n_rows,
                    &params,
                    <float*> out_ptr,
                )
            else:
                lib.transform(
                    handle_[0],
                    <float*>X_ptr,
                    n_rows,
                    n_cols,
                    <float*>orig_ptr,
                    orig_n_rows,
                    <float*> embedding_ptr,
                    orig_n_rows,
                    &params,
                    <float*> out_ptr
                )
        handle.sync()

        return out


def fuzzy_simplicial_set(
    X,
    n_neighbors,
    random_state=None,
    metric="euclidean",
    metric_kwds=None,
    knn_indices=None,
    knn_dists=None,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    verbose=False,
):
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
    metric: string (default='euclidean').
        Distance metric to use. Supported distances are ['l1, 'cityblock',
        'taxicab', 'manhattan', 'euclidean', 'l2', 'sqeuclidean', 'canberra',
        'minkowski', 'chebyshev', 'linf', 'cosine', 'correlation', 'hellinger',
        'hamming', 'jaccard']
        Metrics that take arguments (such as minkowski) can have arguments
        passed via the metric_kwds dictionary.
        Note: The 'jaccard' distance metric is only supported for sparse
        inputs.
    metric_kwds: dict (optional, default=None)
        Metric argument
    knn_indices: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the indices of the k-nearest neighbors as a row for
        each data point.
    knn_dists: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the distances of the k-nearest neighbors as a row for
        each data point.
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
        A fuzzy simplicial set represented as a sparse matrix. The (i, j) entry
        of the matrix represents the membership strength of the 1-simplex
        between the ith and jth sample points.
    """
    X_m = input_to_cuml_array(
        X,
        order="C",
        check_dtype=np.float32,
        convert_to_dtype=np.float32
    ).array

    cdef int n_rows = X_m.shape[0]
    cdef int n_cols = X_m.shape[1]

    cdef lib.UMAPParams params
    params.n_neighbors = n_neighbors
    params.random_state = check_random_seed(random_state)
    params.deterministic = (random_state is not None or n_rows < 300)
    params.set_op_mix_ratio = set_op_mix_ratio
    params.local_connectivity = local_connectivity
    params.metric = coerce_metric(metric)
    params.p = (metric_kwds or {}).get("p", 2.0)
    params.verbosity = logger._verbose_to_level(verbose)

    cdef uintptr_t X_ptr, knn_indices_ptr, knn_dists_ptr
    if knn_indices is not None and knn_dists is not None:
        knn_indices_m = input_to_cuml_array(
            knn_indices,
            order="C",
            check_dtype=np.int64,
            convert_to_dtype=np.int64
        ).array
        knn_dists_m = input_to_cuml_array(
            knn_dists,
            order="C",
            check_dtype=np.float32,
            convert_to_dtype=np.float32
        ).array

        X_ptr = 0
        knn_indices_ptr = knn_indices_m.ptr
        knn_dists_ptr = knn_dists_m.ptr
    else:
        X_ptr = X_m.ptr
        knn_indices_ptr = 0
        knn_dists_ptr = 0

    handle = get_handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
    cdef unique_ptr[lib.COO] fss_graph_ptr = lib.get_graph(
        handle_[0],
        <float*> X_ptr,
        NULL,
        n_rows,
        n_cols,
        <int64_t*> knn_indices_ptr,
        <float*> knn_dists_ptr,
        &params
    )
    fss_graph = RaftCOO.from_ptr(fss_graph_ptr)
    return fss_graph.view_cupy_coo()


@reflect
def simplicial_set_embedding(
    data,
    graph,
    n_components=2,
    initial_alpha=1.0,
    a=None,
    b=None,
    gamma=1.0,
    negative_sample_rate=5,
    n_epochs=None,
    init="spectral",
    random_state=None,
    metric="euclidean",
    metric_kwds=None,
    output_metric="euclidean",
    output_metric_kwds=None,
    convert_dtype=True,
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
    gamma: float
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
            * An array-like with initial embedding positions.
    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.
    metric: string (default='euclidean').
        Distance metric to use. Supported distances are ['l1, 'cityblock',
        'taxicab', 'manhattan', 'euclidean', 'l2', 'sqeuclidean', 'canberra',
        'minkowski', 'chebyshev', 'linf', 'cosine', 'correlation', 'hellinger',
        'hamming', 'jaccard']
        Metrics that take arguments (such as minkowski) can have arguments
        passed via the metric_kwds dictionary.
        Note: The 'jaccard' distance metric is only supported for sparse
        inputs.
    metric_kwds: dict (optional, default=None)
        Metric argument
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
    X = input_to_cuml_array(
        data,
        order="C",
        convert_to_dtype=(np.float32 if convert_dtype else None),
        check_dtype=np.float32,
    ).array

    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]

    if a is None or b is None:
        a, b = find_ab_params()

    cdef lib.UMAPParams params
    params.n_components = n_components
    params.initial_alpha = initial_alpha
    params.learning_rate = initial_alpha
    params.a = a
    params.b = b
    params.repulsion_strength = gamma
    params.negative_sample_rate = negative_sample_rate
    params.n_epochs = n_epochs or 0
    params.random_state = check_random_seed(random_state)
    params.deterministic = (random_state is not None or n_rows < 300)
    params.metric = coerce_metric(metric)
    params.p = (metric_kwds or {}).get("p", 2.0)
    params.target_weight = (output_metric_kwds or {}).get("p", 0.5)
    params.verbosity = logger._verbose_to_level(verbose)

    if output_metric in _TARGET_METRICS:
        params.target_metric = _TARGET_METRICS[output_metric]
    else:
        raise ValueError(
            f"Expected `output_metric` to be one of {list(_TARGET_METRICS)}, "
            f"got {output_metric!r}"
        )

    cdef bool initialized = is_array_like(init)
    if initialized:
        embedding = input_to_cuml_array(
            init,
            order="C",
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=np.float32,
            check_rows=n_rows,
            check_cols=n_components,
        ).array
    elif isinstance(init, str) and init in _INITS:
        params.init = _INITS[init]
        embedding = CumlArray.zeros(
            (n_rows, n_components), order="C", dtype=np.float32, index=X.index,
        )
    else:
        raise ValueError(
            "Expected `init` to be an array or one of ['random', 'spectral'], "
            " got `{init!r}`"
        )

    graph = graph.tocoo()
    graph.sum_duplicates()
    if not isinstance(graph, cupyx.scipy.sparse.coo_matrix):
        graph = cupyx.scipy.sparse.coo_matrix(graph)

    handle = get_handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
    cdef RaftCOO fss_graph = RaftCOO.from_cupy_coo(handle, graph)
    cdef uintptr_t embedding_ptr = embedding.ptr
    cdef uintptr_t X_ptr = X.ptr

    if initialized:
        lib.refine(
            handle_[0],
            <float*> X_ptr,
            n_rows,
            n_cols,
            fss_graph.get(),
            &params,
            <float*> embedding_ptr
        )
    else:
        lib.init_and_refine(
            handle_[0],
            <float*> X_ptr,
            n_rows,
            n_cols,
            fss_graph.get(),
            &params,
            <float*> embedding_ptr
        )
    return embedding
