#
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
#

# distutils: language = c++

import warnings

import cupy
import cupyx
import joblib
import numpy as np

import cuml.accel
import cuml.internals
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.common.sparse_utils import is_sparse
from cuml.common.sparsefuncs import extract_knn_infos
from cuml.internals import logger
from cuml.internals.api_decorators import (
    device_interop_preparation,
    enable_device_interop,
)
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.base import UniversalBase, deprecate_non_keyword_only
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.mem_type import MemoryType
from cuml.internals.mixins import CMajorInputTagMixin, SparseInputTagMixin
from cuml.internals.utils import check_random_seed
from cuml.manifold.simpl_set import fuzzy_simplicial_set  # no-cython-lint
from cuml.manifold.simpl_set import simplicial_set_embedding  # no-cython-lint
from cuml.manifold.umap_utils import GraphHolder, coerce_metric, find_ab_params

from libc.stdint cimport uintptr_t
from libc.stdlib cimport free
from pylibraft.common.handle cimport handle_t

from cuml.internals.logger cimport level_enum
from cuml.manifold.umap_utils cimport *


cdef extern from "cuml/manifold/umap.hpp" namespace "ML::UMAP" nogil:

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
                    int * knn_indices,
                    float * knn_dists,
                    UMAPParams *params,
                    float *embeddings,
                    COO * graph) except +

    void transform(handle_t & handle,
                   float * X,
                   int n,
                   int d,
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
           CMajorInputTagMixin,
           SparseInputTagMixin):
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

    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
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

        - `nnd_intermediate_graph_degree` (int, default=128): Intermediate graph degree for NN Descent.
          Must be > `nnd_graph_degree`.

        - `nnd_max_iterations` (int, default=20): Max NN Descent iterations.

        - `nnd_termination_threshold` (float, default=0.0001): Stricter threshold leads to better convergence
          but longer runtime.

        - `nnd_n_clusters` (int, default=1): Number of clusters for data partitioning.
          Higher values reduce memory usage at the cost of accuracy. When `nnd_n_clusters > 1`, data must be on host memory.
          Refer to data_on_host argument for fit_transform function.

        - `nnd_overlap_factor` (int, default=2): Number of clusters each data point belongs to.
          Valid only when `nnd_n_clusters > 1`. Must be < 'nnd_n_clusters'.

        Hints:

        - Increasing `nnd_graph_degree` and `nnd_max_iterations` may improve accuracy.

        - The ratio `nnd_overlap_factor / nnd_n_clusters` impacts memory usage.
          Approximately `(nnd_overlap_factor / nnd_n_clusters) * num_rows_in_entire_data` rows
          will be loaded onto device memory at once.  E.g., 2/20 uses less device memory than 2/10.

        - Larger `nnd_overlap_factor` results in better accuracy of the final knn graph.
          E.g. While using similar amount of device memory, `(nnd_overlap_factor / nnd_n_clusters)` = 4/20 will have better accuracy
          than 2/10 at the cost of performance.

        - Start with `nnd_overlap_factor = 2` and gradually increase (2->3->4 ...) for better accuracy.

        - Start with `nnd_n_clusters = 4` and increase (4 → 8 → 16...) for less GPU memory usage.
          This is independent from nnd_overlap_factor as long as 'nnd_overlap_factor' < 'nnd_n_clusters'.

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

    _cpu_estimator_import_path = 'umap.UMAP'
    embedding_ = CumlArrayDescriptor(order='C')

    _hyperparam_interop_translator = {
        "metric": {
            "sokalsneath": "NotImplemented",
            "rogerstanimoto": "NotImplemented",
            "sokalmichener": "NotImplemented",
            "yule": "NotImplemented",
            "ll_dirichlet": "NotImplemented",
            "russellrao": "NotImplemented",
            "kulsinski": "NotImplemented",
            "dice": "NotImplemented",
            "wminkowski": "NotImplemented",
            "mahalanobis": "NotImplemented",
            "haversine": "NotImplemented",
        },
        "init": {
            "pca": "NotImplemented",
            "tswspectral": "NotImplemented"
        },
        "target_metric": {
            "l2": "euclidean"
        }
    }

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
                 a=None,
                 b=None,
                 target_n_neighbors=-1,
                 target_weight=0.5,
                 target_metric="categorical",
                 hash_input=False,
                 random_state=None,
                 precomputed_knn=None,
                 callback=None,
                 handle=None,
                 verbose=False,
                 build_algo="auto",
                 build_kwds=None,
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
        elif not cuml.accel.enabled():
            raise Exception(f"Initialization strategy not supported: {init}")

        if a is None or b is None:
            a, b = type(self).find_ab_params(spread, min_dist)

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

        self.random_state = random_state

        if target_metric == "euclidean" or target_metric == "categorical":
            self.target_metric = target_metric
        elif not cuml.accel.enabled():
            raise Exception(f"Invalid target metric: {target_metric}")

        self.callback = callback  # prevent callback destruction
        self.embedding_ = None

        self.validate_hyperparams()

        self.sparse_fit = False
        self._input_hash = None
        self._small_data = False

        self.precomputed_knn = extract_knn_infos(precomputed_knn, n_neighbors)

        if build_algo == "auto" or build_algo == "brute_force_knn" or build_algo == "nn_descent":
            if self.deterministic and build_algo == "auto":
                # TODO: for now, users should be able to see the same results as previous version
                # (i.e. running brute force knn) when they explicitly pass random_state
                # https://github.com/rapidsai/cuml/issues/5985
                with logger.set_level(logger._verbose_to_level(verbose)):
                    logger.info("build_algo set to brute_force_knn because random_state is given")
                self.build_algo ="brute_force_knn"
            else:
                self.build_algo = build_algo
        else:
            raise Exception("Invalid build algo: {}. Only support auto, brute_force_knn and nn_descent" % build_algo)

        self.build_kwds = build_kwds

    def validate_hyperparams(self):

        if self.min_dist > self.spread:
            raise ValueError("min_dist should be <= spread")

    def _build_umap_params(self, sparse):
        cdef UMAPParams* umap_params = new UMAPParams()
        umap_params.n_neighbors = <int> self.n_neighbors
        umap_params.n_components = <int> self.n_components
        umap_params.n_epochs = <int> self.n_epochs if self.n_epochs else 0
        umap_params.learning_rate = <float> self.learning_rate
        umap_params.initial_alpha = <float> self.learning_rate
        umap_params.min_dist = <float> self.min_dist
        umap_params.spread = <float> self.spread
        umap_params.set_op_mix_ratio = <float> self.set_op_mix_ratio
        umap_params.local_connectivity = <float> self.local_connectivity
        umap_params.repulsion_strength = <float> self.repulsion_strength
        umap_params.negative_sample_rate = <int> self.negative_sample_rate
        umap_params.transform_queue_size = <int> self.transform_queue_size
        umap_params.verbosity = <level_enum> self.verbose
        umap_params.a = <float> self.a
        umap_params.b = <float> self.b
        umap_params.target_n_neighbors = <int> self.target_n_neighbors
        umap_params.target_weight = <float> self.target_weight
        umap_params.random_state = <uint64_t> check_random_seed(self.random_state)
        umap_params.deterministic = <bool> self.deterministic

        if self.init == "spectral":
            umap_params.init = <int> 1
        else:  # self.init == "random"
            umap_params.init = <int> 0

        if self.target_metric == "euclidean":
            umap_params.target_metric = MetricType.EUCLIDEAN
        else:  # self.target_metric == "categorical"
            umap_params.target_metric = MetricType.CATEGORICAL

        umap_params.metric = coerce_metric(
            self.metric, sparse=sparse, build_algo=self.build_algo
        )

        if self.metric_kwds is None:
            umap_params.p = <float> 2.0
        else:
            umap_params.p = <float>self.metric_kwds.get('p')

        if self.build_algo == "brute_force_knn":
            umap_params.build_algo = graph_build_algo.BRUTE_FORCE_KNN
        elif self.build_algo == "nn_descent":
            build_kwds = self.build_kwds or {}
            umap_params.build_params.n_clusters = <uint64_t> build_kwds.get("nnd_n_clusters", 1)
            umap_params.build_params.overlap_factor = <uint64_t> build_kwds.get("nnd_overlap_factor", 2)
            if umap_params.build_params.n_clusters > 1 and umap_params.build_params.overlap_factor >= umap_params.build_params.n_clusters:
                raise ValueError("If nnd_n_clusters > 1, then nnd_overlap_factor must be strictly smaller than n_clusters.")
            if umap_params.build_params.n_clusters < 1:
                raise ValueError("nnd_n_clusters must be >= 1")
            umap_params.build_algo = graph_build_algo.NN_DESCENT

            umap_params.build_params.nn_descent_params.graph_degree = <uint64_t> build_kwds.get("nnd_graph_degree", 64)
            umap_params.build_params.nn_descent_params.intermediate_graph_degree = <uint64_t> build_kwds.get("nnd_intermediate_graph_degree", 128)
            umap_params.build_params.nn_descent_params.max_iterations = <uint64_t> build_kwds.get("nnd_max_iterations", 20)
            umap_params.build_params.nn_descent_params.termination_threshold = <float> build_kwds.get("nnd_termination_threshold", 0.0001)

            if umap_params.build_params.nn_descent_params.graph_degree < self.n_neighbors:
                logger.warn("to use nn descent as the build algo, nnd_graph_degree should be larger than or equal to n_neigbors. setting nnd_graph_degree to n_neighbors.")
                umap_params.build_params.nn_descent_params.graph_degree = self.n_neighbors
            if umap_params.build_params.nn_descent_params.intermediate_graph_degree < umap_params.build_params.nn_descent_params.graph_degree:
                logger.warn("to use nn descent as the build algo, nnd_intermediate_graph_degree should be larger than or equal to nnd_graph_degree. \
                setting nnd_intermediate_graph_degree to nnd_graph_degree")
                umap_params.build_params.nn_descent_params.intermediate_graph_degree = umap_params.build_params.nn_descent_params.graph_degree
        else:
            raise ValueError(f"Unsupported value for `build_algo`: {self.build_algo}")

        cdef uintptr_t callback_ptr = 0
        if self.callback:
            callback_ptr = self.callback.get_native_callback()
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
    @deprecate_non_keyword_only("convert_dtype", "knn_graph", "data_on_host")
    def fit(self, X, y=None, convert_dtype=True,
            knn_graph=None, data_on_host=False) -> "UMAP":
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

        .. deprecated:: 25.06
            Using `nnd_n_clusters>1` with data on device is deprecated in version 25.06 and will be removed in 25.08. Use `data_on_host=True` to use with `nnd_n_clusters>1`."
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
            self._sparse_data = True
            if self.build_algo == "nn_descent":
                raise ValueError("NN Descent does not support sparse inputs")

        # Handle dense inputs
        else:
            self._sparse_data = False
            if data_on_host:
                convert_to_mem_type = MemoryType.host
            else:
                build_kwds = self.build_kwds or {}
                if build_kwds.get("nnd_n_clusters", 1) > 1:
                    warnings.warn(
                        ("Using nnd_n_clusters>1 with data on device is deprecated in version 25.06 and will be removed in 25.08. Use data_on_host=True to use with nnd_n_clusters>1."),
                        FutureWarning,
                    )
                    convert_to_mem_type = MemoryType.host
                else:
                    convert_to_mem_type = MemoryType.device

            self._raw_data, self.n_rows, self.n_dims, _ = \
                input_to_cuml_array(X, order='C', check_dtype=np.float32,
                                    convert_to_dtype=(np.float32
                                                      if convert_dtype
                                                      else None),
                                    convert_to_mem_type=convert_to_mem_type)

        if self.build_algo == "auto":
            if self.n_rows <= 50000 or self.sparse_fit:
                # brute force is faster for small datasets
                logger.info("Building knn graph using brute force")
                self.build_algo = "brute_force_knn"
            else:
                logger.info("Building knn graph using nn descent")
                self.build_algo = "nn_descent"

        if self.build_algo == "brute_force_knn" and data_on_host:
            raise ValueError("Data cannot be on host for building with brute force knn")

        if self.n_rows <= 1:
            raise ValueError("There needs to be more than 1 sample to "
                             "build nearest the neighbors graph")
        if self.build_algo == "nn_descent" and self.n_rows < 150:
            # https://github.com/rapidsai/cuvs/issues/184
            warnings.warn("using nn_descent as build_algo on a small dataset (< 150 samples) is unstable")

        cdef uintptr_t _knn_dists_ptr = 0
        cdef uintptr_t _knn_indices_ptr = 0
        if knn_graph is not None or self.precomputed_knn is not None:
            if knn_graph is not None:
                knn_indices, knn_dists = extract_knn_infos(knn_graph,
                                                           self.n_neighbors)
            elif self.precomputed_knn is not None:
                knn_indices, knn_dists = self.precomputed_knn

            if self.sparse_fit:
                knn_indices, _, _, _ = \
                    input_to_cuml_array(knn_indices, convert_to_dtype=np.int32)

            _knn_dists_ptr = knn_dists.ptr
            _knn_indices_ptr = knn_indices.ptr
            self._knn_dists = knn_dists
            self._knn_indices = knn_indices

        self.n_neighbors = min(self.n_rows, self.n_neighbors)

        self.embedding_ = CumlArray.zeros((self.n_rows,
                                           self.n_components),
                                          order="C", dtype=np.float32,
                                          index=self._raw_data.index)

        if self.hash_input:
            self._input_hash = joblib.hash(self._raw_data.to_output('numpy'))

        cdef uintptr_t _embed_raw_ptr = self.embedding_.ptr

        cdef uintptr_t _y_raw_ptr = 0

        if y is not None:
            y_m, _, _, _ = \
                input_to_cuml_array(y, check_dtype=np.float32,
                                    convert_to_dtype=(np.float32
                                                      if convert_dtype
                                                      else None))
            _y_raw_ptr = y_m.ptr

        cdef handle_t * handle_ = \
            <handle_t*> <size_t> self.handle.getHandle()
        fss_graph = GraphHolder.new_graph(handle_.get_stream())
        cdef UMAPParams* umap_params = \
            <UMAPParams*> <size_t> self._build_umap_params(
                                                           self.sparse_fit)
        if self.sparse_fit:
            fit_sparse(handle_[0],
                       <int*><uintptr_t> self._raw_data.indptr.ptr,
                       <int*><uintptr_t> self._raw_data.indices.ptr,
                       <float*><uintptr_t> self._raw_data.data.ptr,
                       <size_t> self._raw_data.nnz,
                       <float*> _y_raw_ptr,
                       <int> self.n_rows,
                       <int> self.n_dims,
                       <int*> _knn_indices_ptr,
                       <float*> _knn_dists_ptr,
                       <UMAPParams*> umap_params,
                       <float*> _embed_raw_ptr,
                       <COO*> fss_graph.get())

        else:
            fit(handle_[0],
                <float*><uintptr_t> self._raw_data.ptr,
                <float*> _y_raw_ptr,
                <int> self.n_rows,
                <int> self.n_dims,
                <int64_t*> _knn_indices_ptr,
                <float*> _knn_dists_ptr,
                <UMAPParams*>umap_params,
                <float*>_embed_raw_ptr,
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
    @deprecate_non_keyword_only("convert_dtype", "knn_graph", "data_on_host")
    def fit_transform(self, X, y=None, convert_dtype=True,
                      knn_graph=None, data_on_host=False) -> CumlArray:
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

        .. deprecated:: 25.06
            Using `nnd_n_clusters>1` with data on device is deprecated in version 25.06 and will be removed in 25.08. Use `data_on_host=True` to use with `nnd_n_clusters>1`."
        """
        self.fit(X, y, convert_dtype=convert_dtype, knn_graph=knn_graph, data_on_host=data_on_host)

        return self.embedding_

    @generate_docstring(convert_dtype_cast='np.float32',
                        return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Embedding of the \
                                                       data in \
                                                       low-dimensional space.',
                                       'shape': '(n_samples, n_components)'})
    @enable_device_interop
    @deprecate_non_keyword_only("convert_dtype")
    def transform(self, X, convert_dtype=True) -> CumlArray:
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
            X_m, n_rows, n_cols, _ = \
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

        embedding = CumlArray.zeros((n_rows, self.n_components),
                                    order="C", dtype=np.float32,
                                    index=index)
        cdef uintptr_t _xformed_ptr = embedding.ptr

        cdef uintptr_t _embed_ptr = self.embedding_.ptr

        # NN Descent doesn't support transform yet
        if self.build_algo == "nn_descent" or self.build_algo == "auto":
            self.build_algo = "brute_force_knn"
            logger.info("Transform can only be run with brute force. Using brute force.")

        cdef UMAPParams* umap_params = <UMAPParams*> <size_t> self._build_umap_params(self.sparse_fit)
        cdef handle_t * handle_ = <handle_t*> <size_t> self.handle.getHandle()
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
                             <float*> _embed_ptr,
                             <int> self._raw_data.shape[0],
                             <UMAPParams*> umap_params,
                             <float*> _xformed_ptr)
        else:
            transform(handle_[0],
                      <float*><uintptr_t> X_m.ptr,
                      <int> n_rows,
                      <int> n_cols,
                      <float*><uintptr_t>self._raw_data.ptr,
                      <int> self._raw_data.shape[0],
                      <float*> _embed_ptr,
                      <int> n_rows,
                      <UMAPParams*> umap_params,
                      <float*> _xformed_ptr)
        self.handle.sync()

        UMAP._destroy_umap_params(<size_t>umap_params)

        del X_m
        return embedding

    @property
    def _n_neighbors(self):
        return self.n_neighbors

    @_n_neighbors.setter
    def _n_neighbors(self, value):
        self.n_neighbors = value

    @property
    def _a(self):
        return self.a

    @_a.setter
    def _a(self, value):
        self.a = value

    @property
    def _b(self):
        return self.b

    @_b.setter
    def _b(self, value):
        self.b = value

    @property
    def _initial_alpha(self):
        return self.learning_rate

    @_initial_alpha.setter
    def _initial_alpha(self, value):
        self.learning_rate = value

    @property
    def _disconnection_distance(self):
        from umap.umap_ import DISCONNECTION_DISTANCES
        self.disconnection_distance = DISCONNECTION_DISTANCES.get(self.metric, np.inf)
        return self.disconnection_distance

    @_disconnection_distance.setter
    def _disconnection_distance(self, value):
        self.disconnection_distance = value

    def gpu_to_cpu(self):
        from umap.umap_ import nearest_neighbors
        if hasattr(self, 'knn_dists') and hasattr(self, 'knn_indices'):
            self._knn_dists = self.knn_dists
            self._knn_indices = self.knn_indices
            self._knn_search_index = None
        if hasattr(self, '_raw_data'):
            self._knn_dists, self._knn_indices, self._knn_search_index = \
                nearest_neighbors(self._raw_data.to_output('numpy'), self.n_neighbors, self.metric,
                                  self.metric_kwds, False, self.random_state)

        super().gpu_to_cpu()
        self._cpu_model._validate_parameters()

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
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
            "build_kwds"
        ]

    def get_attr_names(self):
        return ['_raw_data', 'embedding_', '_input_hash', '_small_data',
                '_knn_dists', '_knn_indices', '_knn_search_index',
                '_disconnection_distance', '_n_neighbors', '_a', '_b',
                '_initial_alpha', '_sparse_data']
