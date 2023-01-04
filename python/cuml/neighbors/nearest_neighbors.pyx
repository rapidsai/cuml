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

import numpy as np
import cupy as cp
import cupyx
import ctypes
import warnings
import math

import cuml.internals
from cuml.internals.base import UniversalBase
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.common.doc_utils import generate_docstring
from cuml.common.doc_utils import insert_into_docstring
from cuml.internals.import_utils import has_scipy
from cuml.internals.mixins import CMajorInputTagMixin
from cuml.internals.input_utils import input_to_cupy_array
from cuml.common import input_to_cuml_array
from cuml.common.sparse_utils import is_sparse
from cuml.common.sparse_utils import is_dense
from cuml.metrics.distance_type cimport DistanceType
from cuml.internals.api_decorators import device_interop_preparation
from cuml.internals.api_decorators import enable_device_interop

from cuml.neighbors.ann cimport *
from pylibraft.common.handle cimport handle_t

from cython.operator cimport dereference as deref

from libcpp cimport bool
from libcpp.memory cimport shared_ptr

from libc.stdint cimport uintptr_t, int64_t, uint32_t
from libc.stdlib cimport calloc, malloc, free

from libcpp.vector cimport vector

from numba import cuda
import rmm

cimport cuml.common.cuda


if has_scipy():
    import scipy.sparse


cdef extern from "raft/spatial/knn/ball_cover_common.h" \
        namespace "raft::spatial::knn":
    cdef cppclass BallCoverIndex[int64_t, float, uint32_t]:
        BallCoverIndex(const handle_t &handle,
                       float *X,
                       uint32_t n_rows,
                       uint32_t n_cols,
                       DistanceType metric) except +

cdef extern from "cuml/neighbors/knn.hpp" namespace "ML":
    void brute_force_knn(
        const handle_t &handle,
        vector[float*] &inputs,
        vector[int] &sizes,
        int D,
        float *search_items,
        int n,
        int64_t *res_I,
        float *res_D,
        int k,
        bool rowMajorIndex,
        bool rowMajorQuery,
        DistanceType metric,
        float metric_arg
    ) except +

    void rbc_build_index(
        const handle_t &handle,
        BallCoverIndex[int64_t, float, uint32_t] &index,
    ) except +

    void rbc_knn_query(
        const handle_t &handle,
        BallCoverIndex[int64_t, float, uint32_t] &index,
        uint32_t k,
        float *search_items,
        uint32_t n_search_items,
        int64_t *out_inds,
        float *out_dists
    ) except +

    void approx_knn_build_index(
        handle_t &handle,
        knnIndex* index,
        knnIndexParam* params,
        DistanceType metric,
        float metricArg,
        float *index_array,
        int n,
        int D
    ) except +

    void approx_knn_search(
        handle_t &handle,
        float *distances,
        int64_t* indices,
        knnIndex* index,
        int k,
        const float *query_array,
        int n
    ) except +


cdef extern from "cuml/neighbors/knn_sparse.hpp" namespace "ML::Sparse":
    void brute_force_knn(handle_t &handle,
                         const int *idxIndptr,
                         const int *idxIndices,
                         const float *idxData,
                         size_t idxNNZ,
                         int n_idx_rows,
                         int n_idx_cols,
                         const int *queryIndptr,
                         const int *queryIndices,
                         const float *queryData,
                         size_t queryNNZ,
                         int n_query_rows,
                         int n_query_cols,
                         int *output_indices,
                         float *output_dists,
                         int k,
                         size_t batch_size_index,
                         size_t batch_size_query,
                         DistanceType metric,
                         float metricArg) except +


class NearestNeighbors(UniversalBase,
                       CMajorInputTagMixin):
    """
    NearestNeighbors is an queries neighborhoods from a given set of
    datapoints. Currently, cuML supports k-NN queries, which define
    the neighborhood as the closest `k` neighbors to each query point.

    Parameters
    ----------
    n_neighbors : int (default=5)
        Default number of neighbors to query
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    algorithm : string (default='auto')
        The query algorithm to use. Valid options are:

        - ``'auto'``: to automatically select brute-force or
          random ball cover based on data shape and metric
        - ``'rbc'``: for the random ball algorithm, which partitions
          the data space and uses the triangle inequality to lower the
          number of potential distances. Currently, this algorithm
          supports Haversine (2d) and Euclidean in 2d and 3d.
        - ``'brute'``: for brute-force, slow but produces exact results
        - ``'ivfflat'``: for inverted file, divide the dataset in partitions
          and perform search on relevant partitions only
        - ``'ivfpq'``: for inverted file and product quantization,
          same as inverted list, in addition the vectors are broken
          in n_features/M sub-vectors that will be encoded thanks
          to intermediary k-means clusterings. This encoding provide
          partial information allowing faster distances calculations
        - ``'ivfsq'``: for inverted file and scalar quantization,
          same as inverted list, in addition vectors components
          are quantized into reduced binary representation allowing
          faster distances calculations

    metric : string (default='euclidean').
        Distance metric to use. Supported distances are ['l1, 'cityblock',
        'taxicab', 'manhattan', 'euclidean', 'l2', 'braycurtis', 'canberra',
        'minkowski', 'chebyshev', 'jensenshannon', 'cosine', 'correlation']
    p : float (default=2)
        Parameter for the Minkowski metric. When p = 1, this is equivalent to
        manhattan distance (l1), and euclidean distance (l2) for p = 2. For
        arbitrary p, minkowski distance (lp) is used.
    algo_params : dict, optional (default=None)
        Used to configure the nearest neighbor algorithm to be used.
        If set to None, parameters will be generated automatically.
        Parameters for algorithm ``'brute'`` when inputs are sparse:

            - batch_size_index : (int) number of rows in each batch of \
                                 index array
            - batch_size_query : (int) number of rows in each batch of \
                                 query array

        Parameters for algorithm ``'ivfflat'``:

            - nlist: (int) number of cells to partition dataset into
            - nprobe: (int) at query time, number of cells used for search

        Parameters for algorithm ``'ivfpq'``:

            - nlist: (int) number of cells to partition dataset into
            - nprobe: (int) at query time, number of cells used for search
            - M: (int) number of subquantizers
            - n_bits: (int) bits allocated per subquantizer
            - usePrecomputedTables : (bool) wether to use precomputed tables

        Parameters for algorithm ``'ivfsq'``:

            - nlist: (int) number of cells to partition dataset into
            - nprobe: (int) at query time, number of cells used for search
            - qtype: (string) quantizer type (among QT_8bit, QT_4bit,
              QT_8bit_uniform, QT_4bit_uniform, QT_fp16, QT_8bit_direct,
              QT_6bit)
            - encodeResidual: (bool) wether to encode residuals

    metric_expanded : bool
        Can increase performance in Minkowski-based (Lp) metrics (for p > 1)
        by using the expanded form and not computing the n-th roots.
        This is currently ignored.

    metric_params : dict, optional (default = None)
        This is currently ignored.

    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Examples
    --------

    .. code-block:: python

        >>> import cudf
        >>> from cuml.neighbors import NearestNeighbors
        >>> from cuml.datasets import make_blobs

        >>> X, _ = make_blobs(n_samples=5, centers=5,
        ...                   n_features=10, random_state=42)

        >>> # build a cudf Dataframe
        >>> X_cudf = cudf.DataFrame(X)

        >>> # fit model
        >>> model = NearestNeighbors(n_neighbors=3)
        >>> model.fit(X)
        NearestNeighbors()

        >>> # get 3 nearest neighbors
        >>> distances, indices = model.kneighbors(X_cudf)

        >>> # print results
        >>> print(indices)
        0  1  2
        0  0  1  3
        1  1  0  2
        2  2  4  0
        3  3  0  2
        4  4  2  3
        >>> print(distances) # doctest: +SKIP
                0          1          2
        0  0.007812  24.786566  26.399996
        1  0.000000  24.786566  30.045017
        2  0.007812   5.458400  27.051241
        3  0.000000  26.399996  27.543869
        4  0.000000   5.458400  29.583437

    Notes
    -----

    Warning: Approximate Nearest Neighbor methods might be unstable
    in this version of cuML. This is due to a known issue in
    the FAISS release that this cuML version is linked to.
    (see cuML issue #4020)

    Warning: For compatibility with libraries that rely on scikit-learn,
    kwargs allows for passing of arguments that are not explicit in the
    class constructor, such as 'n_jobs', but they have no effect on behavior.

    For an additional example see `the NearestNeighbors notebook
    <https://github.com/rapidsai/cuml/blob/main/notebooks/nearest_neighbors_demo.ipynb>`_.

    For additional docs, see `scikit-learn's NearestNeighbors
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors>`_.

    """

    _cpu_estimator_import_path = 'sklearn.neighbors.NearestNeighbors'
    _fit_X = CumlArrayDescriptor(order='C')

    @device_interop_preparation
    def __init__(self, *,
                 n_neighbors=5,
                 verbose=False,
                 handle=None,
                 algorithm="auto",
                 metric="euclidean",
                 p=2,
                 algo_params=None,
                 metric_params=None,
                 output_type=None,
                 **kwargs):

        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)

        self.n_neighbors = n_neighbors
        self.n_indices = 0
        self.effective_metric_ = metric
        self.effective_metric_params_ = metric_params if metric_params else {}
        self.algo_params = algo_params
        self.p = p
        self.algorithm = algorithm
        self._fit_method = self.algorithm
        self.selected_algorithm_ = algorithm
        self.algo_params = algo_params
        self.knn_index = None

    @generate_docstring(X='dense_sparse')
    @enable_device_interop
    def fit(self, X, convert_dtype=True) -> "NearestNeighbors":
        """
        Fit GPU index for performing nearest neighbor queries.

        """
        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        self.n_samples_fit_, self.n_features_in_ = X.shape

        if self.algorithm == "auto":
            if (self.n_features_in_ == 2 or self.n_features_in_ == 3) and \
                    not is_sparse(X) and self.effective_metric_ in \
                    cuml.neighbors.VALID_METRICS["rbc"] and \
                    math.sqrt(X.shape[0]) >= self.n_neighbors:
                self._fit_method = "rbc"
            else:
                self._fit_method = "brute"

        if self.algorithm == "rbc" and self.n_features_in_ > 3:
            raise ValueError("The rbc algorithm is not supported for"
                             " >3 dimensions currently.")

        if is_sparse(X):
            valid_metrics = cuml.neighbors.VALID_METRICS_SPARSE
            valid_metric_str = "_SPARSE"
            self._fit_X = SparseCumlArray(X, convert_to_dtype=cp.float32,
                                          convert_format=False)

        else:
            valid_metrics = cuml.neighbors.VALID_METRICS
            valid_metric_str = ""
            self._fit_X, _, _, dtype = \
                input_to_cuml_array(X, order='C', check_dtype=np.float32,
                                    convert_to_dtype=(np.float32
                                                      if convert_dtype
                                                      else None))

        self._output_index = self._fit_X.index
        self.feature_names_in_ = self._fit_X.index

        if self.effective_metric_ not in \
                valid_metrics[self._fit_method]:
            raise ValueError("Metric %s is not valid. "
                             "Use sorted(cuml.neighbors.VALID_METRICS%s[%s]) "
                             "to get valid options." %
                             (valid_metric_str,
                              self.effective_metric_,
                              self._fit_method))

        cdef handle_t* handle_ = <handle_t*><uintptr_t> self.handle.getHandle()
        cdef knnIndexParam* algo_params = <knnIndexParam*> 0
        if self._fit_method in ['ivfflat', 'ivfpq', 'ivfsq']:
            warnings.warn("\nWarning: Approximate Nearest Neighbor methods "
                          "might be unstable in this version of cuML. "
                          "This is due to a known issue in the FAISS "
                          "release that this cuML version is linked to. "
                          "(see cuML issue #4020)")

            if not is_dense(X):
                raise ValueError("Approximate Nearest Neigbors methods "
                                 "require dense data")

            additional_info = {'n_samples': self.n_samples_fit_,
                               'n_features': self.n_features_in_}
            knn_index = new knnIndex()
            self.knn_index = <uintptr_t> knn_index
            algo_params = <knnIndexParam*><uintptr_t> \
                build_algo_params(self._fit_method, self.algo_params,
                                  additional_info)
            metric = self._build_metric_type(self.effective_metric_)

            approx_knn_build_index(handle_[0],
                                   <knnIndex*>knn_index,
                                   <knnIndexParam*>algo_params,
                                   <DistanceType>metric,
                                   <float>self.p,
                                   <float*><uintptr_t>self._fit_X.ptr,
                                   <int>self.n_samples_fit_,
                                   <int>self.n_features_in_)
            self.handle.sync()

            destroy_algo_params(<uintptr_t>algo_params)

            del self._fit_X
        elif self._fit_method == "rbc":
            metric = self._build_metric_type(self.effective_metric_)

            rbc_index = new BallCoverIndex[int64_t, float, uint32_t](
                handle_[0], <float*><uintptr_t>self._fit_X.ptr,
                <uint32_t>self.n_samples_fit_, <uint32_t>self.n_features_in_,
                <DistanceType>metric)
            rbc_build_index(handle_[0],
                            deref(rbc_index))
            self.knn_index = <uintptr_t>rbc_index

        self.n_indices = 1
        return self

    def get_param_names(self):
        return super().get_param_names() + \
            ["n_neighbors", "algorithm", "metric",
                "p", "metric_params", "algo_params", "n_jobs"]

    def get_attr_names(self):
        return ['_fit_X', 'effective_metric_', 'effective_metric_params_',
                'n_samples_fit_', 'n_features_in_', 'feature_names_in_',
                '_fit_method']

    @staticmethod
    def _build_metric_type(metric):
        if metric == "euclidean" or metric == "l2":
            m = DistanceType.L2SqrtExpanded
        elif metric == "sqeuclidean":
            m = DistanceType.L2Expanded
        elif metric in ["cityblock", "l1", "manhattan", 'taxicab']:
            m = DistanceType.L1
        elif metric == "braycurtis":
            m = DistanceType.BrayCurtis
        elif metric == "canberra":
            m = DistanceType.Canberra
        elif metric == "minkowski" or metric == "lp":
            m = DistanceType.LpUnexpanded
        elif metric == "chebyshev" or metric == "linf":
            m = DistanceType.Linf
        elif metric == "jensenshannon":
            m = DistanceType.JensenShannon
        elif metric == "cosine":
            m = DistanceType.CosineExpanded
        elif metric == "correlation":
            m = DistanceType.CorrelationExpanded
        elif metric == "inner_product":
            m = DistanceType.InnerProduct
        elif metric == "jaccard":
            m = DistanceType.JaccardExpanded
        elif metric == "hellinger":
            m = DistanceType.HellingerExpanded
        elif metric == "haversine":
            m = DistanceType.Haversine
        else:
            raise ValueError("Metric %s is not supported" % metric)

        return m

    @insert_into_docstring(parameters=[('dense', '(n_samples, n_features)')],
                           return_values=[('dense', '(n_samples, n_features)'),
                                          ('dense',
                                           '(n_samples, n_features)')])
    @enable_device_interop
    def kneighbors(
        self,
        X=None,
        n_neighbors=None,
        return_distance=True,
        convert_dtype=True,
        two_pass_precision=False
    ) -> typing.Union[CumlArray, typing.Tuple[CumlArray, CumlArray]]:
        """
        Query the GPU index for the k nearest neighbors of column vectors in X.

        Parameters
        ----------
        X : {}

        n_neighbors : Integer
            Number of neighbors to search. If not provided, the n_neighbors
            from the model instance is used (default=10)

        return_distance: Boolean
            If False, distances will not be returned

        convert_dtype : bool, optional (default = True)
            When set to True, the kneighbors method will automatically
            convert the inputs to np.float32.

        two_pass_precision : bool, optional (default = False)
            When set to True, a slow second pass will be used to improve the
            precision of results returned for searches using L2-derived
            metrics. FAISS uses the Euclidean distance decomposition trick to
            compute distances in this case, which may result in numerical
            errors for certain data. In particular, when several samples
            are close to the query sample (relative to typical inter-sample
            distances), numerical instability may cause the computed distance
            between the query and itself to be larger than the computed
            distance between the query and another sample. As a result, the
            query is not returned as the nearest neighbor to itself.  If this
            flag is set to true, distances to the query vectors will be
            recomputed with high precision for all retrieved samples, and the
            results will be re-sorted accordingly. Note that for large values
            of k or large numbers of query vectors, this correction becomes
            impractical in terms of both runtime and memory. It should be used
            with care and only when strictly necessary (when precise results
            are critical and samples may be tightly clustered).

        Returns
        -------
        distances : {}
            The distances of the k-nearest neighbors for each column vector
            in X

        indices : {}
            The indices of the k-nearest neighbors for each column vector in X
        """

        return self._kneighbors_internal(X, n_neighbors, return_distance,
                                         convert_dtype,
                                         two_pass_precision=two_pass_precision)

    def _kneighbors_internal(self, X=None, n_neighbors=None,
                             return_distance=True, convert_dtype=True,
                             _output_type=None, two_pass_precision=False):
        """
        Query the GPU index for the k nearest neighbors of column vectors in X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        n_neighbors : Integer
            Number of neighbors to search. If not provided, the n_neighbors
            from the model instance is used (default=10)

        return_distance: Boolean
            If False, distances will not be returned

        convert_dtype : bool, optional (default = True)
            When set to True, the kneighbors method will automatically
            convert the inputs to np.float32.

        _output_cumlarray : bool, optional (default = False)
            When set to True, the class self.output_type is overwritten
            and this method returns the output as a cumlarray

        two_pass_precision : bool, optional (default = False)
            When set to True, a slow second pass will be used to improve the
            precision of results returned for searches using L2-derived
            metrics. FAISS uses the Euclidean distance decomposition trick to
            compute distances in this case, which may result in numerical
            errors for certain data. In particular, when several samples
            are close to the query sample (relative to typical inter-sample
            distances), numerical instability may cause the computed distance
            between the query and itself to be larger than the computed
            distance between the query and another sample. As a result, the
            query is not returned as the nearest neighbor to itself.  If this
            flag is set to true, distances to the query vectors will be
            recomputed with high precision for all retrieved samples, and the
            results will be re-sorted accordingly. Note that for large values
            of k or large numbers of query vectors, this correction becomes
            impractical in terms of both runtime and memory. It should be used
            with care and only when strictly necessary (when precise results
            are critical and samples may be tightly clustered).

        Returns
        -------
        distances: cupy ndarray
            The distances of the k-nearest neighbors for each column vector
            in X

        indices: cupy ndarray
            The indices of the k-nearest neighbors for each column vector in X
        """
        n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors

        use_training_data = X is None
        if X is None:
            X = self._fit_X
            n_neighbors += 1

        if (n_neighbors is None and self.n_neighbors is None) \
                or n_neighbors <= 0:
            raise ValueError("k or n_neighbors must be a positive integers")

        if n_neighbors > self.n_samples_fit_:
            raise ValueError("n_neighbors must be <= number of "
                             "samples in index")

        if X is None:
            raise ValueError("Model needs to be trained "
                             "before calling kneighbors()")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("Dimensions of X need to match dimensions of "
                             "indices (%d)" % self.n_features_in_)

        if hasattr(self, '_fit_X') and isinstance(self._fit_X,
                                                  SparseCumlArray):
            D_ndarr, I_ndarr = self._kneighbors_sparse(X, n_neighbors)
        else:
            D_ndarr, I_ndarr = self._kneighbors_dense(X, n_neighbors,
                                                      convert_dtype)

        self.handle.sync()

        out_type = _output_type \
            if _output_type is not None else self._get_output_type(X)

        if two_pass_precision:
            metric = self._build_metric_type(self.effective_metric_)
            metric_is_l2_based = (
                metric == DistanceType.L2SqrtExpanded or
                metric == DistanceType.L2Expanded or
                (metric == DistanceType.LpUnexpanded and self.p == 2)
            )

            # FAISS employs imprecise distance algorithm only for L2-based
            # expanded metrics. This code correct numerical instabilities
            # that could arise.
            if metric_is_l2_based:
                index = I_ndarr.index
                X = input_to_cupy_array(X).array
                I_cparr = I_ndarr.to_output('cupy')

                self_diff = X[I_cparr] - X[:, cp.newaxis, :]
                precise_distances = cp.sum(
                    self_diff * self_diff, axis=2
                )

                correct_order = cp.argsort(precise_distances, axis=1)

                D_cparr = cp.take_along_axis(precise_distances,
                                             correct_order,
                                             axis=1)
                I_cparr = cp.take_along_axis(I_cparr, correct_order, axis=1)

                D_ndarr = cuml.common.input_to_cuml_array(D_cparr).array
                D_ndarr.index = index
                I_ndarr = cuml.common.input_to_cuml_array(I_cparr).array
                I_ndarr.index = index

        I_ndarr = I_ndarr.to_output(out_type)
        D_ndarr = D_ndarr.to_output(out_type)

        # drop first column if using training data as X
        # this will need to be moved to the C++ layer (cuml issue #2562)
        if use_training_data:
            if out_type in {'cupy', 'numpy', 'numba'}:
                I_ndarr = I_ndarr[:, 1:]
                D_ndarr = D_ndarr[:, 1:]
            else:
                I_ndarr.drop(I_ndarr.columns[0], axis=1)
                D_ndarr.drop(D_ndarr.columns[0], axis=1)

        return (D_ndarr, I_ndarr) if return_distance else I_ndarr

    def _kneighbors_dense(self, X, n_neighbors, convert_dtype=None):

        if not is_dense(X):
            raise ValueError("A NearestNeighbors model trained on dense "
                             "data requires dense input to kneighbors()")

        metric = self._build_metric_type(self.effective_metric_)

        X_m, N, _, dtype = \
            input_to_cuml_array(X, order='C', check_dtype=np.float32,
                                convert_to_dtype=(np.float32 if convert_dtype
                                                  else False))

        # Need to establish result matrices for indices (Nxk)
        # and for distances (Nxk)
        I_ndarr = CumlArray.zeros((N, n_neighbors), dtype=np.int64, order="C",
                                  index=X_m.index)
        D_ndarr = CumlArray.zeros((N, n_neighbors),
                                  dtype=np.float32, order="C",
                                  index=X_m.index)

        cdef uintptr_t I_ptr = I_ndarr.ptr
        cdef uintptr_t D_ptr = D_ndarr.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef vector[float*] *inputs = new vector[float*]()
        cdef vector[int] *sizes = new vector[int]()
        cdef knnIndex* knn_index = <knnIndex*> 0
        cdef BallCoverIndex[int64_t, float, uint32_t]* rbc_index = \
            <BallCoverIndex[int64_t, float, uint32_t]*> 0

        fallback_to_brute = self._fit_method == "rbc" and \
            n_neighbors > math.sqrt(self.n_samples_fit_)

        if fallback_to_brute:
            warnings.warn("algorithm='rbc' requires sqrt(%s) be "
                          "> n_neighbors (%s). falling back to "
                          "brute force search" %
                          (self.n_samples_fit_, n_neighbors))

        if self._fit_method == 'brute' or fallback_to_brute:
            inputs.push_back(<float*><uintptr_t>self._fit_X.ptr)
            sizes.push_back(<int>self.n_samples_fit_)

            brute_force_knn(
                handle_[0],
                deref(inputs),
                deref(sizes),
                <int>self.n_features_in_,
                <float*><uintptr_t>X_m.ptr,
                <int>N,
                <int64_t*>I_ptr,
                <float*>D_ptr,
                <int>n_neighbors,
                True,
                True,
                <DistanceType>metric,
                # minkowski order is currently the only metric argument.
                <float>self.p
            )
        elif self._fit_method == "rbc":
            rbc_index = <BallCoverIndex[int64_t, float, uint32_t]*>\
                <uintptr_t>self.knn_index
            rbc_knn_query(handle_[0],
                          deref(rbc_index),
                          <uint32_t> n_neighbors,
                          <float*><uintptr_t>X_m.ptr,
                          <uint32_t> N,
                          <int64_t*>I_ptr,
                          <float*>D_ptr)
        else:
            knn_index = <knnIndex*><uintptr_t> self.knn_index
            approx_knn_search(
                handle_[0],
                <float*>D_ptr,
                <int64_t*>I_ptr,
                <knnIndex*>knn_index,
                <int>n_neighbors,
                <float*><uintptr_t>X_m.ptr,
                <int>N
            )

        return D_ndarr, I_ndarr

    def _kneighbors_sparse(self, X, n_neighbors):

        if isinstance(self._fit_X, SparseCumlArray) and not is_sparse(X):
            raise ValueError("A NearestNeighbors model trained on sparse "
                             "data requires sparse input to kneighbors()")

        batch_size_index = 10000
        if self.algo_params is not None and \
                "batch_size_index" in self.algo_params:
            batch_size_index = self.algo_params['batch_size_index']

        batch_size_query = 10000
        if self.algo_params is not None and \
                "batch_size_query" in self.algo_params:
            batch_size_query = self.algo_params['batch_size_query']

        X_m = SparseCumlArray(X, convert_to_dtype=cp.float32,
                              convert_format=False)
        metric = self._build_metric_type(self.effective_metric_)

        cdef uintptr_t idx_indptr = self._fit_X.indptr.ptr
        cdef uintptr_t idx_indices = self._fit_X.indices.ptr
        cdef uintptr_t idx_data = self._fit_X.data.ptr

        cdef uintptr_t search_indptr = X_m.indptr.ptr
        cdef uintptr_t search_indices = X_m.indices.ptr
        cdef uintptr_t search_data = X_m.data.ptr

        # Need to establish result matrices for indices (Nxk)
        # and for distances (Nxk)
        I_ndarr = CumlArray.zeros((X_m.shape[0], n_neighbors),
                                  dtype=np.int32, order="C")
        D_ndarr = CumlArray.zeros((X_m.shape[0], n_neighbors),
                                  dtype=np.float32, order="C")

        cdef uintptr_t I_ptr = I_ndarr.ptr
        cdef uintptr_t D_ptr = D_ndarr.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        brute_force_knn(handle_[0],
                        <int*> idx_indptr,
                        <int*> idx_indices,
                        <float*> idx_data,
                        self._fit_X.nnz,
                        self.n_samples_fit_,
                        self.n_features_in_,
                        <int*> search_indptr,
                        <int*> search_indices,
                        <float*> search_data,
                        X_m.nnz,
                        X_m.shape[0],
                        X_m.shape[1],
                        <int*>I_ptr,
                        <float*>D_ptr,
                        n_neighbors,
                        <size_t>batch_size_index,
                        <size_t>batch_size_query,
                        <DistanceType> metric,
                        <float>self.p)

        return D_ndarr, I_ndarr

    @insert_into_docstring(parameters=[('dense', '(n_samples, n_features)')])
    @enable_device_interop
    def kneighbors_graph(self,
                         X=None,
                         n_neighbors=None,
                         mode='connectivity') -> SparseCumlArray:
        """
        Find the k nearest neighbors of column vectors in X and return as
        a sparse matrix in CSR format.

        Parameters
        ----------
        X : {}

        n_neighbors : Integer
            Number of neighbors to search. If not provided, the n_neighbors
            from the model instance is used

        mode : string (default='connectivity')
            Values in connectivity matrix: 'connectivity' returns the
            connectivity matrix with ones and zeros, 'distance' returns the
            edges as the distances between points with the requested metric.

        Returns
        -------
        A : sparse graph in CSR format, shape = (n_samples, n_samples_fit)
            n_samples_fit is the number of samples in the fitted data where
            A[i, j] is assigned the weight of the edge that connects i to j.
            Values will either be ones/zeros or the selected distance metric.
            Return types are either cupy's CSR sparse graph (device) or
            numpy's CSR sparse graph (host)

        """
        if not self._fit_X:
            raise ValueError('This NearestNeighbors instance has not been '
                             'fitted yet, call "fit" before using this '
                             'estimator')

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if mode == 'connectivity':
            indices = self._kneighbors_internal(X, n_neighbors,
                                                return_distance=False,
                                                _output_type="cupy")

            n_samples = indices.shape[0]
            distances = cp.ones(n_samples * n_neighbors, dtype=np.float32)

        elif mode == 'distance':
            distances, indices = self._kneighbors_internal(X, n_neighbors,
                                                           _output_type="cupy")
            distances = cp.ravel(distances)

        else:
            raise ValueError('Unsupported mode, must be one of "connectivity"'
                             ' or "distance" but got "%s" instead' % mode)

        n_samples = indices.shape[0]
        indices = cp.ravel(indices)

        n_nonzero = n_samples * n_neighbors
        rowptr = cp.arange(0, n_nonzero + 1, n_neighbors)

        sparse_csr = cupyx.scipy.sparse.csr_matrix((distances,
                                                    cp.ravel(
                                                        cp.asarray(indices)),
                                                    rowptr),
                                                   shape=(n_samples,
                                                          self.n_samples_fit_))

        return sparse_csr

    def __del__(self):
        cdef knnIndex* knn_index = <knnIndex*>0
        cdef BallCoverIndex* rbc_index = <BallCoverIndex*>0

        kidx = self.__dict__['knn_index'] \
            if 'knn_index' in self.__dict__ else None
        if kidx is not None:
            if self._fit_method in ["ivfflat", "ivfpq", "ivfsq"]:
                knn_index = <knnIndex*><uintptr_t>kidx
                del knn_index
            else:
                rbc_index = <BallCoverIndex*><uintptr_t>kidx
                del rbc_index


@cuml.internals.api_return_sparse_array()
def kneighbors_graph(X=None, n_neighbors=5, mode='connectivity', verbose=False,
                     handle=None, algorithm="brute", metric="euclidean", p=2,
                     include_self=False, metric_params=None):
    """
    Computes the (weighted) graph of k-Neighbors for points in X.

    Parameters
    ----------
    X : array-like (device or host) shape = (n_samples, n_features)
        Dense matrix (floats or doubles) of shape (n_samples, n_features).
        Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
        ndarray, cuda array interface compliant array like CuPy

    n_neighbors : Integer
        Number of neighbors to search. If not provided, the n_neighbors
        from the model instance is used (default=5)

    mode : string (default='connectivity')
        Values in connectivity matrix: 'connectivity' returns the
        connectivity matrix with ones and zeros, 'distance' returns the
        edges as the distances between points with the requested metric.

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.

    algorithm : string (default='brute')
        The query algorithm to use. Valid options are:

        - ``'auto'``: to automatically select brute-force or
          random ball cover based on data shape and metric
        - ``'rbc'``: for the random ball algorithm, which partitions
          the data space and uses the triangle inequality to lower the
          number of potential distances. Currently, this algorithm
          supports 2d Euclidean and Haversine.
        - ``'brute'``: for brute-force, slow but produces exact results
        - ``'ivfflat'``: for inverted file, divide the dataset in partitions
          and perform search on relevant partitions only
        - ``'ivfpq'``: for inverted file and product quantization,
          same as inverted list, in addition the vectors are broken
          in n_features/M sub-vectors that will be encoded thanks
          to intermediary k-means clusterings. This encoding provide
          partial information allowing faster distances calculations
        - ``'ivfsq'``: for inverted file and scalar quantization,
          same as inverted list, in addition vectors components
          are quantized into reduced binary representation allowing
          faster distances calculations

    metric : string (default='euclidean').
        Distance metric to use. Supported distances are ['l1, 'cityblock',
        'taxicab', 'manhattan', 'euclidean', 'l2', 'braycurtis', 'canberra',
        'minkowski', 'chebyshev', 'jensenshannon', 'cosine', 'correlation']

    p : float (default=2) Parameter for the Minkowski metric. When p = 1, this
        is equivalent to manhattan distance (l1), and euclidean distance (l2)
        for p = 2. For arbitrary p, minkowski distance (lp) is used.

    include_self : bool or 'auto' (default=False)
        Whether or not to mark each sample as the first nearest neighbor to
        itself. If 'auto', then True is used for mode='connectivity' and False
        for mode='distance'.

    metric_params : dict, optional (default = None) This is currently ignored.

    Returns
    -------
    A : sparse graph in CSR format, shape = (n_samples, n_samples_fit)
        n_samples_fit is the number of samples in the fitted data where
        A[i, j] is assigned the weight of the edge that connects i to j.
        Values will either be ones/zeros or the selected distance metric.
        Return types are either cupy's CSR sparse graph (device) or
        numpy's CSR sparse graph (host)

    """
    # Set the default output type to "cupy". This will be ignored if the user
    # has set `cuml.global_settings.output_type`. Only necessary for array
    # generation methods that do not take an array as input
    cuml.internals.set_api_output_type("cupy")

    X = NearestNeighbors(
        n_neighbors=n_neighbors,
        verbose=verbose,
        handle=handle,
        algorithm=algorithm,
        metric=metric,
        p=p,
        metric_params=metric_params,
        output_type=cuml.global_settings.root_cm.output_type
    ).fit(X)

    if include_self == 'auto':
        include_self = mode == 'connectivity'

    with cuml.internals.exit_internal_api():
        if not include_self:
            query = None
        else:
            query = X._fit_X

    return X.kneighbors_graph(X=query, n_neighbors=n_neighbors, mode=mode)
