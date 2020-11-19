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

import typing

import numpy as np
import cupy as cp
import cupyx
import cudf
import ctypes
import warnings

import cuml.internals
from cuml.common.base import Base
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.array import CumlArray
from cuml.common.array_sparse import SparseCumlArray
from cuml.common.doc_utils import generate_docstring
from cuml.common.doc_utils import insert_into_docstring
from cuml.common.import_utils import has_scipy
from cuml.common import input_to_cuml_array
from cuml.common.sparse_utils import is_sparse
from cuml.common.sparse_utils import is_dense

from cython.operator cimport dereference as deref

from cuml.raft.common.handle cimport handle_t

from libcpp cimport bool
from libcpp.memory cimport shared_ptr

from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport calloc, malloc, free

from libcpp.vector cimport vector


from numba import cuda
import rmm

cimport cuml.common.cuda


if has_scipy():
    import scipy.sparse

cdef extern from "cuml/neighbors/knn.hpp" namespace "ML":

    enum MetricType:
        METRIC_INNER_PRODUCT = 0,
        METRIC_L2,
        METRIC_L1,
        METRIC_Linf,
        METRIC_Lp,

        METRIC_Canberra = 20,
        METRIC_BrayCurtis,
        METRIC_JensenShannon,

        METRIC_Cosine = 100,
        METRIC_Correlation

    void brute_force_knn(
        handle_t &handle,
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
        MetricType metric,
        float metric_arg,
        bool expanded
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
                         MetricType metric,
                         float metricArg,
                         bool expanded_form) except +


class NearestNeighbors(Base):
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
    algorithm : string (default='brute')
        The query algorithm to use. Currently, only 'brute' is supported.
    metric : string (default='euclidean').
        Distance metric to use. Supported distances are ['l1, 'cityblock',
        'taxicab', 'manhattan', 'euclidean', 'l2', 'braycurtis', 'canberra',
        'minkowski', 'chebyshev', 'jensenshannon', 'cosine', 'correlation']
    p : float (default=2) Parameter for the Minkowski metric. When p = 1, this
        is equivalent to manhattan distance (l1), and euclidean distance (l2)
        for p = 2. For arbitrary p, minkowski distance (lp) is used.
    algo_params : dict, optional (default=None)
        Named arguments for controlling the behavior of different nearest
        neighbors algorithms.

        When algorithm='brute' and inputs are sparse:
            - batch_size_index : (int) number of rows in each batch of
                                 index array
            - batch_size_query : (int) number of rows in each batch of
                                 query array
    metric_expanded : bool
        Can increase performance in Minkowski-based (Lp) metrics (for p > 1)
        by using the expanded form and not computing the n-th roots.
    metric_params : dict, optional (default = None) This is currently ignored.
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_output_type`.
        See :ref:`output-data-type-configuration` for more info.

    Examples
    --------
    .. code-block:: python

      import cudf
      from cuml.neighbors import NearestNeighbors
      from cuml.datasets import make_blobs

      X, _ = make_blobs(n_samples=25, centers=5,
                        n_features=10, random_state=42)

      # build a cudf Dataframe
      X_cudf = cudf.DataFrame(X)

      # fit model
      model = NearestNeighbors(n_neighbors=3)
      model.fit(X)

      # get 3 nearest neighbors
      distances, indices = model.kneighbors(X_cudf)

      # print results
      print(indices)
      print(distances)


    Output:

    .. code-block::

        indices:

             0   1   2
        0    0  14  21
        1    1  19   8
        2    2   9  23
        3    3  14  21
        ...

        22  22  18  11
        23  23  16   9
        24  24  17  10

        distances:

              0         1         2
        0   0.0  4.883116  5.570006
        1   0.0  3.047896  4.105496
        2   0.0  3.558557  3.567704
        3   0.0  3.806127  3.880100
        ...

        22  0.0  4.210738  4.227068
        23  0.0  3.357889  3.404269
        24  0.0  3.428183  3.818043


    Notes
    -----

    For an additional example see `the NearestNeighbors notebook
    <https://github.com/rapidsai/cuml/blob/branch-0.15/notebooks/nearest_neighbors_demo.ipynb>`_.

    For additional docs, see `scikit-learn's NearestNeighbors
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors>`_.
    """

    X_m = CumlArrayDescriptor()

    def __init__(self,
                 n_neighbors=5,
                 verbose=False,
                 handle=None,
                 algorithm="brute",
                 metric="euclidean",
                 p=2,
                 algo_params=None,
                 metric_params=None,
                 output_type=None):

        super(NearestNeighbors, self).__init__(handle=handle,
                                               verbose=verbose,
                                               output_type=output_type)

        if algorithm != "brute":
            raise ValueError("Algorithm %s is not valid. Only 'brute' is"
                             "supported currently." % algorithm)

        if metric not in cuml.neighbors.VALID_METRICS[algorithm]:
            raise ValueError("Metric %s is not valid. "
                             "Use sorted(cuml.neighbors.VALID_METRICS[%s]) "
                             "to get valid options." % (metric, algorithm))

        self.n_neighbors = n_neighbors
        self.n_indices = 0
        self.metric = metric
        self.metric_params = metric_params
        self.algo_params = algo_params
        self.p = p
        self.algorithm = algorithm

    @generate_docstring()
    def fit(self, X, convert_dtype=True) -> "NearestNeighbors":
        """
        Fit GPU index for performing nearest neighbor queries.

        """
        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        self.n_dims = X.shape[1]

        if is_sparse(X):
            self.X_m = SparseCumlArray(X, convert_to_dtype=cp.float32,
                                       convert_format=False)
            self.n_rows = self.X_m.shape[0]

        else:
            self.X_m, self.n_rows, n_cols, dtype = \
                input_to_cuml_array(X, order='F', check_dtype=np.float32,
                                    convert_to_dtype=(np.float32
                                                      if convert_dtype
                                                      else None))

        self.n_indices = 1

        return self

    def get_param_names(self):
        return super().get_param_names() + \
            ["n_neighbors", "algorithm", "metric",
                "p", "metric_params", "algo_params"]

    @staticmethod
    def _build_metric_type(metric):

        expanded = False

        if metric == "euclidean" or metric == "l2":
            m = MetricType.METRIC_L2
        elif metric == "sqeuclidean":
            m = MetricType.METRIC_L2
            expanded = True
        elif metric == "cityblock" or metric == "l1"\
                or metric == "manhattan" or metric == 'taxicab':
            m = MetricType.METRIC_L1
        elif metric == "braycurtis":
            m = MetricType.METRIC_BrayCurtis
        elif metric == "canberra":
            m = MetricType.METRIC_Canberra
        elif metric == "minkowski" or metric == "lp":
            m = MetricType.METRIC_Lp
        elif metric == "chebyshev" or metric == "linf":
            m = MetricType.METRIC_Linf
        elif metric == "jensenshannon":
            m = MetricType.METRIC_JensenShannon
        elif metric == "cosine":
            m = MetricType.METRIC_Cosine
        elif metric == "correlation":
            m = MetricType.METRIC_Correlation
        elif metric == "inner_product":
            m = MetricType.METRIC_INNER_PRODUCT
        else:
            raise ValueError("Metric %s is not supported" % metric)

        return m, expanded

    @insert_into_docstring(parameters=[('dense', '(n_samples, n_features)')],
                           return_values=[('dense', '(n_samples, n_features)'),
                                          ('dense',
                                           '(n_samples, n_features)')])
    def kneighbors(
        self,
        X=None,
        n_neighbors=None,
        return_distance=True,
        convert_dtype=True
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

        Returns
        -------
        distances : {}
            The distances of the k-nearest neighbors for each column vector
            in X

        indices : {}
            The indices of the k-nearest neighbors for each column vector in X
        """

        return self._kneighbors(X, n_neighbors, return_distance, convert_dtype)

    def _kneighbors(self, X=None, n_neighbors=None, return_distance=True,
                    convert_dtype=True, _output_type=None):
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
            X = self.X_m
            n_neighbors += 1

        if (n_neighbors is None and self.n_neighbors is None) \
                or n_neighbors <= 0:
            raise ValueError("k or n_neighbors must be a positive integers")

        if n_neighbors > self.X_m.shape[0]:
            raise ValueError("n_neighbors must be <= number of "
                             "samples in index")

        if X is None:
            raise ValueError("Model needs to be trained "
                             "before calling kneighbors()")

        if X.shape[1] != self.n_dims:
            raise ValueError("Dimensions of X need to match dimensions of "
                             "indices (%d)" % self.n_dims)

        if isinstance(self.X_m, CumlArray):
            D_ndarr, I_ndarr = self._kneighbors_dense(X, n_neighbors,
                                                      convert_dtype)
        elif isinstance(self.X_m, SparseCumlArray):
            D_ndarr, I_ndarr = self._kneighbors_sparse(X, n_neighbors)

        self.handle.sync()

        out_type = _output_type \
            if _output_type is not None else self._get_output_type(X)

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

        if isinstance(self.X_m, CumlArray) and not is_dense(X):
            raise ValueError("A NearestNeighbors model trained on dense "
                             "data requires dense input to kneighbors()")

        metric, expanded = self._build_metric_type(self.metric)

        X_m, N, _, dtype = \
            input_to_cuml_array(X, order='F', check_dtype=np.float32,
                                convert_to_dtype=(np.float32 if convert_dtype
                                                  else False))

        # Need to establish result matrices for indices (Nxk)
        # and for distances (Nxk)
        I_ndarr = CumlArray.zeros((N, n_neighbors), dtype=np.int64, order="C")
        D_ndarr = CumlArray.zeros((N, n_neighbors),
                                  dtype=np.float32, order="C")

        cdef uintptr_t I_ptr = I_ndarr.ptr
        cdef uintptr_t D_ptr = D_ndarr.ptr

        cdef vector[float*] *inputs = new vector[float*]()
        cdef vector[int] *sizes = new vector[int]()

        cdef uintptr_t idx_ptr = self.X_m.ptr
        inputs.push_back(<float*>idx_ptr)
        sizes.push_back(<int>self.X_m.shape[0])

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t x_ctype_st = X_m.ptr

        brute_force_knn(
            handle_[0],
            deref(inputs),
            deref(sizes),
            <int>self.n_dims,
            <float*>x_ctype_st,
            <int>N,
            <int64_t*>I_ptr,
            <float*>D_ptr,
            <int>n_neighbors,
            False,
            False,
            <MetricType>metric,

            # minkowski order is currently the only metric argument.
            <float>self.p,
            < bool > expanded
        )

        return D_ndarr, I_ndarr

    def _kneighbors_sparse(self, X, n_neighbors):

        if isinstance(self.X_m, SparseCumlArray) and not is_sparse(X):
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
        metric, expanded = self._build_metric_type(self.metric)

        cdef uintptr_t idx_indptr = self.X_m.indptr.ptr
        cdef uintptr_t idx_indices = self.X_m.indices.ptr
        cdef uintptr_t idx_data = self.X_m.data.ptr

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
                        self.X_m.nnz,
                        self.X_m.shape[0],
                        self.X_m.shape[1],
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
                        <MetricType> metric,
                        <float>self.p,
                        <bool> expanded)

        return D_ndarr, I_ndarr

    @insert_into_docstring(parameters=[('dense', '(n_samples, n_features)')])
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
        if not self.X_m:
            raise ValueError('This NearestNeighbors instance has not been '
                             'fitted yet, call "fit" before using this '
                             'estimator')

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if mode == 'connectivity':
            indices = self._kneighbors(X, n_neighbors,
                                       return_distance=False,
                                       _output_type="cupy")

            n_samples = indices.shape[0]
            distances = cp.ones(n_samples * n_neighbors, dtype=np.float32)

        elif mode == 'distance':
            distances, indices = self._kneighbors(X, n_neighbors,
                                                  _output_type="cupy")
            distances = cp.ravel(distances)

        else:
            raise ValueError('Unsupported mode, must be one of "connectivity"'
                             ' or "distance" but got "%s" instead' % mode)

        n_samples = indices.shape[0]
        indices = cp.ravel(indices)

        n_samples_fit = self.X_m.shape[0]
        n_nonzero = n_samples * n_neighbors
        rowptr = cp.arange(0, n_nonzero + 1, n_neighbors)

        sparse_csr = cupyx.scipy.sparse.csr_matrix((distances,
                                                    cp.ravel(
                                                        cp.asarray(indices)),
                                                    rowptr),
                                                   shape=(n_samples,
                                                          n_samples_fit))

        return sparse_csr


@cuml.internals.api_return_sparse_array()
def kneighbors_graph(X=None, n_neighbors=5, mode='connectivity', verbose=False,
                     handle=None, algorithm="brute", metric="euclidean", p=2,
                     include_self=False, metric_params=None, output_type=None):
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
        The query algorithm to use. Currently, only 'brute' is supported.

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

    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_output_type`.
        See :ref:`output-data-type-configuration` for more info.

        .. deprecated:: 0.17
           `output_type` is deprecated in 0.17 and will be removed in 0.18.
           Please use the module level output type control,
           `cuml.global_output_type`.
           See :ref:`output-data-type-configuration` for more info.

    Returns
    -------
    A : sparse graph in CSR format, shape = (n_samples, n_samples_fit)
        n_samples_fit is the number of samples in the fitted data where
        A[i, j] is assigned the weight of the edge that connects i to j.
        Values will either be ones/zeros or the selected distance metric.
        Return types are either cupy's CSR sparse graph (device) or
        numpy's CSR sparse graph (host)

    """

    # Check for deprecated `output_type` and warn. Set manually if specified
    if output_type is not None:
        warnings.warn("Using the `output_type` argument is deprecated and "
                      "will be removed in 0.18. Please specify the output "
                      "type using `cuml.using_output_type()` instead",
                      DeprecationWarning)

    X = NearestNeighbors(n_neighbors, verbose, handle, algorithm, metric, p,
                         metric_params=metric_params,
                         output_type=output_type).fit(X)

    if include_self == 'auto':
        include_self = mode == 'connectivity'

    with cuml.internals.exit_internal_api():
        if not include_self:
            query = None
        else:
            query = X.X_m

    return X.kneighbors_graph(X=query, n_neighbors=n_neighbors, mode=mode)
