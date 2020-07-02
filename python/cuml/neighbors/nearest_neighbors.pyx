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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numpy as np
import pandas as pd
import cupy as cp
import cudf
import ctypes
import cuml
import warnings

from cuml.common.base import Base
from cuml.common.array import CumlArray
from cuml.common import input_to_cuml_array

from cython.operator cimport dereference as deref

from cuml.common.handle cimport cumlHandle

from libcpp cimport bool
from libcpp.memory cimport shared_ptr

import rmm
from libc.stdlib cimport malloc, free

from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport calloc, malloc, free

from libcpp.vector cimport vector


from numba import cuda
import rmm

cimport cuml.common.handle
cimport cuml.common.cuda


cdef extern from "cuml/cuml.hpp" namespace "ML" nogil:
    cdef cppclass deviceAllocator:
        pass

    cdef cppclass cumlHandle:
        cumlHandle() except +
        void setStream(cuml.common.cuda._Stream s) except +
        void setDeviceAllocator(shared_ptr[deviceAllocator] a) except +
        cuml.common.cuda._Stream getStream() except +

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
        cumlHandle &handle,
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


class NearestNeighbors(Base):
    """
    NearestNeighbors is an queries neighborhoods from a given set of
    datapoints. Currently, cuML supports k-NN queries, which define
    the neighborhood as the closest `k` neighbors to each query point.

    Parameters
    ----------
    n_neighbors : int (default=5)
        Default number of neighbors to query
    verbose : int or boolean (default = False)
        Logging level
    handle : cumlHandle
        The cumlHandle resources to use
    algorithm : string (default='brute')
        The query algorithm to use. Currently, only 'brute' is supported.
    metric : string (default='euclidean').
        Distance metric to use. Supported distances are ['l1, 'cityblock',
        'taxicab', 'manhattan', 'euclidean', 'l2', 'braycurtis', 'canberra',
        'minkowski', 'chebyshev', 'jensenshannon', 'cosine', 'correlation']
    p : float (default=2) Parameter for the Minkowski metric. When p = 1, this
        is equivalent to manhattan distance (l1), and euclidean distance (l2)
        for p = 2. For arbitrary p, minkowski distance (lp) is used.
    metric_expanded : bool
        Can increase performance in Minkowski-based (Lp) metrics (for p > 1)
        by using the expanded form and not computing the n-th roots.
    metric_params : dict, optional (default = None) This is currently ignored.

    Examples
    ---------
    .. code-block:: python

      import cudf
      from cuml.neighbors import NearestNeighbors
      from cuml.datasets import make_blobs

      X, _ = make_blobs(n_samples=25, centers=5,
                        n_features=10, random_state=42)

      # build a cudf Dataframe
      X_cudf = cudf.DataFrame.from_gpu_matrix(X)

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
    <https://github.com/rapidsai/cuml/blob/branch-0.14/notebooks/nearest_neighbors_demo.ipynb>`_.

    For additional docs, see `scikit-learn's NearestNeighbors
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors>`_.
    """
    def __init__(self,
                 n_neighbors=5,
                 verbose=False,
                 handle=None,
                 algorithm="brute",
                 metric="euclidean",
                 p=2,
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
        self.p = p
        self.algorithm = algorithm

    def fit(self, X, convert_dtype=True):
        """
        Fit GPU index for performing nearest neighbor queries.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool, optional (default = True)
            When set to True, the fit method will automatically
            convert the inputs to np.float32.
        """
        self._set_n_features_in(X)
        self._set_output_type(X)

        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        self.n_dims = X.shape[1]

        self.X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, order='F', check_dtype=np.float32,
                                convert_to_dtype=(np.float32
                                                  if convert_dtype
                                                  else None))

        self.n_rows = n_rows
        self.n_indices = 1

        return self

    def get_param_names(self):
        return ["n_neighbors", "algorithm", "metric",
                "p", "metric_params"]

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
        else:
            raise ValueError("Metric %s is not supported" % metric)

        return m, expanded

    def kneighbors(self, X=None, n_neighbors=None,
                   return_distance=True, convert_dtype=True):
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

        Returns
        -------
        distances: cuDF DataFrame or numpy ndarray
            The distances of the k-nearest neighbors for each column vector
            in X

        indices: cuDF DataFrame or numpy ndarray
            The indices of the k-nearest neighbors for each column vector in X
        """

        n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors
        X = self.X_m if X is None else X

        out_type = self._get_output_type(X)

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

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef uintptr_t x_ctype_st = X_m.ptr

        metric, expanded = self._build_metric_type(self.metric)

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

        self.handle.sync()

        return (D_ndarr.to_output(out_type), I_ndarr.to_output(out_type)) \
            if return_distance else I_ndarr.to_output(out_type)

    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        """
        Find the k nearest neighbors of column vectors in X and return as 
        a sparse matrix in CSR format.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        n_neighbors : Integer
            Number of neighbors to search. If not provided, the n_neighbors
            from the model instance is used (default=10)

        mode : string (default='connectivity')
            Values in connectivity matrix: 'connectivity' returns the
            connectivity matrix with ones and zeros, 'distance' returns the
            edges as the Euclidean distance between points

        Returns
        -------
        A: sparse graph in CSR format, shape = (n_samples, n_samples_fit)
            n_samples_fit is the number of samples in the fitted data where 
            A[i, j] is assigned the weight of the edge that connects i to k.
            Values will be ones/zeros or Euclidean distance based on mode.

        """
        # Check if not fitted by checking if certain attribute is filled, otherwise throw error not fitted
        if n_neighbors is None:
            n_neighbors = self.n_neighbors 

        if mode == 'connectivity':
            indices = self.kneighbors(X, n_neighbors, return_distance=False) # cuDF DataFrame or numpy ndarray
            n_samples = indices.shape[0]
            distances = cp.ones(n_samples * n_neighbors) # returns cupy.ndarray

        elif mode == 'distance':
            distances, indices = self.kneighbors(X, n_neighbors) # cuDF DataFrames or numpy ndarrays

            if isinstance(distances, cudf.DataFrame):
                distances = cp.asarray(distances.as_gpu_matrix())
            else:
                distances = cp.array(distances)
            distances = cp.ravel(distances)

        else:
            raise ValueError('Unsupported mode, must be one of "connectivity" '
                'or "distance" but got "%s" instead' % mode)
        
        if isinstance(indices, cudf.DataFrame):
            indices = cp.asarray(indices.as_gpu_matrix())
        else:
            indices = cp.array(indices)
        n_samples = distances.shape[0]
        n_samples_fit = self.n_rows
        n_nonzero = n_samples * n_neighbors
        rowptr = cp.arange(0, n_nonzero + 1, n_neighbors)
        return cp.sparse.csr_matrix((distances, cp.ravel(indices), rowptr), shape=(n_samples, n_samples_fit))

# Try implementing wrapper component here
def kneighbors_graph(X=None, n_neighbors=5, mode='connectivity', verbose=False, handle=None, algorithm="brute", metric="euclidean", p=2, include_self=False, metric_expanded=False, metric_params=None):
        
    # check if class NearestNeighbors already instantiated, if not then instantiate
    if not isinstance(X, NearestNeighbors):
        X = NearestNeighbors(n_neighbors=n_neighbors, verbose=verbose, handle=handle, algorithm=algorithm, metric=metric, p=p, metric_expanded=metric_expanded, metric_params=metric_params).fit(X)

    if include_self == 'auto':
        include_self = mode == 'connectivity'

    # it does not include each sample as its own neighbors
    if not include_self:
        query = None
    else:
        query = X

    return X.kneighbors_graph(X=query, n_neighbors=n_neighbors, mode=mode)