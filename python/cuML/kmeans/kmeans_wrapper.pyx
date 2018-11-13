#
# Copyright (c) 2018, NVIDIA CORPORATION.
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

cimport c_kmeans
import numpy as np
from numba import cuda
import cudf
from libcpp cimport bool
import ctypes
from libc.stdint cimport uintptr_t
from c_kmeans cimport *


class KMeans:

    """
    Create a DataFrame, fill it with data, and compute Kmeans:

    .. code-block:: python

        from cuml import KMeans
        import cudf
        import numpy as np
        import pandas as pd

        def np2cudf(df):
            # convert numpy array to cuDF dataframe
            df = pd.DataFrame({'fea%d'%i:df[:,i] for i in range(df.shape[1])})
            pdf = cudf.DataFrame()
            for c,column in enumerate(df):
              pdf[str(c)] = df[column]
            return pdf


        a = np.asarray([[1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [4.0, 3.0]],dtype=np.float32)
        b = np2cudf(a)
        print("input:")
        print(b)

        print("Calling fit")
        kmeans_float = KMeans(n_clusters=2, n_gpu=-1)
        kmeans_float.fit(b)

        print("labels:")
        print(kmeans_float.labels_)
        print("cluster_centers:")
        print(kmeans_float.cluster_centers_)


    Output:

    .. code-block:: python

          input:

               0    1
           0  1.0  1.0
           1  1.0  2.0
           2  3.0  2.0
           3  4.0  3.0

          Calling fit

          labels:

             0    1
             1    1
             2    0
             3    0

          cluster_centers:

             0    1
          0  3.5  2.5
          1  1.0  1.5


    For additional docs, see `scikitlearn's Kmeans <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_.

    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, verbose=0, random_state=1, precompute_distances='auto', init='kmeans++', n_init=1, algorithm='auto', n_gpu=1, gpu_id=0):
        self.n_clusters = n_clusters
        self.verbose = verbose
        self.random_state = random_state
        self.precompute_distances = precompute_distances
        self.init = init
        self.n_init = n_init
        self.copy_x = None
        self.n_jobs = None
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = None
        self.cluster_centers_ = None
        self.n_gpu = n_gpu
        self.gpu_id = gpu_id

    def _get_ctype_ptr(self, obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def _get_gdf_as_matrix_ptr(self, gdf):
        c = gdf.as_gpu_matrix(order='C').shape
        return self._get_ctype_ptr(gdf.as_gpu_matrix(order='C'))

    def fit(self, X):
        """
        Compute k-means clustering with X.

        Parameters
        ----------
        X : cuDF DataFrame
            Dense matrix (floats or doubles) of shape (n_samples, n_features)

        """

        self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)

        cdef uintptr_t input_ptr
        if (isinstance(X, cudf.DataFrame)):
            self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = X.as_gpu_matrix(order='C')
            self.n_rows = len(X)
            self.n_cols = len(X._cols)

        elif (isinstance(X, np.ndarray)):
            self.gdf_datatype = X.dtype
            X_m = cuda.to_device(X)
            self.n_rows = X.shape[0]
            self.n_cols = X.shape[1]

        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        input_ptr = self._get_ctype_ptr(X_m)

        self.labels_ = cudf.Series(np.zeros(self.n_rows, dtype=np.int32))
        cdef uintptr_t labels_ptr = self._get_column_ptr(self.labels_)

        self.cluster_centers_ = cuda.to_device(np.zeros(self.n_clusters* self.n_cols, dtype=self.gdf_datatype))
        cdef uintptr_t cluster_centers_ptr = self._get_ctype_ptr(self.cluster_centers_)

        if self.gdf_datatype.type == np.float32:
            c_kmeans.make_ptr_kmeans(
                <int> 0,                    # dopredict
                <int> self.verbose,         # verbose
                <int> self.random_state,    # seed
                <int> self.gpu_id,                    # gpu_id
                <int> self.n_gpu,                    # n_gpu
                <size_t> self.n_rows,       # mTrain (rows)
                <size_t> self.n_cols,       # n (cols)
                <char> 'r',            # ord
                <int> self.n_clusters,       # k
                <int> self.n_clusters,       # k_max
                <int> self.max_iter,         # max_iterations
                <int> 1,                     # init_from_data TODO: can use kmeans++
                <float> self.tol,            # threshold
                <float*> input_ptr,    # srcdata
                #<float*> ptr2,   # srcdata
                <float*> 0,           # centroids
                <float*> cluster_centers_ptr, # pred_centroids
                #<float*> 0, # pred_centroids
                <int*> labels_ptr)          # pred_labels
        else:
            c_kmeans.make_ptr_kmeans(
                <int> 0,                    # dopredict
                <int> self.verbose,         # verbose
                <int> self.random_state,    # seed
                <int> self.gpu_id,                    # gpu_id
                <int> self.n_gpu,                    # n_gpu
                <size_t> self.n_rows,       # mTrain (rows)
                <size_t> self.n_cols,       # n (cols)
                <char> 'r',            # ord
                <int> self.n_clusters,       # k
                <int> self.n_clusters,       # k_max
                <int> self.max_iter,         # max_iterations
                <int> 1,                     # init_from_data TODO: can use kmeans++
                <double> self.tol,            # threshold
                <double*> input_ptr,    # srcdata
                <double*> 0,           # centroids
                <double*> cluster_centers_ptr, # pred_centroids
                <int*> labels_ptr)          # pred_labels

        cluster_centers_gdf = cudf.DataFrame()
        for i in range(0, self.n_cols):
            cluster_centers_gdf[str(i)] = self.cluster_centers_[i:self.n_clusters*self.n_cols:self.n_cols]
        self.cluster_centers_ = cluster_centers_gdf

        del(X_m)

        return self

    def fit_predict(self, X):
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : cuDF DataFrame
                    Dense matrix (floats or doubles) of shape (n_samples, n_features)

        """
        return self.fit(X).labels_

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : cuDF DataFrame
                    Dense matrix (floats or doubles) of shape (n_samples, n_features)

        """
        self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)

        cdef uintptr_t input_ptr
        if (isinstance(X, cudf.DataFrame)):
            self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = X.as_gpu_matrix(order='C')
            self.n_rows = len(X)
            self.n_cols = len(X._cols)

        elif (isinstance(X, np.ndarray)):
            self.gdf_datatype = X.dtype
            X_m = cuda.to_device(X)
            self.n_rows = X.shape[0]
            self.n_cols = X.shape[1]

        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        input_ptr = self._get_ctype_ptr(X_m)

        clust_mat = self.cluster_centers_.as_gpu_matrix(order='C')
        cdef uintptr_t cluster_centers_ptr = self._get_ctype_ptr(clust_mat)

        self.labels_ = cudf.Series(np.zeros(self.n_rows, dtype=np.int32))
        cdef uintptr_t labels_ptr = self._get_column_ptr(self.labels_)

        if self.gdf_datatype.type == np.float32:
            c_kmeans.make_ptr_kmeans(
                <int> 1,                    # dopredict
                <int> self.verbose,                    # verbose
                <int> self.random_state,                    # seed
                <int> self.gpu_id,                    # gpu_id
                <int> self.n_gpu,                    # n_gpu
                <size_t> self.n_rows,       # mTrain (rows)
                <size_t> self.n_cols,       # n (cols)
                <char> 'r',            # ord
                <int> self.n_clusters,       # k
                <int> self.n_clusters,       # k_max
                <int> self.max_iter,         # max_iterations
                <int> 0,                     # init_from_data TODO: can use kmeans++
                <float> self.tol,            # threshold
                #<float*> input_ptr,   # srcdata
                <float*> input_ptr,    # srcdata
                #<float*> ptr2,   # srcdata
                <float*> cluster_centers_ptr,    # centroids
                <float*> 0, # pred_centroids
                <int*> labels_ptr)          # pred_labels
        else:
            c_kmeans.make_ptr_kmeans(
                <int> 1,                    # dopredict
                <int> self.verbose,                    # verbose
                <int> self.random_state,                    # seed
                <int> self.gpu_id,                    # gpu_id
                <int> self.n_gpu,                    # n_gpu
                <size_t> self.n_rows,       # mTrain (rows)
                <size_t> self.n_cols,       # n (cols)
                <char> 'r',            # ord
                <int> self.n_clusters,       # k
                <int> self.n_clusters,       # k_max
                <int> self.max_iter,         # max_iterations
                <int> 0,                     # init_from_data TODO: can use kmeans++
                <double> self.tol,            # threshold
                <double*> input_ptr,    # srcdata
                <double*> cluster_centers_ptr, # centroids
                <double*> 0, # pred_centroids
                <int*> labels_ptr)          # pred_labels

        del(X_m)
        del(clust_mat)
        return self.labels_

    def transform(self, X):
        """
        Transform X to a cluster-distance space.

        Parameters
        ----------
        X : cuDF DataFrame
                    Dense matrix (floats or doubles) of shape (n_samples, n_features)

        """

        cdef uintptr_t input_ptr
        if (isinstance(X, cudf.DataFrame)):
            self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = X.as_gpu_matrix(order='C')
            self.n_rows = len(X)
            self.n_cols = len(X._cols)

        elif (isinstance(X, np.ndarray)):
            self.gdf_datatype = X.dtype
            X_m = cuda.to_device(X)
            self.n_rows = X.shape[0]
            self.n_cols = X.shape[1]

        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        input_ptr = self._get_ctype_ptr(X_m)

        clust_mat = self.cluster_centers_.as_gpu_matrix(order='C')
        cdef uintptr_t cluster_centers_ptr = self._get_ctype_ptr(clust_mat)

        preds_data = cuda.to_device(np.zeros(self.n_clusters*self.n_rows,
                                    dtype=self.gdf_datatype.type))

        cdef uintptr_t preds_ptr = self._get_ctype_ptr(preds_data)

        if self.gdf_datatype.type == np.float32:
            c_kmeans.kmeans_transform(
                <int> self.verbose,                    # verbose
                <int> self.gpu_id,                    # gpu_id
                <int> self.n_gpu,                    # n_gpu
                <size_t> self.n_rows,       # mTrain (rows)
                <size_t> self.n_cols,       # n (cols)
                <char> 'r',            # ord
                <int> self.n_clusters,       # k
                <float*> input_ptr,    # srcdata
                <float*> cluster_centers_ptr,    # centroids
                <float*> preds_ptr)          # preds
        else:
            c_kmeans.kmeans_transform(
                <int> self.verbose,                    # verbose
                <int> self.gpu_id,                    # gpu_id
                <int> self.n_gpu,                    # n_gpu
                <size_t> self.n_rows,       # mTrain (rows)
                <size_t> self.n_cols,       # n (cols)
                <char> 'r',            # ord
                <int> self.n_clusters,       # k
                <double*> input_ptr,    # srcdata
                <double*> cluster_centers_ptr,    # centroids
                <double*> preds_ptr)          # preds

        preds_gdf = cudf.DataFrame()
        for i in range(0, self.n_clusters):
            preds_gdf[str(i)] = preds_data[i*self.n_rows:(i+1)*self.n_rows]

        del(X_m)
        del(clust_mat)
        return preds_gdf

    def fit_transform(self, input_gdf):
        """
        Compute clustering and transform input_gdf to cluster-distance space.

        Parameters
        ----------
        input_gdf : cuDF DataFrame
                    Dense matrix (floats or doubles) of shape (n_samples, n_features)

        """
        return self.fit(input_gdf).transform(input_gdf)
