cimport c_kmeans
import numpy as np
from numba import cuda
import pygdf
from libcpp cimport bool
import ctypes
from libc.stdint cimport uintptr_t
from c_kmeans cimport *


class KMeans:

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

    def fit(self, input_gdf):
        x = []
        for col in input_gdf.columns:
            x.append(input_gdf[col]._column.dtype)
            break

        self.gdf_datatype = np.dtype(x[0])
        self.n_rows = len(input_gdf)
        self.n_cols = len(input_gdf._cols)
        
        cdef np.ndarray[np.float32_t, ndim=2, mode = 'c', cast=True] host_ary = input_gdf.as_gpu_matrix(order='C').copy_to_host()
        
        self.labels_ = pygdf.Series(np.zeros(self.n_rows, dtype=np.int32))
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
                <float*> host_ary.data,    # srcdata
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
                <double*> host_ary.data,    # srcdata
                <double*> 0,           # centroids
                <double*> cluster_centers_ptr, # pred_centroids
                <int*> labels_ptr)          # pred_labels
        
        cluster_centers_gdf = pygdf.DataFrame()
        for i in range(0, self.n_cols):
            cluster_centers_gdf[str(i)] = self.cluster_centers_[i:self.n_clusters*self.n_cols:self.n_cols]
        self.cluster_centers_ = cluster_centers_gdf

        return self

    def fit_predict(self, input_gdf):
        return self.fit(input_gdf).labels_

    def predict(self, input_gdf):
        x = []
        for col in input_gdf.columns:
            x.append(input_gdf[col]._column.dtype)
            break

        self.gdf_datatype = np.dtype(x[0])
        self.n_rows = len(input_gdf)
        self.n_cols = len(input_gdf._cols)

        #cdef uintptr_t input_ptr = self._get_gdf_as_matrix_ptr(input_gdf)
        cdef np.ndarray[np.float32_t, ndim=2, mode = 'c', cast=True] host_ary = input_gdf.as_gpu_matrix(order='C').copy_to_host()
        self.labels_ = pygdf.Series(np.zeros(self.n_rows, dtype=np.int32))
        cdef uintptr_t labels_ptr = self._get_column_ptr(self.labels_)

        #pred_centers = pygdf.Series(np.zeros(self.n_clusters* self.n_cols, dtype=self.gdf_datatype))
        cdef uintptr_t cluster_centers_ptr = self._get_gdf_as_matrix_ptr(self.cluster_centers_)

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
                <float*> host_ary.data,    # srcdata
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
                <double*> host_ary.data,    # srcdata
                <double*> cluster_centers_ptr, # centroids
                <double*> 0, # pred_centroids
                <int*> labels_ptr)          # pred_labels

        return self.labels_


    def transform(self, input_gdf):
        x = []
        for col in input_gdf.columns:
            x.append(input_gdf[col]._column.dtype)
            break

        self.gdf_datatype = np.dtype(x[0])
        self.n_rows = len(input_gdf)
        self.n_cols = len(input_gdf._cols)

        cdef np.ndarray[np.float32_t, ndim=2, mode = 'c', cast=True] host_ary = input_gdf.as_gpu_matrix(order='C').copy_to_host()
        preds_data = cuda.to_device(np.zeros(self.n_clusters*self.n_rows,
                                       dtype=self.gdf_datatype.type))

        cdef uintptr_t preds_ptr = self._get_ctype_ptr(preds_data)


        ary=np.array([1.0,1.5,3.5,2.5],dtype=np.float32)
        dary=cuda.to_device(ary)
        cdef uintptr_t ptr2 = dary.device_ctypes_pointer.value
        cdef uintptr_t cluster_centers_ptr = self._get_gdf_as_matrix_ptr(self.cluster_centers_)

        if self.gdf_datatype.type == np.float32:
            c_kmeans.kmeans_transform(
                <int> self.verbose,                    # verbose
                <int> self.gpu_id,                    # gpu_id
                <int> self.n_gpu,                    # n_gpu
                <size_t> self.n_rows,       # mTrain (rows)
                <size_t> self.n_cols,       # n (cols)
                <char> 'r',            # ord
                <int> self.n_clusters,       # k
                <float*> host_ary.data,    # srcdata
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
                <double*> host_ary.data,    # srcdata
                <double*> cluster_centers_ptr,    # centroids
                <double*> preds_ptr)          # preds

        preds_gdf = pygdf.DataFrame()
        for i in range(0, self.n_clusters):
            preds_gdf[str(i)] = preds_data[i*self.n_rows:(i+1)*self.n_rows]
        
        return preds_gdf


    def fit_transform(self, input_gdf):
        return self.fit(input_gdf).transform(input_gdf)
