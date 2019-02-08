include "pca/pca_wrapper.pyx"
include "tsvd/tsvd_wrapper.pyx"
include "dbscan/dbscan_wrapper.pyx"
include "kmeans/kmeans_wrapper.pyx"
include "linear_model/linear_regression.pyx"
include "knn/knn_wrapper.pyx"
include "kalman/kalman_filter.pyx"
include "linear_model/ridge.pyx"


cdef extern from "ml_mg_utils.h" namespace "ML::MLCommon":
   cdef int get_device(void *ptr)

def device_from_ptr(ptr):
   return get_device(<void*>ptr)
