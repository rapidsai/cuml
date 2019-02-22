include "pca/pca_wrapper.pyx"
include "tsvd/tsvd_wrapper.pyx"
include "dbscan/dbscan_wrapper.pyx"
include "kmeans/kmeans_wrapper.pyx"
include "linear_model/linear_regression.pyx"
include "knn/knn_wrapper.pyx"
include "kalman/kalman_filter.pyx"
include "linear_model/ridge.pyx"

cdef extern from "ml_cuda_utils.h" namespace "ML":
   cdef int get_device(void *ptr)

def device_of_ptr(g):
   cdef uintptr_t cptr = g.device_ctypes_pointer.value
   return get_device( <void*> cptr)