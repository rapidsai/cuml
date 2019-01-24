import numpy as np
cimport numpy as np
from libcpp cimport bool
np.import_array

cdef extern from "knn/knn_c.h" namespace "ML":
    cdef cppclass kNNParams:
        float *ptr
        int N

cdef extern from "knn/knn_c.h" namespace "ML":
    cdef cppclass kNN:
        kNN(int D) except +
        void search(float *search_items,    # ctypes pointer to search items array on device
                    int search_items_size,  # number of rows in search items array
                    long *res_I,            # ctypes pointer to output indices array on device
                    float *res_D,           # ctypes pointer to output distance array on device
                    int k)
        void fit(kNNParams *input,          # array of knnparams to describe multi-GPU training inputs
                 int N)                     # number of items in input array
