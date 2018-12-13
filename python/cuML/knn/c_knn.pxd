import numpy as np
cimport numpy as np
from libcpp cimport bool
np.import_array

cdef extern from "knn/knn_c.h" namespace "ML":

    cdef cppclass kNN:
	kNN() except +
	void search(float *search_items, int search_items_size, long *res_I, float *res_D, int k); 
