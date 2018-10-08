import numpy as np
cimport numpy as np
from libcpp cimport bool
np.import_array

cdef extern from "dbscan/dbscan_c.h" namespace "ML":

    cdef void dbscanFit(float *input, 
                   int n_rows, 
                   int n_cols, 
                   float eps, 
                   int min_pts,
		           int *labels)

    cdef void dbscanFit(double *input, 
                   int n_rows, 
                   int n_cols, 
                   double eps, 
                   int min_pts,
		           int *labels)


 