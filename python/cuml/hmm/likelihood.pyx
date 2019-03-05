# cimport c_likelihood
# import numpy as np
# from numba import cuda
# from libcpp cimport bool
# import ctypes
# from libc.stdint cimport uintptr_t
# from c_likelihood cimport *
# # temporary import for numba_utils
# from cuML import numba_utils
#
# import numpy as np
# from libcpp cimport bool
#
# cdef extern from "hmm/magma/b_likelihood.h" namespace "MLCommon":
#
#   cdef void likelihood_batched(int nCl, int nDim,
#                                int nObs,
#                                float** &dX_array, int lddx,
#                                float** &dmu_array, int lddmu,
#                                float** &dsigma_array, int lddsigma_full, int lddsigma,
#                                float* dLlhd, int lddLlhd)
#
#
# class Likelihood:
#     def __init__(self):
#       self.datatype = None
#       self.n_rows = None
#       self.n_cols = None
#
#     def _get_ctype_ptr(self, obj):
#         # The manner to access the pointers in the gdf's might change, so
#         # encapsulating access in the following 3 methods. They might also be
#         # part of future gdf versions.
#         return obj.device_ctypes_pointer.value
#
#     def _get_column_ptr(self, obj):
#         return self._get_ctype_ptr(obj._column._data.to_gpu_array())
#
#     @static_method
#     def to_device(data):
#       return dict((key, cuda.to_device(val)) for key, val in data.items())
#
#     @static_method
#     def _get_ctype_ptrs(self, data_dict) :
#       return dict((key, _get_column_ptr(val)) for key, val in data_dict.items())
#
#     def compute(self, data):
#         self.datatype = X.dtype
#         data_m = self.to_device(data)
#         self.n_rows = X.shape[0]
#         self.n_cols = X.shape[1]
#
#         input_ptrs = self._get_ctype_ptrs(data_m)
#
#         if self.datatype.type == np.float32:
#             c_likelihood.likelihood_batched(nCl, nDim, nObs,
#                                          dX_array, lddx,
#                                          dmu_array, lddmu,
#                                          dsigma_array, lddsigma_full, lddsigma,
#                                          dLlhd, lddLlhd)
#         else:
#             pass
#         return self
