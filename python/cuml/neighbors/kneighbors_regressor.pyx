#
# Copyright (c) 2019, NVIDIA CORPORATION.
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

from cuml.neighbors.nearest_neighbors import NearestNeighbors

from cuml.common.handle cimport cumlHandle


from libcpp cimport bool
from libcpp.memory cimport shared_ptr

import rmm
from libc.stdlib cimport malloc, free

from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport calloc, malloc, free

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

    void knn_regress(
        cumlHandle &handle,
        float *out,
        int64_t *knn_indices,
        float *y,
        size_t n_samples,
        int k,
    ) except +

class KNeighborsRegressor(NearestNeighbors):

    def __init__(self, **kwargs):
        """

        :param kwargs:
        """
        super(KNeighborsRegressor, self).__init__(**kwargs)
        self.y = None

    def fit(self, X, y, convert_dtype=True):
        """
        Fit a k-nearest neighbors classifier model.
        :param X:
        :param y:
        :param convert_dtype:
        :return:
        """
        super(NearestNeighbors, self).fit(X, convert_dtype)
        self.y = y

        self.handle.sync()

    def predict(self, X, convert_dtype=True):
        """
        Use the trained k-nearest neighbors classifier to
        predict the labels for X
        :param X:
        :param convert_type:
        :return:
        """

        knn_indices = self.kneighbors(X, convert_dtype)



        self.handle.sync()
        pass


    def score(self, X, y, sample_weight=None, convert_dtype=True):
        """
        Compute the R^2 score using the given labels and
        the trained k-nearest neighbors classifier to predict
        the classes for X.
        :param X:
        :param y:
        :param sample_weight:
        :return:
        """

        labels = self.predict(X, y, convert_dtype)
        # Compute the coefficient of determination
