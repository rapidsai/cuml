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

import numpy as np
from libcpp cimport bool

cdef extern from "knn/knn.h" namespace "ML":

    cdef cppclass kNNParams:
        float *ptr,
        int N

    cdef cppclass kNN:
        kNN(int D) except +
        void search(const float *search_items,
                    int search_items_size,
                    long *res_I,
                    float *res_D,
                    int k)
        void fit(kNNParams *input,
                 int N)
