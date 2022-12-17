#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

import ctypes
import math
import numpy as np
import warnings

from libcpp cimport bool
from libc.stdint cimport uintptr_t, uint64_t
from libc.stdlib cimport calloc, malloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string

from pylibraft.common.handle import Handle
from cuml import ForestInference
from cuml.internals.base import Base
from pylibraft.common.handle cimport handle_t
cimport cuml.common.cuda

cdef extern from "treelite/c_api.h":
    ctypedef void* ModelHandle
    ctypedef void* ModelBuilderHandle
    cdef const char* TreeliteGetLastError()

cdef extern from "cuml/ensemble/randomforest.hpp" namespace "ML":
    cdef enum CRITERION:
        GINI,
        ENTROPY,
        MSE,
        MAE,
        POISSON,
        GAMMA,
        INVERSE_GAUSSIAN,
        CRITERION_END

cdef extern from "cuml/ensemble/randomforest.hpp" namespace "ML":

    cdef enum RF_type:
        CLASSIFICATION,
        REGRESSION

    cdef enum task_category:
        REGRESSION_MODEL = 1,
        CLASSIFICATION_MODEL = 2

    cdef struct RF_metrics:
        RF_type rf_type
        float accuracy
        double mean_abs_error
        double mean_squared_error
        double median_abs_error

    cdef struct RF_params:
        int n_trees
        bool bootstrap
        float max_samples
        int seed
        pass

    cdef cppclass RandomForestMetaData[T, L]:
        void* trees
        RF_params rf_params

    #
    # Treelite handling
    #
    cdef void build_treelite_forest[T, L](ModelHandle*,
                                          RandomForestMetaData[T, L]*,
                                          int
                                          ) except +

    cdef void delete_rf_metadata[T, L](RandomForestMetaData[T, L]*) except +

    #
    # Text representation of random forest
    #
    cdef string get_rf_summary_text[T, L](RandomForestMetaData[T, L]*) except +
    cdef string get_rf_detailed_text[T, L](RandomForestMetaData[T, L]*
                                           ) except +
    cdef string get_rf_json[T, L](RandomForestMetaData[T, L]*) except +

    cdef RF_params set_rf_params(int,
                                 int,
                                 float,
                                 int,
                                 int,
                                 int,
                                 float,
                                 bool,
                                 int,
                                 float,
                                 uint64_t,
                                 CRITERION,
                                 int,
                                 int) except +

    cdef vector[unsigned char] save_model(ModelHandle)

    cdef ModelHandle concatenate_trees(
        vector[ModelHandle] &treelite_handles) except +
