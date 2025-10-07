#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
import warnings

import numpy as np

from libc.stdint cimport uint64_t, uintptr_t
from libc.stdlib cimport calloc, free, malloc
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from pylibraft.common.handle import Handle

from cuml.internals.base import Base

from pylibraft.common.handle cimport handle_t

from cuml.internals.treelite cimport *


cdef extern from "cuml/ensemble/randomforest.hpp" namespace "ML" nogil:
    cdef enum CRITERION:
        GINI,
        ENTROPY,
        MSE,
        MAE,
        POISSON,
        GAMMA,
        INVERSE_GAUSSIAN,
        CRITERION_END

cdef extern from "cuml/ensemble/randomforest.hpp" namespace "ML" nogil:

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
        bool oob_score
        bool compute_feature_importance
        pass

    cdef cppclass RandomForestMetaData[T, L]:
        void* trees
        RF_params rf_params
        double oob_score
        vector[T] feature_importances

    #
    # Treelite handling
    #
    cdef void build_treelite_forest[T, L](TreeliteModelHandle*,
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
                                 int,
                                 bool,
                                 bool) except +

    cdef double get_oob_score[T, L](RandomForestMetaData[T, L]*) except +
    cdef vector[T] get_feature_importances[T, L](RandomForestMetaData[T, L]*) except +

    cdef vector[unsigned char] save_model(TreeliteModelHandle)
