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

import ctypes
import math
import numpy as np
import warnings

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libcpp.vector cimport vector

from cuml.common.handle import Handle
from cuml import ForestInference
from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros
cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "treelite/c_api.h":
    ctypedef void* ModelHandle
    ctypedef void* ModelBuilderHandle
    cdef int TreeliteExportProtobufModel(const char* filename,
                                         ModelHandle model)
    cdef const char* TreeliteGetLastError()

cdef extern from "cuml/fil/fil.h" namespace "ML::fil":
    cdef enum algo_t:
        ALGO_AUTO,
        NAIVE,
        TREE_REORG,
        BATCH_TREE_REORG

    cdef enum storage_type_t:
        AUTO,
        DENSE,
        SPARSE

    cdef enum output_t:
        pass

    cdef struct treelite_params_t:
        algo_t algo
        bool output_class
        float threshold
        storage_type_t storage_type

    cdef struct forest:
        pass

    ctypedef forest* forest_t

cdef extern from "cuml/ensemble/randomforest.hpp" namespace "ML":
    cdef enum CRITERION:
        GINI,
        ENTROPY,
        MSE,
        MAE,
        CRITERION_END

cdef extern from "cuml/tree/decisiontree.hpp" namespace "ML::DecisionTree":
    cdef struct DecisionTreeParams:
        int max_depth
        int max_leaves
        float max_features
        int n_bins
        int split_algo
        int min_rows_per_node
        bool bootstrap_features
        bool quantile_per_tree
        CRITERION split_criterion

cdef extern from "cuml/ensemble/randomforest.hpp" namespace "ML":

    cdef enum RF_type:
        CLASSIFICATION,
        REGRESSION

    cdef struct RF_metrics:
        RF_type rf_type
        float accuracy
        double mean_abs_error
        double mean_squared_error
        double median_abs_error

    cdef struct RF_params:
        int n_trees
        bool bootstrap
        float rows_sample
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
                                          int,
                                          int,
                                          vector[unsigned char] &) except +

    cdef vector[unsigned char] save_model_protobuf(ModelHandle) except +

    cdef void print_rf_summary[T, L](RandomForestMetaData[T, L]*) except +
    cdef void print_rf_detailed[T, L](RandomForestMetaData[T, L]*) except +

    cdef RF_params set_rf_class_obj(int,
                                    int,
                                    float,
                                    int,
                                    int,
                                    int,
                                    float,
                                    bool,
                                    bool,
                                    int,
                                    float,
                                    int,
                                    CRITERION,
                                    bool,
                                    int) except +

    cdef vector[unsigned char] save_model(ModelHandle)

    cdef ModelHandle tl_mod_handle(ModelHandle*,
                                   vector[unsigned char]&)

    cdef void predict_mnmg(const cumlHandle&,
                           forest_t,
                           float*, 
                           const float*,
                           size_t num_rows)
