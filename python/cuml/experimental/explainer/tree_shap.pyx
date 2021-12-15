#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

from cuml.common import input_to_cuml_array
from cuml.common.array import CumlArray
from cuml.common.input_utils import determine_array_type
from cuml.common.exceptions import NotFittedError
from cuml.fil.fil import TreeliteModel
from cuml.ensemble import RandomForestRegressor as curfr
from cuml.ensemble import RandomForestClassifier as curfc

from libcpp.memory cimport unique_ptr
from libc.stdint cimport uintptr_t
from libcpp.utility cimport move
import numpy as np
import treelite

cdef extern from "treelite/c_api.h":
    ctypedef void* ModelHandle
    cdef int TreeliteQueryNumClass(ModelHandle handle, size_t* out)

cdef extern from "treelite/c_api_common.h":
    cdef const char* TreeliteGetLastError()

cdef extern from "cuml/explainer/tree_shap.hpp" namespace "ML::Explainer":
    cdef cppclass TreePathInfo:
        pass

    cdef unique_ptr[TreePathInfo] extract_path_info(ModelHandle model) except +
    cdef void gpu_treeshap(const TreePathInfo* path_info,
                           const float* data,
                           size_t n_rows,
                           size_t n_cols,
                           float* out_preds) except +

cdef class TreeExplainer_impl():
    cdef ModelHandle model_ptr
    cdef unique_ptr[TreePathInfo] path_info
    cdef size_t num_class
    cdef public object expected_value

    def __cinit__(self, handle=None):
        self.model_ptr = <ModelHandle> <uintptr_t> handle
        self.num_class = 0
        if TreeliteQueryNumClass(self.model_ptr, &self.num_class) != 0:
            raise RuntimeError('Treelite error: {}'.format(
                TreeliteGetLastError()))
        self.path_info = move(extract_path_info(self.model_ptr))

    def shap_values(self, X):
        cdef uintptr_t X_ptr
        cdef uintptr_t preds_ptr
        X_type = determine_array_type(X)
        # Coerce to CuPy / NumPy because we may need to return 3D array
        output_type = 'numpy' if X_type == 'numpy' else 'cupy'
        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, order='C', convert_to_dtype=np.float32)
        # Storing a C-order 3D array in a CumlArray leads to cryptic error
        # ValueError: len(shape) != len(strides)
        # So we use 2D array here
        pred_shape = (n_rows, self.num_class * (n_cols + 1))
        preds = CumlArray.empty(shape=pred_shape, dtype=np.float32, order='C')
        X_ptr = X_m.ptr
        preds_ptr = preds.ptr
        gpu_treeshap(self.path_info.get(), <const float*> X_ptr,
                     <size_t> n_rows, <size_t> n_cols, <float*> preds_ptr)
        # Reshape to 3D as appropriate
        # To follow the convention of the SHAP package:
        # 1. Store the bias term in the 'expected_value' attribute.
        # 2. Transpose SHAP values in dimension (group_id, row_id, feature_id)
        preds = preds.to_output(output_type=output_type)
        if self.num_class > 1:
            preds = preds.reshape((n_rows, self.num_class, n_cols + 1))
            preds = preds.transpose((1, 0, 2))
            self.expected_value = preds[:, 0, -1]
            return preds[:, :, :-1]
        else:
            assert self.num_class == 1
            self.expected_value = preds[0, -1]
            return preds[:, :-1]


class TreeExplainer:
    """
    Explainer for tree models, using GPUTreeSHAP
    """
    def __init__(self, *, model):
        # Handle various kinds of tree model objects
        cls = model.__class__
        # XGBoost model object
        if cls.__module__ == 'xgboost.core' and cls.__name__ == 'Booster':
            model = treelite.Model.from_xgboost(model)
            handle = model.handle.value
        # cuML RF model object
        elif isinstance(model, curfr):
            try:
                model = model.convert_to_treelite_model()
            except NotFittedError as e:
                raise NotFittedError(
                        'Cannot compute SHAP for un-fitted model') from e
            handle = model.handle
        elif isinstance(model, curfc):
            raise NotImplementedError(
                'cuML RF classifiers are not supported yet')
        elif isinstance(model, treelite.Model):
            handle = model.handle.value
        elif isinstance(model, TreeliteModel):
            handle = model.handle
        else:
            raise ValueError('Unrecognized model object type')
        self.impl_ = TreeExplainer_impl(handle=handle)

    def shap_values(self, X) -> CumlArray:
        """
        Interface to estimate the SHAP values for a set of samples.
        """
        out = self.impl_.shap_values(X)
        self.expected_value = self.impl_.expected_value
        return out
