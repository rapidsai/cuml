#
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
from cuml.common.import_utils import has_sklearn
from cuml.common.input_utils import determine_array_type
from cuml.fil.fil import TreeliteModel
from cuml.ensemble import RandomForestRegressor as curfr
from cuml.ensemble import RandomForestClassifier as curfc

from libcpp.memory cimport shared_ptr
from libc.stdint cimport uintptr_t
from libcpp.utility cimport move
import re
import numpy as np
import treelite

if has_sklearn():
    from sklearn.ensemble import RandomForestRegressor as sklrfr
    from sklearn.ensemble import RandomForestClassifier as sklrfc
else:
    sklrfr = object
    sklrfc = object

cdef extern from "treelite/c_api.h":
    ctypedef void * ModelHandle
    cdef int TreeliteQueryNumClass(ModelHandle handle, size_t * out)

cdef extern from "treelite/c_api_common.h":
    cdef const char * TreeliteGetLastError()

cdef extern from "cuml/explainer/tree_shap.hpp" namespace "ML::Explainer":
    cdef cppclass TreePathInfo:
        pass

    cdef shared_ptr[TreePathInfo] extract_path_info(ModelHandle model) except +
    cdef void gpu_treeshap(TreePathInfo * path_info,
                           const float * data,
                           size_t n_rows,
                           size_t n_cols,
                           float * out_preds) except +
    cdef void gpu_treeshap(TreePathInfo * path_info,
                           const double * data,
                           size_t n_rows,
                           size_t n_cols,
                           double * out_preds) except +


cdef class TreeExplainer:
    cdef public object expected_value
    cdef shared_ptr[TreePathInfo] path_info
    cdef size_t num_class
    """ 
    Model explainer that calculates Shapley values for the predictions of tree-based models. Shapley values are a method of attributing the contribution of various input features to a given model prediction. 
    
    Uses GPUTreeShap as a back-end to accelerate computation using GPUs.

    Mitchell, Rory, Eibe Frank, and Geoffrey Holmes.
    "GPUTreeShap: Massively Parallel Exact Calculation of SHAP Scores for Tree Ensembles."
    arXiv preprint arXiv:2010.13972 (2020).

    Different variants of Shapley values exist based on different interpretations of marginalising out (or conditioning on) features. For the "tree_path_dependent" approach, see:
    Lundberg, Scott M., et al. "From local explanations to global understanding with explainable AI for trees." Nature machine intelligence 2.1 (2020): 56-67.

    For the "interventional" approach see:
    Janzing, Dominik, Lenon Minorics, and Patrick Blöbaum. "Feature relevance quantification in explainable AI: A causal problem." International Conference on artificial intelligence and statistics. PMLR, 2020.

    We also provide two variants of feature interactions. For the "standard" variant of interactions:
    Lundberg, Scott M., et al. "From local explanations to global understanding with explainable AI for trees." Nature machine intelligence 2.1 (2020): 56-67.

    For the "taylor" variant, see:
    Sundararajan, Mukund, Kedar Dhamdhere, and Ashish Agarwal. "The Shapley Taylor Interaction Index." International Conference on Machine Learning. PMLR, 2020. 


    Parameters
    ----------
    model : model object
        The tree based machine learning model. XGBoost, LightGBM, cuml random forest and sklearn random forest models are supported. Categorical features in XGBoost or LightGBM models are natively supported.
    data : array or DataFrame
        Optional background dataset to use for integrating out features. This argument is used with the "interventional" feature perturbation method. Computation time increases with the size of this background data set, consider starting with between 100-1000 examples.
    feature_perturbation : "interventional" (default when data is specified) or "tree_path_dependent" (default when data=None).
        Method of conditioning on features. See the above references for more information.

    Example
    --------
    """

    def __init__(self, *, model):
        # Handle various kinds of tree model objects
        cls = model.__class__
        cls_module, cls_name = cls.__module__, cls.__name__
        # XGBoost model object
        if re.match(r'xgboost.*$', cls_module) and cls_name == 'Booster':
            model = treelite.Model.from_xgboost(model)
            handle = model.handle.value
        # LightGBM model object
        if re.match(r'lightgbm.*$', cls_module) and cls_name == 'Booster':
            model = treelite.Model.from_lightgbm(model)
            handle = model.handle.value
        # cuML RF model object
        elif isinstance(model, (curfr, curfc)):
            model = model.convert_to_treelite_model()
            handle = model.handle
        # scikit-learn RF model object
        elif isinstance(model, (sklrfr, sklrfc)):
            model = treelite.sklearn.import_model(model)
            handle = model.handle.value
        elif isinstance(model, treelite.Model):
            handle = model.handle.value
        elif isinstance(model, TreeliteModel):
            handle = model.handle
        else:
            raise ValueError('Unrecognized model object type')

        cdef ModelHandle model_ptr = <ModelHandle > <uintptr_t > handle
        self.num_class = 0
        if TreeliteQueryNumClass(model_ptr, & self.num_class) != 0:
            raise RuntimeError('Treelite error: {}'.format(
                TreeliteGetLastError()))
        self.path_info = extract_path_info(model_ptr)

    def shap_values(self, X) -> CumlArray:
        """ 
        Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : 
            A matrix of samples (# samples x # features) on which to explain the model's output.

        Returns
        -------
        array
            Returns a matrix of SHAP values of shape (# classes x # samples x # features).
        """
        X_type = determine_array_type(X)
        # Coerce to CuPy / NumPy because we may need to return 3D array
        output_type = 'numpy' if X_type == 'numpy' else 'cupy'
        try:
            X_m, n_rows, n_cols, dtype = input_to_cuml_array(
                X, order='C', check_dtype=[np.float32, np.float64])
        except ValueError:
            # input can be a DataFrame with mixed types
            # in this case coerce to 64-bit
            X_m, n_rows, n_cols, dtype = input_to_cuml_array(
                X, order='C', convert_to_dtype=np.float64)

        # Storing a C-order 3D array in a CumlArray leads to cryptic error
        # ValueError: len(shape) != len(strides)
        # So we use 2D array here
        pred_shape = (n_rows, self.num_class * (n_cols + 1))
        preds = CumlArray.empty(shape=pred_shape, dtype=dtype, order='C')
        if dtype == np.float32:
            gpu_treeshap(self.path_info.get(), < const float*> < uintptr_t > X_m.ptr,
                         < size_t > n_rows, < size_t > n_cols, < float*> < uintptr_t > preds.ptr)
        elif dtype == np.float64:
            gpu_treeshap(self.path_info.get(), < const double * > < uintptr_t > X_m.ptr,
                         < size_t > n_rows, < size_t > n_cols, < double * > < uintptr_t > preds.ptr)
        else:
            raise ValueError("Unsupported dtype")

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
