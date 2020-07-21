#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import copy
import cudf
import ctypes
import math
import numpy as np
import warnings
import pandas as pd

import rmm

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.common import input_to_cuml_array

import treelite
import treelite.sklearn as tl_skl

cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "treelite/c_api.h":
    ctypedef void* ModelHandle
    cdef int TreeliteLoadXGBoostModel(const char* filename,
                                      ModelHandle* out) except +
    cdef int TreeliteLoadXGBoostModelFromMemoryBuffer(const void* buf,
                                                      size_t len,
                                                      ModelHandle* out) \
        except +
    cdef int TreeliteFreeModel(ModelHandle handle) except +
    cdef int TreeliteQueryNumTree(ModelHandle handle, size_t* out) except +
    cdef int TreeliteQueryNumFeature(ModelHandle handle, size_t* out) except +
    cdef int TreeliteLoadLightGBMModel(const char* filename,
                                       ModelHandle* out) except +
    cdef int TreeliteLoadProtobufModel(const char* filename,
                                       ModelHandle* out) except +
    cdef const char* TreeliteGetLastError()


cdef class TreeliteModel():
    """
    Wrapper for Treelite-loaded forest

    Note: This is only used for loading saved models into ForestInference,
    it does not actually perform inference. Users typically do
    not need to access TreeliteModel instances directly.

    Attributes
    ----------

    handle : ModelHandle
        Opaque pointer to Treelite model
    """
    cpdef ModelHandle handle
    cpdef bool owns_handle

    def __cinit__(self, owns_handle=True):
        """If owns_handle is True, free the handle's model in destructor.
        Set this to False if another owner will free the model."""
        self.handle = <ModelHandle>NULL
        self.owns_handle = owns_handle

    cdef set_handle(self, ModelHandle new_handle):
        self.handle = new_handle

    cdef ModelHandle get_handle(self):
        return self.handle

    def __dealloc__(self):
        if self.handle != NULL and self.owns_handle:
            TreeliteFreeModel(self.handle)

    @property
    def num_trees(self):
        assert self.handle != NULL
        cdef size_t out
        TreeliteQueryNumTree(self.handle, &out)
        return out

    @property
    def num_features(self):
        assert self.handle != NULL
        cdef size_t out
        TreeliteQueryNumFeature(self.handle, &out)
        return out

    @staticmethod
    def free_treelite_model(model_handle):
        cdef uintptr_t model_ptr = <uintptr_t>model_handle
        TreeliteFreeModel(<ModelHandle> model_ptr)

    @staticmethod
    def from_filename(filename, model_type="xgboost"):
        """
        Returns a TreeliteModel object loaded from `filename`

        Parameters
        ----------
        filename : string
            Path to treelite model file to load

        model_type : string
            Type of model: 'xgboost', or 'lightgbm'
        """
        filename_bytes = filename.encode("UTF-8")
        cdef ModelHandle handle
        if model_type == "xgboost":
            res = TreeliteLoadXGBoostModel(filename_bytes, &handle)
            if res < 0:
                err = TreeliteGetLastError()
                raise RuntimeError("Failed to load %s (%s)" % (filename, err))
        elif model_type == "lightgbm":
            res = TreeliteLoadLightGBMModel(filename_bytes, &handle)
            if res < 0:
                err = TreeliteGetLastError()
                raise RuntimeError("Failed to load %s (%s)" % (filename, err))
        else:
            raise ValueError("Unknown model type %s" % model_type)
        model = TreeliteModel()
        model.set_handle(handle)
        return model

    @staticmethod
    def from_treelite_model_handle(treelite_handle,
                                   take_handle_ownership=False):
        cdef ModelHandle handle = <ModelHandle> <size_t> treelite_handle
        model = TreeliteModel(owns_handle=take_handle_ownership)
        model.set_handle(handle)
        return model

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

    cdef struct forest:
        pass

    ctypedef forest* forest_t

    cdef struct treelite_params_t:
        algo_t algo
        bool output_class
        float threshold
        storage_type_t storage_type

    cdef void free(cumlHandle& handle,
                   forest_t)

    cdef void predict(cumlHandle& handle,
                      forest_t,
                      float*,
                      float*,
                      size_t,
                      bool)

    cdef forest_t from_treelite(cumlHandle& handle,
                                forest_t*,
                                ModelHandle,
                                treelite_params_t*)

cdef class ForestInference_impl():

    cpdef object handle
    cdef forest_t forest_data

    def __cinit__(self,
                  handle=None):
        self.handle = handle
        self.forest_data = NULL

    def get_algo(self, algo_str):
        algo_dict={'AUTO': algo_t.ALGO_AUTO,
                   'auto': algo_t.ALGO_AUTO,
                   'NAIVE': algo_t.NAIVE,
                   'naive': algo_t.NAIVE,
                   'BATCH_TREE_REORG': algo_t.BATCH_TREE_REORG,
                   'batch_tree_reorg': algo_t.BATCH_TREE_REORG,
                   'TREE_REORG': algo_t.TREE_REORG,
                   'tree_reorg': algo_t.TREE_REORG}
        if algo_str not in algo_dict.keys():
            raise Exception(' Wrong algorithm selected please refer'
                            ' to the documentation')
        return algo_dict[algo_str]

    def get_storage_type(self, storage_type_str):
        storage_type_dict={'auto': storage_type_t.AUTO,
                           'False': storage_type_t.DENSE,
                           'True': storage_type_t.SPARSE}

        if storage_type_str not in storage_type_dict.keys():
            raise ValueError(
                "The value entered for storage_type is not "
                "supported. Please refer to the documentation at"
                "(https://docs.rapids.ai/api/cuml/nightly/api.html#"
                "forest-inferencing) to see the accepted values.")
        return storage_type_dict[storage_type_str]

    def predict(self, X, output_type='numpy',
                output_dtype=None, predict_proba=False, preds=None):
        """
        Returns the results of forest inference on the examples in X

        Parameters
        ----------
        X : float32 array-like (device or host) shape = (n_samples, n_features)
            For optimal performance, pass a device array with C-style layout
        output_type : string (default = 'numpy')
            possible options are : {'input', 'cudf', 'cupy', 'numpy'}, optional
            Variable to control output type of the results and attributes of
            the estimators.
        preds : float32 device array, shape = n_samples
        predict_proba : bool, whether to output class probabilities(vs classes)
            Supported only for binary classification. output format
            matches sklearn

        Returns
        ----------
        Predicted results of type as defined by the output_type variable
        """
        cdef uintptr_t X_ptr
        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, order='C',
                                convert_to_dtype=np.float32,
                                check_dtype=np.float32)
        X_ptr = X_m.ptr

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        if preds is None:
            shape = (n_rows, )
            if predict_proba:
                shape += (2,)
            preds = CumlArray.empty(shape=shape, dtype=np.float32, order='C')
        elif (not isinstance(preds, cudf.Series) and
              not rmm.is_cuda_array(preds)):
            raise ValueError("Invalid type for output preds,"
                             " need GPU array")

        cdef uintptr_t preds_ptr
        preds_ptr = preds.ptr

        predict(handle_[0],
                self.forest_data,
                <float*> preds_ptr,
                <float*> X_ptr,
                <size_t> n_rows,
                <bool> predict_proba)
        self.handle.sync()

        # special case due to predict and predict_proba
        # both coming from the same CUDA/C++ function
        if predict_proba:
            output_dtype = None
        return preds.to_output(
            output_type=output_type,
            output_dtype=output_dtype
        )

    def load_from_treelite_model_handle(self,
                                        uintptr_t model_handle,
                                        bool output_class,
                                        str algo,
                                        float threshold,
                                        str storage_type):
        cdef treelite_params_t treelite_params
        treelite_params.output_class = output_class
        treelite_params.threshold = threshold
        treelite_params.algo = self.get_algo(algo)
        treelite_params.storage_type = self.get_storage_type(storage_type)

        self.forest_data = NULL
        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()
        cdef uintptr_t model_ptr = <uintptr_t>model_handle

        from_treelite(handle_[0],
                      &self.forest_data,
                      <ModelHandle> model_ptr,
                      &treelite_params)
        return self

    def load_from_treelite_model(self,
                                 TreeliteModel model,
                                 bool output_class,
                                 str algo,
                                 float threshold,
                                 str storage_type):
        return self.load_from_treelite_model_handle(<uintptr_t>model.handle,
                                                    output_class, algo,
                                                    threshold, storage_type)

    def load_using_treelite_handle(self,
                                   model_handle,
                                   bool output_class,
                                   str algo,
                                   float threshold,
                                   str storage_type):

        cdef treelite_params_t treelite_params

        treelite_params.output_class = output_class
        treelite_params.threshold = threshold
        treelite_params.algo = self.get_algo(algo)
        treelite_params.storage_type = self.get_storage_type(storage_type)
        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()
        cdef uintptr_t model_ptr = <uintptr_t>model_handle

        from_treelite(handle_[0],
                      &self.forest_data,
                      <ModelHandle> model_ptr,
                      &treelite_params)
        return self

    def __dealloc__(self):
        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()
        if self.forest_data !=NULL:
            free(handle_[0],
                 self.forest_data)


class ForestInference(Base):
    """ForestInference provides GPU-accelerated inference (prediction)
    for random forest and boosted decision tree models.

    This module does not support training models. Rather, users should
    train a model in another package and save it in a
    treelite-compatible format. (See https://github.com/dmlc/treelite)
    Currently, LightGBM, XGBoost and SKLearn GBDT and random forest models
    are supported.

    Users typically create a ForestInference object by loading a saved model
    file with ForestInference.load. It is also possible to create it from an
    SKLearn model using ForestInference.load_from_sklearn. The resulting object
    provides a `predict` method for carrying out inference.

    **Known limitations**:
     * A single row of data should fit into the shared memory of a thread
       block, which means that more than 12288 features are not supported.
     * From sklearn.ensemble, only
       {RandomForest,GradientBoosting}{Classifier,Regressor} models are
       supported; other sklearn.ensemble models are currently not supported.
     * Importing large SKLearn models can be slow, as it is done in Python.
     * LightGBM categorical features are not supported.
     * Inference uses a dense matrix format, which is efficient for many
       problems but can be suboptimal for sparse datasets.
     * Only binary classification and regression are supported.

    Parameters
    ----------
    handle : cuml.Handle
       If it is None, a new one is created just for this class.

    Examples
    --------

    In the example below, synthetic data is copied to the host before
    inference. ForestInference can also accept a numpy array directly at the
    cost of a slight performance overhead.

    .. code-block:: python

        # Assume that the file 'xgb.model' contains a classifier model that was
        # previously saved by XGBoost's save_model function.

        import sklearn, sklearn.datasets, numpy as np
        from numba import cuda
        from cuml import ForestInference

        model_path = 'xgb.model'
        X_test, y_test = sklearn.datasets.make_classification()
        X_gpu = cuda.to_device(np.ascontiguousarray(X_test.astype(np.float32)))
        fm = ForestInference.load(model_path, output_class=True)
        fil_preds_gpu = fm.predict(X_gpu)
        accuracy_score = sklearn.metrics.accuracy_score(y_test,
                       np.asarray(fil_preds_gpu))

    Notes
    ------
    For additional usage examples, see the sample notebook at
    https://github.com/rapidsai/cuml/blob/branch-0.14/notebooks/forest_inference_demo.ipynb

    """
    def __init__(self,
                 handle=None, output_type=None):
        super(ForestInference, self).__init__(handle,
                                              output_type=output_type)
        self._impl = ForestInference_impl(self.handle)

    def predict(self, X, preds=None):
        """
        Predicts the labels for X with the loaded forest model.
        By default, the result is the raw floating point output
        from the model, unless output_class was set to True
        during model loading.

        See the documentation of ForestInference.load for details.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
           Dense matrix (floats) of shape (n_samples, n_features).
           Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
           ndarray, cuda array interface compliant array like CuPy
           For optimal performance, pass a device array with C-style layout
        preds: gpuarray or cudf.Series, shape = (n_samples,)
           Optional 'out' location to store inference results

        Returns
        ----------
        GPU array of length n_samples with inference results
        (or 'preds' filled with inference results if preds was specified)
        """
        out_type = self._get_output_type(X)
        return self._impl.predict(X, out_type, predict_proba=False, preds=None)

    def predict_proba(self, X, preds=None):
        """
        Predicts the class probabilities for X with the loaded forest model.
        The result is the raw floating point output
        from the model.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
           Dense matrix (floats) of shape (n_samples, n_features).
           Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
           ndarray, cuda array interface compliant array like CuPy
           For optimal performance, pass a device array with C-style layout
        preds: gpuarray or cudf.Series, shape = (n_samples,2)
           binary probability output
           Optional 'out' location to store inference results

        Returns
        ----------
        GPU array of shape (n_samples,2) with inference results
        (or 'preds' filled with inference results if preds was specified)
        """
        out_type = self._get_output_type(X)

        return self._impl.predict(X, out_type, predict_proba=True, preds=None)

    def load_from_treelite_model(self, model, output_class=False,
                                 algo='auto',
                                 threshold=0.5,
                                 storage_type='auto'):
        """
        Creates a FIL model using the treelite model
        passed to the function.

        Parameters
        ----------
        model : the trained model information in the treelite format
           loaded from a saved model using the treelite API
           https://treelite.readthedocs.io/en/latest/treelite-api.html
        output_class: boolean (default=False)
           If True, return a 1 or 0 depending on whether the raw prediction
           exceeds the threshold. If False, just return the raw prediction.
        algo : string (default='auto')
            name of the algo from (from algo_t enum)
             'AUTO' or 'auto' - choose the algorithm automatically;
                   currently 'BATCH_TREE_REORG' is used for dense storage,
                   and 'NAIVE' for sparse storage
             'NAIVE' or 'naive' - simple inference using shared memory
             'TREE_REORG' or 'tree_reorg' - similar to naive but trees
                              rearranged to be more coalescing-friendly
             'BATCH_TREE_REORG' or 'batch_tree_reorg' - similar to TREE_REORG
                                    but predicting multiple rows
                                    per thread block
        threshold : float (default=0.5)
            Threshold is used to for classification. It is applied
            only if output_class == True, else it is ignored.
        storage_type : string or boolean (default='auto')
            In-memory storage format to be used for the FIL model.
             'auto' - choose the storage type automatically
                      (currently DENSE is always used)
             False - create a dense forest
             True - create a sparse forest;
                      requires algo='NAIVE' or algo='AUTO'

        Returns
        ----------
        fil_model :
            A Forest Inference model which can be used to perform
            inferencing on the random forest/ XGBoost model.
        """
        if isinstance(model, TreeliteModel):
            # TreeliteModel defined in this file
            return self._impl.load_from_treelite_model(
                model, output_class, algo, threshold, str(storage_type))
        else:
            # assume it is treelite.Model
            return self._impl.load_from_treelite_model_handle(
                model.handle.value, output_class, algo, threshold,
                str(storage_type))

    @staticmethod
    def load_from_sklearn(skl_model,
                          output_class=False,
                          threshold=0.50,
                          algo='auto',
                          storage_type='auto',
                          handle=None):
        """
        Creates a FIL model using the scikit-learn model passed to the
        function. This function requires Treelite 0.90 to be installed.

        Parameters
        ----------
        skl_model : The scikit-learn model from which to build the FIL version.
        output_class: boolean (default=False)
           If True, return a 1 or 0 depending on whether the raw prediction
           exceeds the threshold. If False, just return the raw prediction.
        algo : string (default='auto')
            name of the algo from (from algo_t enum)
             'AUTO' or 'auto' - choose the algorithm automatically;
                   currently 'BATCH_TREE_REORG' is used for dense storage,
                   and 'NAIVE' for sparse storage
             'NAIVE' or 'naive' - simple inference using shared memory
             'TREE_REORG' or 'tree_reorg' - similar to naive but trees
                              rearranged to be more coalescing-friendly
             'BATCH_TREE_REORG' or 'batch_tree_reorg' - similar to TREE_REORG
                                    but predicting multiple rows
                                    per thread block
        threshold : float (default=0.5)
            Threshold is used to for classification. It is applied
            only if output_class == True, else it is ignored.
        storage_type : string or boolean (default='auto')
            In-memory storage format to be used for the FIL model.
             'auto' - choose the storage type automatically
                      (currently DENSE is always used)
             False - create a dense forest
             True - create a sparse forest;
                      requires algo='NAIVE' or algo='AUTO'

        Returns
        ----------
        fil_model :
            A Forest Inference model created from the scikit-learn
            model passed.

        """
        cuml_fm = ForestInference(handle=handle)
        tl_model = tl_skl.import_model(skl_model)
        cuml_fm.load_from_treelite_model(
            tl_model, algo=algo, output_class=output_class,
            storage_type=str(storage_type), threshold=threshold)
        return cuml_fm

    @staticmethod
    def load(filename,
             output_class=False,
             threshold=0.50,
             algo='auto',
             storage_type='auto',
             model_type="xgboost",
             handle=None):
        """
        Returns a FIL instance containing the forest saved in 'filename'
        This uses Treelite to load the saved model.

        Parameters
        ----------
        filename : string
           Path to saved model file in a treelite-compatible format
           (See https://treelite.readthedocs.io/en/latest/treelite-api.html
            for more information)
        output_class : bool (default=False)
           If True, return a 1 or 0 depending on whether the raw prediction
           exceeds the threshold. If False, just return the raw prediction.
        threshold : float (default=0.5)
           Cutoff value above which a prediction is set to 1.0
           Only used if the model is classification and output_class is True
        algo : string (default='auto')
           Which inference algorithm to use.
           See documentation in FIL.load_from_treelite_model
        storage_type : string (default='auto')
            In-memory storage format to be used for the FIL model.
            See documentation in FIL.load_from_treelite_model
        model_type : string (default="xgboost")
            Format of the saved treelite model to be load.
            It can be 'xgboost', 'lightgbm'.

        Returns
        ----------
        fil_model :
            A Forest Inference model which can be used to perform
            inferencing on the model read from the file.
        """
        cuml_fm = ForestInference(handle=handle)
        tl_model = TreeliteModel.from_filename(filename, model_type=model_type)
        cuml_fm.load_from_treelite_model(tl_model,
                                         algo=algo,
                                         output_class=output_class,
                                         storage_type=str(storage_type),
                                         threshold=threshold)
        return cuml_fm

    def load_using_treelite_handle(self,
                                   model_handle,
                                   output_class=False,
                                   algo='auto',
                                   storage_type='auto',
                                   threshold=0.50):
        """
        Returns a FIL instance by converting a treelite model to
        FIL model by using the treelite ModelHandle passed.

        Parameters
        ----------
        model_handle : Modelhandle to the treelite forest model
            (See https://treelite.readthedocs.io/en/latest/treelite-api.html
            for more information)
        output_class : bool (default=False)
           If True, return a 1 or 0 depending on whether the raw prediction
           exceeds the threshold. If False, just return the raw prediction.
        threshold : float (default=0.5)
           Cutoff value above which a prediction is set to 1.0
           Only used if the model is classification and output_class is True
        algo : string (default='auto')
           Which inference algorithm to use.
           See documentation in FIL.load_from_treelite_model
        storage_type : string (default='auto')
            In-memory storage format to be used for the FIL model.
            See documentation in FIL.load_from_treelite_model

        Returns
        ----------
        fil_model :
            A Forest Inference model which can be used to perform
            inferencing on the random forest model.
        """
        return self._impl.load_using_treelite_handle(model_handle,
                                                     output_class,
                                                     algo, threshold,
                                                     str(storage_type))
