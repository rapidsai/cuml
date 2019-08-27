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

import copy
import cudf
import ctypes
import math
import numpy as np
import warnings

from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_dev_array_ptr, input_to_dev_array, zeros
cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "treelite/c_api.h":
    ctypedef void* ModelHandle
    cdef int TreeliteLoadXGBoostModel(const char* filename,
                                      ModelHandle* out)
    cdef int TreeliteLoadXGBoostModelFromMemoryBuffer(const void* buf,
                                                      size_t len,
                                                      ModelHandle* out)
    cdef int TreeliteFreeModel(ModelHandle handle)
    cdef int TreeliteQueryNumTree(ModelHandle handle, size_t* out)
    cdef int TreeliteQueryNumFeature(ModelHandle handle, size_t* out)
    cdef int TreeliteLoadLightGBMModel(const char* filename, ModelHandle* out)
    cdef int TreeliteLoadProtobufModel(const char* filename, ModelHandle* out)


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

    def __cinit__(self):
        self.handle = <ModelHandle>NULL

    cdef set_handle(self, ModelHandle new_handle):
        self.handle = new_handle

    cdef ModelHandle get_handle(self):
        return self.handle

    def __cdel__(self):
        if self.handle != NULL:
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
    def from_filename(filename, model_type="xgboost"):
        """
        Returns a TreeliteModel object loaded from `filename`

        Parameters
        ----------
        filename : string
            Path to treelite model file to load

        model_type : string
            Type of model: 'xgboost', 'protobuf', or 'lightgbm'
        """
        filename_bytes = filename.encode("UTF-8")
        cdef ModelHandle handle
        if model_type == "xgboost":
            res = TreeliteLoadXGBoostModel(filename_bytes, &handle)
            if res < 0:
                raise RuntimeError("Failed to load %s" % filename)
        elif model_type == "protobuf":
            # XXX Not tested
            res = TreeliteLoadProtobufModel(filename_bytes, &handle)
            if res < 0:
                raise RuntimeError("Failed to load %s" % filename)
        elif model_type == "lightgbm":
            res = TreeliteLoadLightGBMModel(filename_bytes, &handle)
            if res < 0:
                raise RuntimeError("Failed to load %s" % filename)
        else:
            raise ValueError("Unknown model type %s" % model_type)
        model = TreeliteModel()
        model.set_handle(handle)
        return model


cdef extern from "fil/fil.h" namespace "ML::fil":
    cdef enum algo_t:
        NAIVE,
        TREE_REORG,
        BATCH_TREE_REORG

    cdef struct forest:
        pass

    ctypedef forest* forest_t

    cdef struct treelite_params_t:
        algo_t algo
        bool output_class
        float threshold

    cdef void free(cumlHandle& handle,
                   forest_t)

    cdef void predict(cumlHandle& handle,
                      forest_t,
                      float*,
                      float*,
                      size_t)

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

    def get_algo(self, algo_str):
        algo_dict={'NAIVE': algo_t.NAIVE,
                   'BATCH_TREE_REORG': algo_t.BATCH_TREE_REORG,
                   'TREE_REORG': algo_t.TREE_REORG}
        if algo_str not in algo_dict.keys():
            raise Exception(' Wrong algorithm selected please refer'
                            ' to the documentation')
        return algo_dict[algo_str]

    def predict(self, X, preds=None):
        """
        Returns the results of forest inference on the exampes in X

        Parameters
        ----------
        X : float32 array-like (device or host) shape = (n_samples, n_features)
            For optimal performance, pass a device array with C-style layout

        preds : float32 device array, shape = n_samples
        """
        cdef uintptr_t X_ptr
        X_m, X_ptr, n_rows, _, X_dtype = \
            input_to_dev_array(X, order='C', check_dtype=np.float32)

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        if preds is None:
            preds = cuda.device_array(n_rows, dtype=np.float32)
        elif (not isinstance(preds, cudf.Series) and
              not cuda.is_cuda_array(preds)):
            raise ValueError("Invalid type for output preds,"
                             " need GPU array")

        cdef uintptr_t preds_ptr
        preds_m, preds_ptr, _, _, _ = input_to_dev_array(
            preds,
            check_dtype=np.float32)

        predict(handle_[0],
                self.forest_data,
                <float*> preds_ptr,
                <float*> X_ptr,
                <size_t> n_rows)
        self.handle.sync()
        # synchronous w/o a stream
        return preds

    def load_from_treelite_model(self,
                                 TreeliteModel model,
                                 bool output_class,
                                 str algo,
                                 float threshold):
        cdef treelite_params_t treelite_params
        treelite_params.output_class = output_class
        treelite_params.threshold = threshold
        treelite_params.algo = self.get_algo(algo)
        self.forest_data = NULL
        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()
        cdef uintptr_t model_ptr = <uintptr_t>model.handle
        from_treelite(handle_[0],
                      &self.forest_data,
                      <ModelHandle> model_ptr,
                      &treelite_params)
        return self

    def load_from_randomforest(self,
                               model_handle,
                               bool output_class,
                               str algo,
                               float threshold):
        cdef treelite_params_t treelite_params
        treelite_params.output_class = output_class
        treelite_params.threshold = threshold
        treelite_params.algo = self.get_algo(algo)
        self.forest_data = NULL
        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()
        cdef uintptr_t model_ptr = <uintptr_t> model_handle
        from_treelite(handle_[0],
                      &self.forest_data,
                      <ModelHandle> model_ptr,
                      &treelite_params)
        return self

    def __cdel__(self):
        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()
        free(handle_[0],
             self.forest_data)
        return self


class ForestInference(Base):
    """
    ForestInference provides GPU-accelerated inference (prediction)
    for random forest and boosted decision tree models.

    This module does not support training models. Rather, users should
    train a model in another package and save it in a
    treelite-compatible format. (See https://github.com/dmlc/treelite)
    Currently, LightGBM and XGBoost GBDT and random forest models are
    supported.

    Users typically create a ForestInference object by loading a
    saved model file with ForestInference.load. The resulting object
    provides a `predict` method for carrying out inference.

    **Known limitations**:
     * Trees are represented as complete binary trees, so a tree of depth k
       will be stored in (2**k) - 1 nodes. This will be less space-efficient
       for sparse trees.
     * While treelite supports additional formats, only XGBoost and LightGBM
       are tested in FIL currently.
     * LightGBM categorical features are not supported
     * Inference uses a dense matrix format, which is efficient for many
       problems but will be suboptimal for sparse datasets.
     * Only binary classification and regression are supported.

    Parameters
    ----------
    handle : cuml.Handle
       If it is None, a new one is created just for this class.

    Examples
    --------
    For additional usage examples, see the sample notebook at
    https://github.com/rapidsai/notebooks/blob/branch-0.9/cuml/fil_demo.ipynb

    In the example below, synthetic data is copied to the host before
    infererence. ForestInference can also accept a numpy array directly at the
    cost of a slight performance overhead.

    >>> # Assume that the file 'xgb.model' contains a classifier model that was
    >>> # previously saved by XGBoost's save_model function.
    >>>
    >>> import sklearn
    >>> from numba import cuda
    >>> X_test, y_test = sklearn.datasets.make_classification()
    >>> X_gpu = cuda.to_device(np.ascontiguousarray(X_test.astype(np.float32)))
    >>> fm = ForestInference.load(model_path, output_class=True)
    >>> fil_preds_gpu = fm.predict(X_gpu)
    >>> accuracy_score = sklearn.metrics.accuracy_score(y_test,
    >>>						     np.asarray(fil_preds_gpu))

    """
    def __init__(self,
                 handle=None):
        super(ForestInference, self).__init__(handle)
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
        return self._impl.predict(X, preds)

    def load_from_treelite_model(self, model, output_class,
                                 algo='TREE_REORG',
                                 threshold=0.5):
        """
        Creates a FIL model using the treelite model
        passed to the function.

        Parameters
        ----------
        model : the trained model information in the treelite format
           loaded from a saved model using the treelite API
           https://treelite.readthedocs.io/en/latest/treelite-api.html
        output_class: boolean
           If true, return a 1 or 0 depending on whether the raw prediction
           exceeds the threshold. If False, just return the raw prediction.
        algo : string name of the algo from (from algo_t enum)
             'NAIVE' - simple inference using shared memory
             'TREE_REORG' - similar to naive but trees rearranged to be more
                              coalescing-friendly
             'BATCH_TREE_REORG' - similar to TREE_REORG but predicting
                                    multiple rows per thread block
        threshold : threshold is used to for classification
           applied if output_class == True, else it is ignored
        """
        return self._impl.load_from_treelite_model(model, output_class,
                                                   algo, threshold)

    @staticmethod
    def load(filename,
             output_class=False,
             threshold=0.50,
             algo='TREE_REORG',
             model_type="xgboost",
             handle=None):
        """
        Returns a FIL instance containing the forest saved in 'filename'
        This uses Treelite to load the saved model.

        Parameters
        ----------
        filename : str
           Path to saved model file in a treelite-compatible format
           (See https://treelite.readthedocs.io/en/latest/treelite-api.html
        output_class : bool
           If true, return a 1 or 0 depending on whether the raw prediction
           exceeds the threshold. If False, just return the raw prediction.
        threshold : float
           Cutoff value above which a prediction is set to 1.0
           Only used if the model is classification and output_class is True
        algo : string
           Which inference algorithm to use.
           See documentation in FIL.load_from_treelite_model
        model_type : str
            Format of saved treelite model to load.
            Can be 'xgboost', 'lightgbm', or 'protobuf'
        """
        cuml_fm = ForestInference(handle=handle)
        tl_model = TreeliteModel.from_filename(filename, model_type=model_type)
        cuml_fm.load_from_treelite_model(tl_model,
                                         algo=algo,
                                         output_class=output_class,
                                         threshold=threshold)
        return cuml_fm

    def load_from_randomforest(self,
                               model_handle,
                               output_class=False,
                               algo='TREE_REORG',
                               threshold=0.50):
        
        return self._impl.load_from_randomforest(model_handle, output_class,
                                          algo, threshold)
