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

# distutils: language = c++

import copy
import ctypes
import math
import numpy as np
import warnings
import pandas as pd
from inspect import getdoc

import rmm

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

import cuml.internals
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from pylibraft.common.handle cimport handle_t
from cuml.common import input_to_cuml_array
from cuml.internals import logger
from cuml.internals.mixins import CMajorInputTagMixin
from cuml.common.doc_utils import _parameters_docstrings
from rmm._lib.memory_resource cimport DeviceMemoryResource
from rmm._lib.memory_resource cimport get_current_device_resource

import treelite
import treelite.sklearn as tl_skl

cimport cuml.common.cuda

cdef extern from "treelite/c_api.h":
    ctypedef void* ModelHandle
    cdef int TreeliteLoadXGBoostModel(const char* filename,
                                      ModelHandle* out) except +
    cdef int TreeliteLoadXGBoostJSON(const char* filename,
                                     ModelHandle* out) except +
    cdef int TreeliteFreeModel(ModelHandle handle) except +
    cdef int TreeliteQueryNumTree(ModelHandle handle, size_t* out) except +
    cdef int TreeliteQueryNumFeature(ModelHandle handle, size_t* out) except +
    cdef int TreeliteQueryNumClass(ModelHandle handle, size_t* out) except +
    cdef int TreeliteLoadLightGBMModel(const char* filename,
                                       ModelHandle* out) except +
    cdef int TreeliteSerializeModel(const char* filename,
                                    ModelHandle handle) except +
    cdef const char* TreeliteGetLastError()


cdef class TreeliteModel():
    """
    Wrapper for Treelite-loaded forest

    .. note:: This is only used for loading saved models into ForestInference,
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

    @property
    def handle(self):
        return <uintptr_t>(self.handle)

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
            Type of model: 'xgboost', 'xgboost_json', or 'lightgbm'
        """
        filename_bytes = filename.encode("UTF-8")
        cdef ModelHandle handle
        if model_type == "xgboost":
            res = TreeliteLoadXGBoostModel(filename_bytes, &handle)
            if res < 0:
                err = TreeliteGetLastError()
                raise RuntimeError("Failed to load %s (%s)" % (filename, err))
        elif model_type == "xgboost_json":
            res = TreeliteLoadXGBoostJSON(filename_bytes, &handle)
            if res < 0:
                err = TreeliteGetLastError()
                raise RuntimeError("Failed to load %s (%s)" % (filename, err))
        elif model_type == "lightgbm":
            logger.warn("Treelite currently does not support float64 model"
                        " parameters. Accuracy may degrade slightly relative"
                        " to native LightGBM invocation.")
            res = TreeliteLoadLightGBMModel(filename_bytes, &handle)
            if res < 0:
                err = TreeliteGetLastError()
                raise RuntimeError("Failed to load %s (%s)" % (filename, err))
        else:
            raise ValueError("Unknown model type %s" % model_type)
        model = TreeliteModel()
        model.set_handle(handle)
        return model

    def to_treelite_checkpoint(self, filename):
        """
        Serialize to a Treelite binary checkpoint

        Parameters
        ----------
        filename : string
            Path to Treelite binary checkpoint
        """
        assert self.handle != NULL
        filename_bytes = filename.encode("UTF-8")
        TreeliteSerializeModel(filename_bytes, self.handle)

    @staticmethod
    def from_treelite_model_handle(treelite_handle,
                                   take_handle_ownership=False):
        cdef ModelHandle handle = <ModelHandle> <size_t> treelite_handle
        model = TreeliteModel(owns_handle=take_handle_ownership)
        model.set_handle(handle)
        return model

cdef extern from "variant" namespace "std":
    cdef cppclass variant[T1, T2]:
        variant()
        variant(T1)
        size_t index()

    cdef T& get[T, T1, T2](variant[T1, T2]& v)

cdef extern from "cuml/fil/fil.h" namespace "ML::fil":
    cdef enum algo_t:
        ALGO_AUTO,
        NAIVE,
        TREE_REORG,
        BATCH_TREE_REORG

    cdef enum storage_type_t:
        AUTO,
        DENSE,
        SPARSE,
        SPARSE8

    cdef enum precision_t:
        PRECISION_NATIVE,
        PRECISION_FLOAT32,
        PRECISION_FLOAT64

    cdef cppclass forest[real_t]:
        pass

    ctypedef forest[float]* forest32_t
    ctypedef forest[double]* forest64_t
    ctypedef variant[forest32_t, forest64_t] forest_variant

    # TODO(canonizer): use something like
    # ctypedef forest[real_t]* forest_t[real_t]
    # once it is supported in Cython

    cdef struct treelite_params_t:
        algo_t algo
        bool output_class
        float threshold
        # changing the parameters below may speed up inference
        # tree storage format, tradeoffs in big O(), node size
        # not all formats fit all models
        storage_type_t storage_type
        # limit number of CUDA blocks launched per GPU SM (or unlimited if 0)
        int blocks_per_sm
        # multiple (neighboring) threads infer on the same tree within a block
        # this improves memory bandwith near tree root (but uses more shared
        # memory)
        int threads_per_tree
        # n_items is how many input samples (items) any thread processes.
        # if 0 is given, FIL chooses itself
        int n_items
        # this affects inference performance and will become configurable soon
        char** pforest_shape_str
        precision_t precision

    cdef void free[real_t](handle_t& handle,
                           forest[real_t]*)

    cdef void predict[real_t](handle_t& handle,
                              forest[real_t]*,
                              real_t*,
                              real_t*,
                              size_t,
                              bool) except +

    cdef void from_treelite(handle_t& handle,
                            forest_variant*,
                            ModelHandle,
                            treelite_params_t*) except +

cdef class ForestInference_impl():

    cdef object handle
    cdef forest_variant forest_data
    cdef size_t num_class
    cdef bool output_class
    cdef char* shape_str
    cdef DeviceMemoryResource mr

    cdef forest32_t get_forest32(self):
        return get[forest32_t, forest32_t, forest64_t](self.forest_data)

    cdef forest64_t get_forest64(self):
        return get[forest64_t, forest32_t, forest64_t](self.forest_data)

    def __cinit__(self,
                  handle=None):
        self.handle = handle
        self.forest_data = forest_variant(<forest32_t> NULL)
        self.shape_str = NULL
        self.mr = get_current_device_resource()

    def get_shape_str(self):
        if self.shape_str:
            return unicode(self.shape_str, 'utf-8')
        return None

    def get_dtype(self):
        dtype_array = [np.float32, np.float64]
        return dtype_array[self.forest_data.index()]

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

    def get_storage_type(self, storage_type):
        storage_type_str = str(storage_type)
        storage_type_dict={'auto': storage_type_t.AUTO,
                           'False': storage_type_t.DENSE,
                           'dense': storage_type_t.DENSE,
                           'True': storage_type_t.SPARSE,
                           'sparse': storage_type_t.SPARSE,
                           'sparse8': storage_type_t.SPARSE8}

        if storage_type_str not in storage_type_dict.keys():
            raise ValueError(
                "The value entered for storage_type is not "
                "supported. Please refer to the documentation at"
                "(https://docs.rapids.ai/api/cuml/nightly/api.html#"
                "forest-inferencing) to see the accepted values.")
        if storage_type_str == 'sparse8':
            logger.info('storage_type=="sparse8" is an experimental feature')
        return storage_type_dict[storage_type_str]

    def get_precision(self, precision):
        precision_dict = {'native': precision_t.PRECISION_NATIVE,
                          'float32': precision_t.PRECISION_FLOAT32,
                          'float64': precision_t.PRECISION_FLOAT64}
        if precision not in precision_dict:
            raise ValueError(
                "The value entered for precision is not "
                "supported. Please refer to the documentation at"
                "(https://docs.rapids.ai/api/cuml/nightly/api.html#"
                "forest-inferencing) to see the accepted values.")
        return precision_dict[precision]

    def predict(self, X,
                output_dtype=None,
                predict_proba=False,
                preds=None,
                safe_dtype_conversion=False):
        """
        Returns the results of forest inference on the examples in X

        Parameters
        ----------
        X : float32 array-like (device or host) shape = (n_samples, n_features)
            For optimal performance, pass a device array with C-style layout.
            For categorical features: category < 0.0 or category > 16'777'214
            is equivalent to out-of-dictionary category (not matching).
            -0.0 represents category 0.
            If float(int(category)) != category, we will discard the
            fractional part. E.g. 3.8 represents category 3 regardless of
            max_matching value. FIL will reject a model where an integer
            within [0, max_matching + 1] cannot be represented precisely
            as a float32.
            NANs work the same between numerical and categorical inputs:
            they are missing values and follow Treelite's DefaultLeft.
        preds : float32 device array, shape = n_samples
        predict_proba : bool, whether to output class probabilities(vs classes)
            Supported only for binary classification. output format
            matches sklearn

        Returns
        -------

        Predicted results of type as defined by the output_type variable

        """

        # Set the output_dtype. None is fine here
        cuml.internals.set_api_output_dtype(output_dtype)

        if (not self.output_class) and predict_proba:
            raise NotImplementedError("Predict_proba function is not available"
                                      " for Regression models. If you are "
                                      " using a Classification model, please "
                                      " set `output_class=True` while creating"
                                      " the FIL model.")
        fil_dtype = self.get_dtype()
        cdef uintptr_t X_ptr
        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, order='C',
                                convert_to_dtype=fil_dtype,
                                safe_dtype_conversion=safe_dtype_conversion,
                                check_dtype=fil_dtype)
        X_ptr = X_m.ptr

        cdef handle_t* handle_ =\
            <handle_t*><size_t>self.handle.getHandle()

        if preds is None:
            shape = (n_rows, )
            if predict_proba:
                if self.num_class <= 2:
                    shape += (2,)
                else:
                    shape += (self.num_class,)
            preds = CumlArray.empty(shape=shape, dtype=fil_dtype, order='C',
                                    index=X_m.index)
        else:
            if not hasattr(preds, "__cuda_array_interface__"):
                raise ValueError("Invalid type for output preds,"
                                 " need GPU array")
            preds.index = X_m.index

        cdef uintptr_t preds_ptr
        preds_ptr = preds.ptr

        if fil_dtype == np.float32:
            predict(handle_[0],
                    self.get_forest32(),
                    <float*> preds_ptr,
                    <float*> X_ptr,
                    <size_t> n_rows,
                    <bool> predict_proba)
        elif fil_dtype == np.float64:
            predict(handle_[0],
                    self.get_forest64(),
                    <double*> preds_ptr,
                    <double*> X_ptr,
                    <size_t> n_rows,
                    <bool> predict_proba)
        else:
            # should not reach here
            assert False, 'invalid fil_dtype, must be np.float32 or np.float64'

        self.handle.sync()

        # special case due to predict and predict_proba
        # both coming from the same CUDA/C++ function
        if predict_proba:
            cuml.internals.set_api_output_dtype(None)

        return preds

    def load_from_treelite_model_handle(self, **kwargs):
        self.forest_data = forest_variant(<forest32_t> NULL)
        return self.load_using_treelite_handle(**kwargs)

    def load_from_treelite_model(self, **kwargs):
        cdef TreeliteModel model = kwargs['model']
        return self.load_from_treelite_model_handle(
            model_handle=<uintptr_t>model.handle, **kwargs)

    def load_using_treelite_handle(self, **kwargs):
        cdef treelite_params_t treelite_params

        self.output_class = kwargs['output_class']
        treelite_params.output_class = self.output_class
        treelite_params.threshold = kwargs['threshold']
        treelite_params.algo = self.get_algo(kwargs['algo'])
        treelite_params.storage_type =\
            self.get_storage_type(kwargs['storage_type'])
        treelite_params.blocks_per_sm = kwargs['blocks_per_sm']
        treelite_params.n_items = kwargs['n_items']
        treelite_params.threads_per_tree = kwargs['threads_per_tree']
        if kwargs['compute_shape_str']:
            if self.shape_str:
                free(self.shape_str)
            treelite_params.pforest_shape_str = &self.shape_str
        else:
            treelite_params.pforest_shape_str = NULL
        treelite_params.precision = self.get_precision(kwargs['precision'])

        cdef handle_t* handle_ =\
            <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t model_ptr = <uintptr_t>kwargs['model_handle']

        from_treelite(handle_[0],
                      &self.forest_data,
                      <ModelHandle> model_ptr,
                      &treelite_params)
        TreeliteQueryNumClass(<ModelHandle> model_ptr,
                              &self.num_class)
        return self

    def __dealloc__(self):
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        fil_dtype = self.get_dtype()
        if fil_dtype == np.float32:
            if self.get_forest32() != NULL:
                free[float](handle_[0], self.get_forest32())
        elif fil_dtype == np.float64:
            if self.get_forest64() != NULL:
                free[double](handle_[0], self.get_forest64())
        else:
            # should not reach here
            assert False, 'invalid fil_dtype, must be np.float32 or np.float64'


class ForestInference(Base,
                      CMajorInputTagMixin):
    """
    ForestInference provides GPU-accelerated inference (prediction)
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
       block, otherwise (starting from 5000-12288 features) FIL might infer
       slower
     * From sklearn.ensemble, only
       `{RandomForest,GradientBoosting,ExtraTrees}{Classifier,Regressor}`
       models are supported. Other sklearn.ensemble models are currently not
       supported.
     * Importing large SKLearn models can be slow, as it is done in Python.
     * LightGBM categorical features are not supported.
     * Inference uses a dense matrix format, which is efficient for many
       problems but can be suboptimal for sparse datasets.
     * Only classification and regression are supported.
     * Many other random forest implementations including LightGBM, and SKLearn
       GBDTs make use of 64-bit floating point parameters, but the underlying
       library for ForestInference uses only 32-bit parameters. Because of the
       truncation that will occur when loading such models into
       ForestInference, you may observe a slight degradation in accuracy.

    Parameters
    ----------
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Examples
    --------

    In the example below, synthetic data is copied to the host before
    inference. ForestInference can also accept a numpy array directly at the
    cost of a slight performance overhead.

    .. code-block:: python

        >>> # Assume that the file 'xgb.model' contains a classifier model
        >>> # that was previously saved by XGBoost's save_model function.

        >>> import sklearn, sklearn.datasets
        >>> import numpy as np
        >>> from numba import cuda
        >>> from cuml import ForestInference

        >>> model_path = 'xgb.model'
        >>> X_test, y_test = sklearn.datasets.make_classification()
        >>> X_gpu = cuda.to_device(
        ...     np.ascontiguousarray(X_test.astype(np.float32)))
        >>> fm = ForestInference.load(
        ...     model_path, output_class=True) # doctest: +SKIP
        >>> fil_preds_gpu = fm.predict(X_gpu) # doctest: +SKIP
        >>> accuracy_score = sklearn.metrics.accuracy_score(y_test,
        ...     np.asarray(fil_preds_gpu)) # doctest: +SKIP

    Notes
    -----
    For additional usage examples, see the sample notebook at
    https://github.com/rapidsai/cuml/blob/main/notebooks/forest_inference_demo.ipynb

    """

    def common_load_params_docstring(func):
        func.__doc__ = getdoc(func).format("""
    output_class: boolean (default=False)
        For a Classification model `output_class` must be True.
        For a Regression model `output_class` must be False.
    algo : string (default='auto')
        Name of the algo from (from algo_t enum):

         - ``'AUTO'`` or ``'auto'``: Choose the algorithm automatically.
           Currently 'BATCH_TREE_REORG' is used for dense storage,
           and 'NAIVE' for sparse storage
         - ``'NAIVE'`` or ``'naive'``: Simple inference using shared memory
         - ``'TREE_REORG'`` or ``'tree_reorg'``: Similar to naive but trees
           rearranged to be more coalescing-friendly
         - ``'BATCH_TREE_REORG'`` or ``'batch_tree_reorg'``: Similar to
           TREE_REORG but predicting multiple rows per thread block

    threshold : float (default=0.5)
        Threshold is used to for classification. It is applied
        only if ``output_class == True``, else it is ignored.
    storage_type : string or boolean (default='auto')
        In-memory storage format to be used for the FIL model:

         - ``'auto'``: Choose the storage type automatically
           (currently DENSE is always used)
         - ``False``: Create a dense forest
         - ``True``: Create a sparse forest. Requires algo='NAIVE' or
           algo='AUTO'

    blocks_per_sm : integer (default=0)
        (experimental) Indicates how the number of thread blocks to lauch
        for the inference kernel is determined.

        - ``0`` (default): Launches the number of blocks proportional to
          the number of data rows
        - ``>= 1``: Attempts to lauch blocks_per_sm blocks per SM. This
          will fail if blocks_per_sm blocks result in more threads than the
          maximum supported number of threads per GPU. Even if successful,
          it is not guaranteed that blocks_per_sm blocks will run on an SM
          concurrently.
    compute_shape_str : boolean (default=False)
        if True or equivalent, creates a ForestInference.shape_str
        (writes a human-readable forest shape description as a
        multiline ascii string)
    precision : string (default='native')
        precision of weights and thresholds of the FIL model loaded from
        the treelite model.

        - ``'native'``: load in float64 if the treelite model contains float64
          weights or thresholds, otherwise load in float32
        - ``'float32'``: always load in float32, may lead to loss of precision
          if the treelite model contains float64 weights or thresholds
        - ``'float64'``: always load in float64
    """)
        return func

    def common_predict_params_docstring(func):
        func.__doc__ = getdoc(func).format(
          _parameters_docstrings['dense'].format(
            name='X', shape='(n_samples, n_features)') +
          '\n    For optimal performance, pass a float device array '
          'with C-style layout')
        return func

    def __init__(self, *,
                 handle=None,
                 output_type=None,
                 verbose=False):
        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)
        self._impl = ForestInference_impl(self.handle)

    @common_predict_params_docstring
    def predict(self, X, preds=None,
                safe_dtype_conversion=False) -> CumlArray:
        """
        Predicts the labels for X with the loaded forest model.
        By default, the result is the raw floating point output
        from the model, unless `output_class` was set to True
        during model loading.

        See the documentation of `ForestInference.load` for details.

        Parameters
        ----------
        preds : gpuarray or cudf.Series, shape = (n_samples,)
           Optional 'out' location to store inference results

        safe_dtype_conversion : bool (default = False)
            FIL converts data to np.float32 when needed. Set this parameter to
            True to enable checking for information loss during that
            conversion, but note that this check can have a significant
            performance penalty. Parameter will be dropped in a future
            version.

        Returns
        -------
        GPU array of length n_samples with inference results
        (or 'preds' filled with inference results if preds was specified)
        """
        return self._impl.predict(X, predict_proba=False, preds=None,
                                  safe_dtype_conversion=safe_dtype_conversion)

    @common_predict_params_docstring
    def predict_proba(self, X, preds=None,
                      safe_dtype_conversion=False) -> CumlArray:
        """
        Predicts the class probabilities for X with the loaded forest model.
        The result is the raw floating point output
        from the model.

        Parameters
        ----------
        preds : gpuarray or cudf.Series, shape = (n_samples,2)
           Binary probability output
           Optional 'out' location to store inference results

        safe_dtype_conversion : bool (default = False)
            FIL converts data to np.float32 when needed. Set this parameter to
            True to enable checking for information loss during that
            conversion, but note that this check can have a significant
            performance penalty. Parameter will be dropped in a future
            version.

        Returns
        -------
        GPU array of shape (n_samples,2) with inference results
        (or 'preds' filled with inference results if preds was specified)
        """
        return self._impl.predict(X, predict_proba=True, preds=None,
                                  safe_dtype_conversion=safe_dtype_conversion)

    @common_load_params_docstring
    def load_from_treelite_model(self, model, output_class=False,
                                 algo='auto',
                                 threshold=0.5,
                                 storage_type='auto',
                                 blocks_per_sm=0,
                                 threads_per_tree=1,
                                 n_items=0,
                                 compute_shape_str=False,
                                 precision='native'):
        """Creates a FIL model using the treelite model
        passed to the function.

        Parameters
        ----------
        model
            the trained model information in the treelite format
            loaded from a saved model using the treelite API
            https://treelite.readthedocs.io/en/latest/treelite-api.html
    {}
        Returns
        -------
        fil_model
            A Forest Inference model which can be used to perform
            inferencing on the random forest/ XGBoost model.

        """
        if isinstance(model, TreeliteModel):
            # TreeliteModel defined in this file
            self._impl.load_from_treelite_model(**locals())
        else:
            # assume it is treelite.Model
            self._impl.load_from_treelite_model_handle(
                model_handle=model.handle.value, **locals())
        self.shape_str = self._impl.get_shape_str()
        return self

    @staticmethod
    def load_from_sklearn(skl_model,
                          output_class=False,
                          threshold=0.50,
                          algo='auto',
                          storage_type='auto',
                          blocks_per_sm=0,
                          threads_per_tree=1,
                          n_items=0,
                          compute_shape_str=False,
                          precision='native',
                          handle=None):
        """
        Creates a FIL model using the scikit-learn model passed to the
        function. This function requires Treelite 1.0.0+ to be installed.

        Parameters
        ----------
        skl_model
            The scikit-learn model from which to build the FIL version.
        output_class: boolean (default=False)
            For a Classification model `output_class` must be True.
            For a Regression model `output_class` must be False.
        algo : string (default='auto')
            Name of the algo from (from algo_t enum):

             - ``'AUTO'`` or ``'auto'``: Choose the algorithm automatically.
               Currently 'BATCH_TREE_REORG' is used for dense storage,
               and 'NAIVE' for sparse storage
             - ``'NAIVE'`` or ``'naive'``: Simple inference using shared memory
             - ``'TREE_REORG'`` or ``'tree_reorg'``: Similar to naive but trees
               rearranged to be more coalescing-friendly
             - ``'BATCH_TREE_REORG'`` or ``'batch_tree_reorg'``: Similar to
               TREE_REORG but predicting multiple rows per thread block

        threshold : float (default=0.5)
            Threshold is used to for classification. It is applied
            only if ``output_class == True``, else it is ignored.
        storage_type : string or boolean (default='auto')
            In-memory storage format to be used for the FIL model:

             - ``'auto'``: Choose the storage type automatically
               (currently DENSE is always used)
             - ``False``: Create a dense forest
             - ``True``: Create a sparse forest. Requires algo='NAIVE' or
               algo='AUTO'

        blocks_per_sm : integer (default=0)
            (experimental) Indicates how the number of thread blocks to lauch
            for the inference kernel is determined.

            - ``0`` (default): Launches the number of blocks proportional to
              the number of data rows
            - ``>= 1``: Attempts to lauch blocks_per_sm blocks per SM. This
              will fail if blocks_per_sm blocks result in more threads than the
              maximum supported number of threads per GPU. Even if successful,
              it is not guaranteed that blocks_per_sm blocks will run on an SM
              concurrently.
        compute_shape_str : boolean (default=False)
            if True or equivalent, creates a ForestInference.shape_str
            (writes a human-readable forest shape description as a
            multiline ascii string)
        precision : string (default='native')
            precision of weights and thresholds of the FIL model loaded from
            the treelite model.

            - ``'native'``: load in float64 if the treelite model contains
              float64 weights or thresholds, otherwise load in float32
            - ``'float32'``: always load in float32, may lead to loss of
              precision if the treelite model contains float64 weights or
              thresholds
            - ``'float64'``: always load in float64

        Returns
        -------
        fil_model
            A Forest Inference model created from the scikit-learn
            model passed.

        """
        kwargs = locals()
        [kwargs.pop(key) for key in ['skl_model', 'handle']]
        cuml_fm = ForestInference(handle=handle)
        logger.warn("Treelite currently does not support float64 model"
                    " parameters. Accuracy may degrade slightly relative to"
                    " native sklearn invocation.")
        tl_model = tl_skl.import_model(skl_model)
        cuml_fm.load_from_treelite_model(model=tl_model, **kwargs)
        return cuml_fm

    @staticmethod
    def load(filename,
             output_class=False,
             threshold=0.50,
             algo='auto',
             storage_type='auto',
             blocks_per_sm=0,
             threads_per_tree=1,
             n_items=0,
             compute_shape_str=False,
             precision='native',
             model_type="xgboost",
             handle=None):
        """
        Returns a FIL instance containing the forest saved in `filename`
        This uses Treelite to load the saved model.

        Parameters
        ----------
        filename : string
            Path to saved model file in a treelite-compatible format
            (See https://treelite.readthedocs.io/en/latest/treelite-api.html
            for more information)
        output_class: boolean (default=False)
            For a Classification model `output_class` must be True.
            For a Regression model `output_class` must be False.
        algo : string (default='auto')
            Name of the algo from (from algo_t enum):

             - ``'AUTO'`` or ``'auto'``: Choose the algorithm automatically.
               Currently 'BATCH_TREE_REORG' is used for dense storage,
               and 'NAIVE' for sparse storage
             - ``'NAIVE'`` or ``'naive'``: Simple inference using shared memory
             - ``'TREE_REORG'`` or ``'tree_reorg'``: Similar to naive but trees
               rearranged to be more coalescing-friendly
             - ``'BATCH_TREE_REORG'`` or ``'batch_tree_reorg'``: Similar to
               TREE_REORG but predicting multiple rows per thread block

        threshold : float (default=0.5)
            Threshold is used to for classification. It is applied
            only if ``output_class == True``, else it is ignored.
        storage_type : string or boolean (default='auto')
            In-memory storage format to be used for the FIL model:

             - ``'auto'``: Choose the storage type automatically
               (currently DENSE is always used)
             - ``False``: Create a dense forest
             - ``True``: Create a sparse forest. Requires algo='NAIVE' or
               algo='AUTO'

        blocks_per_sm : integer (default=0)
            (experimental) Indicates how the number of thread blocks to lauch
            for the inference kernel is determined.

            - ``0`` (default): Launches the number of blocks proportional to
              the number of data rows
            - ``>= 1``: Attempts to lauch blocks_per_sm blocks per SM. This
              will fail if blocks_per_sm blocks result in more threads than the
              maximum supported number of threads per GPU. Even if successful,
              it is not guaranteed that blocks_per_sm blocks will run on an SM
              concurrently.
        compute_shape_str : boolean (default=False)
            if True or equivalent, creates a ForestInference.shape_str
            (writes a human-readable forest shape description as a
            multiline ascii string)
        precision : string (default='native')
            precision of weights and thresholds of the FIL model loaded from
            the treelite model.

            - ``'native'``: load in float64 if the treelite model contains
              float64 weights or thresholds, otherwise load in float32
            - ``'float32'``: always load in float32, may lead to loss of
              precision if the treelite model contains float64 weights or
              thresholds
            - ``'float64'``: always load in float64

        model_type : string (default="xgboost")
            Format of the saved treelite model to be load.
            It can be 'xgboost', 'xgboost_json', 'lightgbm'.

        Returns
        -------
        fil_model
            A Forest Inference model which can be used to perform
            inferencing on the model read from the file.

        """
        kwargs = locals()
        [kwargs.pop(key) for key in ['filename', 'handle', 'model_type']]
        cuml_fm = ForestInference(handle=handle)
        tl_model = TreeliteModel.from_filename(filename, model_type=model_type)
        cuml_fm.load_from_treelite_model(model=tl_model, **kwargs)
        return cuml_fm

    @common_load_params_docstring
    def load_using_treelite_handle(self,
                                   model_handle,
                                   output_class=False,
                                   algo='auto',
                                   storage_type='auto',
                                   threshold=0.50,
                                   blocks_per_sm=0,
                                   threads_per_tree=1,
                                   n_items=0,
                                   compute_shape_str=False,
                                   precision='native'
                                   ):
        """
        Returns a FIL instance by converting a treelite model to
        FIL model by using the treelite ModelHandle passed.

        Parameters
        ----------
        model_handle : Modelhandle to the treelite forest model
            (See https://treelite.readthedocs.io/en/latest/treelite-api.html
            for more information)
    {}

        Returns
        -------
        fil_model
            A Forest Inference model which can be used to perform
            inferencing on the random forest model.
        """
        self._impl.load_using_treelite_handle(**locals())
        self.shape_str = self._impl.get_shape_str()
        # DO NOT RETURN self._impl here!!
        return self
