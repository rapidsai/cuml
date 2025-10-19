#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
import cudf
import cupy as cp
import numpy as np
import pandas as pd

import cuml.internals.logger as logger
from cuml.explainer.common import (
    get_handle_from_cuml_model_func,
    get_link_fn_from_str_or_fn,
    get_tag_from_model_func,
    model_func_call,
    output_list_shap_values,
)
from cuml.internals.input_utils import input_to_cupy_array, input_to_host_array

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/explainer/permutation_shap.hpp" namespace "ML" nogil:

    void shap_main_effect_dataset "ML::Explainer::shap_main_effect_dataset"(
        const handle_t& handle,
        float* dataset,
        const float* background,
        int nrows,
        int ncols,
        const float* row,
        int* idx,
        bool rowMajor) except +

    void shap_main_effect_dataset "ML::Explainer::shap_main_effect_dataset"(
            const handle_t& handle,
            double* dataset,
            const double* background,
            int nrows,
            int ncols,
            const double* row,
            int* idx,
            bool rowMajor) except +


class SHAPBase():
    """
    Base class for SHAP based explainers.

    Parameters
    ----------
    model : function
        Function that takes a matrix of samples (n_samples, n_features) and
        computes the output for those samples with shape (n_samples). Function
        must use either CuPy or NumPy arrays as input/output.
    data : Dense matrix containing floats or doubles.
        Background dataset. Dense arrays are supported.
    order : 'F', 'C' or None (default = None)
        Set to override detection of row ('C') or column ('F') major order,
        if None it will be attempted to be inferred from model.
    order_default : 'F' or 'C' (default = 'C')
        Used when `order` is None. If the order cannot be inferred from the
        model, then order is set to `order_default`.
    link : function or str (default = 'identity')
        The link function used to map between the output units of the
        model and the SHAP value units.
    random_state: int, RandomState instance or None (default = None)
        Seed for the random number generator for dataset creation.
    is_gpu_model : bool or None (default = None)
        If None Explainer will try to infer whether `model` can take GPU data
        (as CuPy arrays), otherwise it will use NumPy arrays to call `model`.
        Set to True to force the explainer to use GPU data,  set to False to
        force the Explainer to use NumPy data.
    handle : pylibraft.common.handle
        Specifies the handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    dtype : np.float32 or np.float64 (default = np.float32)
        Parameter to specify the precision of data to generate to call the
        model.
    output_type : 'cupy' or 'numpy' (default = None)
        Parameter to specify the type of data to output.
        If not specified, the explainer will try to see if model is gpu based,
        if so it will be set to `cupy`, otherwise it will be set to `numpy`.
        For compatibility with SHAP's graphing libraries, specify `numpy`.

    """

    def __init__(self,
                 *,
                 model,
                 background,
                 order=None,
                 order_default='C',
                 link='identity',
                 verbose=False,
                 random_state=None,
                 is_gpu_model=None,
                 handle=None,
                 dtype=np.float32,
                 output_type=None):

        if verbose is True:
            self.verbose = logger.level_enum.debug
        elif verbose is False:
            self.verbose = logger.level_enum.error
        else:
            self.verbose = verbose

        if self.verbose >= logger.level_enum.debug:
            self.time_performance = True
        else:
            self.time_performance = False

        if handle is None:
            self.handle = get_handle_from_cuml_model_func(model,
                                                          create_new=True)
        else:
            self.handle = handle

        if order is None:
            self.order = get_tag_from_model_func(func=model,
                                                 tag='preferred_input_order',
                                                 default=order_default)
        else:
            self.order = order

        self.link = link
        self.link_fn = get_link_fn_from_str_or_fn(link)
        self.model = model
        if is_gpu_model is None:
            # todo (dgd): when sparse support is added, use this tag to see if
            # model can accept sparse data
            self.is_gpu_model = \
                get_tag_from_model_func(func=model,
                                        tag='X_types_gpu',
                                        default=None) is not None
        else:
            self.is_gpu_model = is_gpu_model

        # we are defaulting to numpy for now for compatibility
        if output_type is None:
            self.output_type = 'numpy'
        else:
            self.output_type = output_type

        if (dtype := np.dtype(dtype)) not in [np.float32, np.float64]:
            raise ValueError("dtype must be either np.float32 or np.float64.")
        self.dtype = dtype

        self.background, self.nrows, self.ncols, _ = \
            input_to_cupy_array(background, order=self.order,
                                convert_to_dtype=self.dtype)

        self.random_state = random_state

        if isinstance(background, pd.DataFrame) or isinstance(background, cudf.DataFrame):
            self.feature_names = background.columns.to_list()
        else:
            self.feature_names = [None for _ in range(len(background))]

        # evaluate the model in background to get the expected_value
        self._expected_value = self.link_fn(
            cp.mean(
                model_func_call(X=self.background,
                                model_func=self.model,
                                gpu_model=self.is_gpu_model),
                axis=0
            )
        )

        # public attribute saved as NumPy for compatibility with the legacy
        # SHAP potting functions
        self.expected_value = cp.asnumpy(self._expected_value)

        # Calculate the dimension of the model. For example, `predict_proba`
        # functions typically return n values for n classes as opposed to
        # 1 valued for a typical `predict`
        if len(self._expected_value.shape) == 0:
            self.model_dimensions = 1
            self.expected_value = float(self.expected_value)
        else:
            self.model_dimensions = self._expected_value.shape[0]

        self._reset_timers()

    def _explain(self,
                 X,
                 testing=False,
                 synth_data_shape=None,
                 free_synth_data=True,
                 return_as_list=True,
                 **kwargs):
        """
        Function that calls inheriting explainers _explain_single_observation
        in each row of X.

        Parameters
        ----------
        X : Dense matrix containing floats or doubles.
            Acceptable formats: CUDA array interface compliant objects like
            CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas
            DataFrame/Series.
        testing : bool (default: False)
            Flag to control random behaviors used by some explainers for
            running pytests. Might be removed in a future version, meant only
            for testing code.
        synth_data_shape : tuple (default: None)
            Shape of temporary data needed by inheriting explainer.
        free_synth_data : bool (default: True)
            Whether to free temporary memory after the call. Useful in case a
            workflow requires multiple calls to shap_values with small data
            as opposed to fewer calls with bigger data.
        **kwargs: dictionary
            Specific parameters that the _explain_single_observation of
            inheriting classes need.

        Returns
        -------
        shap_values : array
            Aray with the shap values, using cuml.internals output type logic.

        """
        self._reset_timers()

        X = input_to_cupy_array(X,
                                order=self.order,
                                convert_to_dtype=self.dtype)[0]

        if X.ndim == 1:
            X = X.reshape((1, self.ncols))

        # shap_values is a list so we can return a list in the case that
        # model is a multidimensional-output function
        shap_values = []

        for i in range(self.model_dimensions):
            shap_values.append(cp.zeros(X.shape, dtype=self.dtype))

        # Allocate synthetic dataset array once for multiple explanations
        if getattr(self, "_synth_data", None) is None and synth_data_shape \
                is not None:
            self._synth_data = cp.zeros(
                shape=synth_data_shape,
                dtype=self.dtype,
                order=self.order
            )

        # Explain each observation
        for idx, x in enumerate(X):
            # use mutability of lists and cupy arrays to get all shap values
            self._explain_single_observation(
                shap_values=shap_values,
                row=x.reshape(1, self.ncols),
                idx=idx,
                **kwargs
            )

        if free_synth_data and getattr(self, "synth_data", None) is not None:
            del self._synth_data

        if return_as_list:
            shap_values = output_list_shap_values(
                X=shap_values,
                dimensions=self.model_dimensions,
                output_type=self.output_type
            )

        return shap_values

    def __call__(self, X, main_effects=False, **kwargs):
        try:
            import shap
            from packaging.version import Version
            from shap import Explanation

            shap_atleast_037 = Version(shap.__version__) >= Version("0.37")
        except ImportError:
            shap_atleast_037 = False

        if not shap_atleast_037:
            raise ImportError(
                "SHAP >= 0.37 was not found, please install it or use the "
                "explainer.shap_values function instead."
            )

        logger.warn(
            "Support for the new API is in experimental state, tested with SHAP 0.37, but "
            "changes in further versions could affect its functioning. The functions "
            "explainer.shap_values and explainer.main_effects are the stable calls currently."
        )

        shap_values = self.shap_values(X,
                                       as_list=False,
                                       **kwargs)

        # reshaping of arrays to match SHAP's behavior for building
        # Explanation objects
        if self.model_dimensions > 1:
            shap_values == cp.asnumpy(cp.array(shap_values)).reshape(
                len(X), X.shape[1], self.model_dimensions
            )
            base_values = np.tile(self.expected_value, (len(X), 1))
        else:
            shap_values = cp.asnumpy(shap_values[0])
            base_values = np.tile(self.expected_value, len(X))

        if main_effects:
            main_effect_values = self.main_effects(X)
        else:
            main_effect_values = None

        out = Explanation(
            values=shap_values,
            base_values=base_values,
            data=input_to_host_array(X).array,
            feature_names=self.feature_names,
            main_effects=main_effect_values
        )

        return out

    def main_effects(self,
                     X):
        """
        A utility method to compute the main effects of a model.
        """

        main_effects = []
        for idx, x in enumerate(X):
            main_effects.append(self._calculate_main_effects(x))

        return main_effects

    def _calculate_main_effects(self,
                                main_effect_values,
                                row,
                                inds=None):
        if inds is None:
            inds = cp.arange(len(self.masker), dtype=np.float32)

        masked_inputs = cp.empty(
            shape=((self.nrows * self.ncols + self.nrows), self.ncols),
            dtype=self.dtype,
            order=self.masker.order
        )

        cdef handle_t* handle_ = \
            <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t row_ptr, bg_ptr, idx_ptr, masked_ptr

        masked_ptr = masked_inputs.__cuda_array_interface__['data'][0]

        bg_ptr = self.masker.ptr
        row_ptr = row.ptr
        idx_ptr = inds.__cuda_array_interface__['data'][0]
        row_major = self.masker.order == "C"

        if self.masker.order.dtype == cp.float32:
            shap_main_effect_dataset(handle_[0],
                                     <float*> masked_ptr,
                                     <float*> bg_ptr,
                                     <int> self.nrows,
                                     <int> self.ncols,
                                     <float*> row_ptr,
                                     <int*> idx_ptr,
                                     <bool> row_major)
        else:
            shap_main_effect_dataset(handle_[0],
                                     <double*> masked_ptr,
                                     <double*> bg_ptr,
                                     <int> self.nrows,
                                     <int> self.ncols,
                                     <double*> row_ptr,
                                     <int*> idx_ptr,
                                     <bool> row_major)

        self.handle.sync()

        main_effects = model_func_call(masked_inputs) - self._expected_value
        return main_effects

    def _reset_timers(self):
        self.total_time = 0
        self.model_call_time = 0
