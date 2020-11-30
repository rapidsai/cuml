#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
import pandas

import cuml.common.logger as logger
from cuml.experimental.explainer.common import get_dtype_from_model_func
from cuml.experimental.explainer.common import get_handle_from_cuml_model_func
from cuml.experimental.explainer.common import get_link_fn_from_str_or_fn
from cuml.experimental.explainer.common import get_tag_from_model_func
from cuml.experimental.explainer.common import model_func_call
from cuml.common.input_utils import input_to_cupy_array


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
    gpu_model : bool or None (default = None)
        If None Explainer will try to infer whether `model` can take GPU data
        (as CuPy arrays), otherwise it will use NumPy arrays to call `model`.
        Set to True to force the explainer to use GPU data,  set to False to
        force the Explainer to use NumPy data.
    handle : cuml.raft.common.handle
        Specifies the handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    dtype : np.float32 or np.float64 (default = None)
        Parameter to specify the precision of data to generate to call the
        model. If not specified, the explainer will try to get the dtype
        of the model, if it cannot be queried, then it will defaul to
        np.float32.
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
                 gpu_model=None,
                 handle=None,
                 dtype=None,
                 output_type=None):

        if verbose is True:
            self.verbose = logger.level_debug
        elif verbose is False:
            self.verbose = logger.level_error
        else:
            self.verbose = verbose

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
        if gpu_model is None:
            # todo: when sparse support is added, use this tag to see if
            # model can accept sparse data
            self.gpu_model = \
                get_tag_from_model_func(func=model,
                                        tag='X_types_gpu',
                                        default=None) is not None
        else:
            self.gpu_model = gpu_model

        # we are defaulting to numpy for now for compatibility
        if output_type is None:
            # self.output_type = 'cupy' if self.gpu_model else 'numpy'
            self.output_type = 'numpy'
        else:
            self.output_type = output_type

        # if not dtype is specified, we try to get it from the model
        if dtype is None:
            self.dtype = get_dtype_from_model_func(func=model,
                                                   default=np.float32)
        else:
            if dtype in [np.float32, np.float64]:
                self.dtype = np.dtype(dtype)
            raise ValueError("dtype must be either np.float32 or np.float64")

        self.background, self.N, self.M, _ = \
            input_to_cupy_array(background, order=self.order,
                                convert_to_dtype=self.dtype)

        self.random_state = random_state

        if isinstance(background,
                      pandas.DataFrame) or isinstance(background,
                                                      cudf.DataFrame):
            self.feature_names = background.columns.to_list()
        else:
            self.feature_names = [None for _ in range(len(background))]

        # evaluate the model in background to get the expected_value
        self.expected_value = self.link_fn(
            cp.mean(
                model_func_call(X=self.background,
                                model_func=self.model,
                                gpu_model=self.gpu_model),
                axis=0
            )
        )

        # D tells us the dimension of the model. For example, `predict_proba`
        # functions typically return n values for n classes as opposed to
        # 1 valued for a typical `predict`
        if len(self.expected_value.shape) == 0:
            self.D = 1
        else:
            self.D = self.expected_value.shape[0]
