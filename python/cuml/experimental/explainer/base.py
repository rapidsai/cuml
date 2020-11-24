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

import numpy as np

from cuml.experimental.explainer.common import get_dtype_from_model_func
from cuml.experimental.explainer.common import get_handle_from_cuml_model_func
from cuml.experimental.explainer.common import get_link_fn_from_str
from cuml.experimental.explainer.common import get_tag_from_model_func
from cuml.common.input_utils import input_to_cupy_array


class SHAPBase():
    """
    Base class for SHAP based explainers.
    """

    def __init__(self,
                 *,
                 model,
                 data,
                 order=None,
                 default_order='C',
                 link='identity',
                 verbosity=False,
                 random_state=None,
                 gpu_model=None,
                 handle=None,
                 dtype=None,
                 output_type=None):

        if handle is None:
            self.handle = get_handle_from_cuml_model_func(model,
                                                          create_new=True)
        else:
            self.handle = handle

        if order is None:
            self.order = get_tag_from_model_func(func=model,
                                                 tag='preferred_input_order',
                                                 default=default_order)
        else:
            self.order = order

        self.link = link
        self.link_fn = get_link_fn_from_str(link)
        self.model = model
        if gpu_model is None:
            # todo: when sparse support is added, use this tag to see if
            # model can accept sparse data
            self.model_gpu_based = \
                get_tag_from_model_func(func=model,
                                        tag='X_types_gpu',
                                        default=None) is not None
        else:
            self.model_gpu_based = gpu_model

        if output_type is None:
            self.output_type = 'cupy' if self.model_gpu_based else 'numpy'
        else:
            self.output_type = output_type

        # if not dtype is specified, we try to get it from the model
        if dtype is None:
            self.dtype = get_dtype_from_model_func(func=model,
                                                   default=np.float32)
        else:
            self.dtype = np.dtype(dtype)

        self.background, self.N, self.M, _ = \
            input_to_cupy_array(data, order=self.order,
                                convert_to_dtype=self.dtype)
