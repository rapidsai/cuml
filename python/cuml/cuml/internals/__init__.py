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

from cuml.internals.available_devices import is_cuda_available
from cuml.internals.base_helpers import BaseMetaClass, _tags_class_and_instance
from cuml.internals.api_decorators import (
    api_base_fit_transform,
    api_base_return_any_skipall,
    api_base_return_any,
    api_base_return_array_skipall,
    api_base_return_array,
    api_base_return_generic_skipall,
    api_base_return_generic,
    api_base_return_sparse_array,
    api_return_any,
    api_return_array,
    api_return_generic,
    api_return_sparse_array,
    exit_internal_api,
)
from cuml.internals.api_context_managers import (
    in_internal_api,
    set_api_output_dtype,
    set_api_output_type,
)

if is_cuda_available():
    from cuml.internals.internals import GraphBasedDimRedCallback

from cuml.internals.constants import CUML_WRAPPED_FLAG
