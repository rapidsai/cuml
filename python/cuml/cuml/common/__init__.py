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

# from cuml.internals.array import CumlArray
# from cuml.internals.array_sparse import SparseCumlArray

from cuml.common.device_selection import using_device_type
from cuml.internals import logger
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.available_devices import is_cuda_available
from cuml.internals.import_utils import (
    check_min_cupy_version,
    check_min_numba_version,
    has_cupy,
    has_dask,
    has_scipy,
)
from cuml.internals.input_utils import (
    input_to_cuml_array,
    input_to_host_array,
    input_to_host_array_with_sparse_support,
)
from cuml.internals.memory_utils import (
    rmm_cupy_ary,
    set_global_output_type,
    using_memory_type,
    using_output_type,
    with_cupy_rmm,
)

# utils


if is_cuda_available():
    from cuml.common.pointer_utils import device_of_gpu_matrix

# legacy to be removed after complete CumlAray migration

from cuml.common.timing_utils import timed
from cuml.internals.input_utils import sparse_scipy_to_cp

__all__ = [
    "CumlArray",
    "SparseCumlArray",
    "device_of_gpu_matrix",
    "has_cupy",
    "has_dask",
    "check_min_numba_version",
    "check_min_cupy_version",
    "has_scipy",
    "input_to_cuml_array",
    "input_to_host_array",
    "input_to_host_array_with_sparse_support",
    "rmm_cupy_ary",
    "set_global_output_type",
    "using_device_type",
    "using_memory_type",
    "using_output_type",
    "with_cupy_rmm",
    "sparse_scipy_to_cp",
    "timed",
]
