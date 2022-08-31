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

from cuml.common.array import CumlArray
from cuml.common.array_sparse import SparseCumlArray

# utils

from cuml.common.import_utils import has_cupy
from cuml.common.import_utils import has_dask
from cuml.common.import_utils import check_min_numba_version
from cuml.common.import_utils import check_min_cupy_version, has_scipy

from cuml.common.input_utils import input_to_cuml_array
from cuml.common.input_utils import input_to_host_array
from cuml.common.input_utils import inp_array

from cuml.common.memory_utils import rmm_cupy_ary
from cuml.common.memory_utils import set_global_output_type
from cuml.common.memory_utils import using_output_type
from cuml.common.memory_utils import with_cupy_rmm

from cuml.common.pointer_utils import device_of_gpu_matrix

# legacy to be removed after complete CumlAray migration

from cuml.common.input_utils import sparse_scipy_to_cp
from cuml.common.timing_utils import timed

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
    "inp_array",
    "rmm_cupy_ary",
    "set_global_output_type",
    "using_output_type",
    "with_cupy_rmm",
    "sparse_scipy_to_cp",
    "timed",
]
