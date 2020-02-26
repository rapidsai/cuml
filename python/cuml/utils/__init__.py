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

from cuml.utils.pointer_utils import device_of_gpu_matrix

from cuml.utils.memory_utils import rmm_cupy_ary
from cuml.utils.memory_utils import set_global_output_type
from cuml.utils.memory_utils import using_output_type
from cuml.utils.memory_utils import with_cupy_rmm

from cuml.utils.numba_utils import row_matrix, zeros, device_array_from_ptr

from cuml.utils.input_utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_cuml_array, input_to_dev_array, input_to_host_array, inp_array

from cuml.utils.import_utils import has_cupy, has_dask, \
    check_min_numba_version, check_min_cupy_version, has_scipy

from cuml.utils.kernel_utils import get_dtype_str
from cuml.utils.kernel_utils import cuda_kernel_factory
