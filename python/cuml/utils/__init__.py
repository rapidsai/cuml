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

from cuml.utils.pointer_utils import device_of_gpu_matrix
from cuml.utils.numba_utils import row_matrix, zeros
from cuml.utils.input_utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array

from cuml.utils.import_utils import has_cupy, has_dask
