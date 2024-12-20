#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from cuml.internals.global_settings import GlobalSettings
from cuml.internals.safe_imports import (
    gpu_only_import_from,
    UnavailableNullContext,
)

cupy_using_allocator = gpu_only_import_from(
    "cupy.cuda", "using_allocator", alt=UnavailableNullContext
)
rmm_cupy_allocator = gpu_only_import_from(
    "rmm.allocators.cupy", "rmm_cupy_allocator"
)

global_settings = GlobalSettings()

class CumlAPIContext:
    """A context manager controlling behavior at the cuML API boundary

    This context manager does the following:
    1. Increment a thread-local counter for how deep a call is within the cuML
    API boundary
    2. If we are at the API boundary, enter a cupy.cuda.using_allocator stack
    to ensure RMM is used for cupy allocations
    3. If output_type, memory_type, or device_type are specified in the
    initializer, these will be set in cuML's global settings and the
    previous value restored when the context exits.
    """

    def __enter__(self):
        if global_settings.api_depth == 0:
            self.cupy_allocator_cm = cupy_using_allocator(rmm_cupy_allocator)
            self.cupy_allocator_cm.__enter__()
        else:
            self.cupy_allocator_cm = None
        global_settings.increment_api_depth()

    def __exit__(self, exc_type, exc_val, exc_tb):
        global_settings.decrement_api_depth()
        if self.cupy_allocator_cm is not None:
            self.cupy_allocator_cm.__exit__()


class GlobalSettingsContext:
    def __init__(
        self,
        output_type=None,
        memory_type=None,
        device_type=None,
        output_dtype=None
    ):
        self.prev_output_type = global_settings.output_type
        self.output_type = output_type
        self.prev_memory_type = global_settings.memory_type
        self.memory_type = memory_type
        self.prev_device_type = global_settings.device_type
        self.device_type = device_type
        self.prev_output_dtype = global_settings.output_dtype
        self.output_dtype = output_dtype

    def __enter__(self):
        if self.output_type is not None:
            global_settings.output_type = self.output_type
        if self.memory_type is not None:
            global_settings.memory_type = self.memory_type
        if self.device_type is not None:
            global_settings.device_type = self.device_type
        if self.output_dtype is not None:
            global_settings.output_dtype = self.output_dtype

    def __exit__(self):
        global_settings.output_dtype = self.prev_output_dtype
        global_settings.device_type = self.prev_device_type
        global_settings.memory_type = self.prev_memory_type
        global_settings.output_type = self.prev_output_type
