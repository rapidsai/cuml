#
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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


from enum import Enum, auto
from cuml.internals.device_support import GPU_ENABLED
from cuml.internals.safe_imports import cpu_only_import, gpu_only_import

cudf = gpu_only_import("cudf")
cp = gpu_only_import("cupy")
cpx_sparse = gpu_only_import("cupyx.scipy.sparse")
np = cpu_only_import("numpy")
pandas = cpu_only_import("pandas")
scipy_sparse = cpu_only_import("scipy.sparse")


class MemoryTypeError(Exception):
    """An exception thrown to indicate inconsistent memory type selection"""


class MemoryType(Enum):
    device = auto()
    host = auto()
    managed = auto()
    mirror = auto()

    @classmethod
    def from_str(cls, memory_type):
        if isinstance(memory_type, str):
            memory_type = memory_type.lower()
        elif isinstance(memory_type, cls):
            return memory_type

        try:
            return cls[memory_type]
        except KeyError:
            raise ValueError(
                'Parameter memory_type must be one of "device", '
                '"host", "managed" or "mirror"'
            )

    @property
    def xpy(self):
        if self is MemoryType.host or (
            self is MemoryType.mirror and not GPU_ENABLED
        ):
            return np
        else:
            return cp

    @property
    def xdf(self):
        if self is MemoryType.host or (
            self is MemoryType.mirror and not GPU_ENABLED
        ):
            return pandas
        else:
            return cudf

    @property
    def xsparse(self):
        if self is MemoryType.host or (
            self is MemoryType.mirror and not GPU_ENABLED
        ):
            return scipy_sparse
        else:
            return cpx_sparse

    @property
    def is_device_accessible(self):
        return self in (MemoryType.device, MemoryType.managed)

    @property
    def is_host_accessible(self):
        return self in (MemoryType.host, MemoryType.managed)
