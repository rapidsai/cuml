#
# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

from cuml.internals.mem_type import MemoryType


class DeviceTypeError(Exception):
    """An exception thrown to indicate bad device type selection"""


class DeviceType(Enum):
    host = auto()
    device = auto()

    @classmethod
    def from_str(cls, device_type):
        if isinstance(device_type, str):
            device_type = device_type.lower()

        if device_type in ("cpu", "host", DeviceType.host):
            return cls.host
        elif device_type in ("gpu", "device", DeviceType.device):
            return cls.device
        else:
            raise ValueError(
                'Parameter device_type must be one of "cpu" or ' '"gpu"'
            )

    def is_compatible(self, mem_type: MemoryType) -> bool:
        if self is DeviceType.device:
            return mem_type.is_device_accessible
        else:
            return mem_type.is_host_accessible

    @property
    def default_memory_type(self):
        if self is DeviceType.device:
            return MemoryType.device
        else:
            return MemoryType.host
