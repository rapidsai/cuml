#
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
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
                'Parameter device_type must be one of "cpu" or "gpu"'
            )

    def is_compatible(self, mem_type: MemoryType) -> bool:
        if self is DeviceType.device:
            return mem_type is MemoryType.device
        else:
            return mem_type is MemoryType.host

    @property
    def default_memory_type(self):
        if self is DeviceType.device:
            return MemoryType.device
        else:
            return MemoryType.host
