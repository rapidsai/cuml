#
# Copyright (c) 2022, NVIDIA CORPORATION.
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


import cuml
import contextlib
from enum import Enum, auto


class DeviceType(Enum):
    host = auto(),
    device = auto()

    @staticmethod
    def from_str(device_type):
        if isinstance(device_type, str):
            device_type = device_type.lower()

        if device_type in ['cpu', 'host']:
            return DeviceType.host
        elif device_type in ['gpu', 'device']:
            return DeviceType.device
        else:
            raise ValueError('Parameter device_type must be one of "cpu" or '
                             '"gpu"')


def set_global_device_type(device_type):
    cuml.global_settings.device_type = DeviceType.from_str(device_type)


@contextlib.contextmanager
def using_device_type(device_type):
    prev_device_type = cuml.global_settings.device_type
    try:
        set_global_device_type(device_type)
        yield prev_device_type
    finally:
        cuml.global_settings.device_type = prev_device_type


class MemoryType(Enum):
    device = auto(),
    host = auto()
    managed = auto()
    mirror = auto()

    @staticmethod
    def from_str(memory_type):
        if isinstance(memory_type, str):
            memory_type = memory_type.lower()

        try:
            return MemoryType[memory_type]
        except KeyError:
            raise ValueError('Parameter memory_type must be one of "device", '
                             '"host", "managed" or "mirror"')


def set_global_memory_type(memory_type):
    cuml.global_settings.memory_type = MemoryType.from_str(memory_type)


@contextlib.contextmanager
def using_memory_type(memory_type):
    prev_memory_type = cuml.global_settings.memory_type
    try:
        set_global_memory_type(memory_type)
        yield prev_memory_type
    finally:
        cuml.global_settings.memory_type = prev_memory_type
