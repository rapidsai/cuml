#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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


def set_global_device_type(device_type):
    if (isinstance(device_type, str)):
        device_type = device_type.lower()

    if device_type not in ['cpu', 'gpu', None]:
        raise ValueError('Parameter device_type must be one of "cpu", '
                         '"gpu", or None')

    cuml.global_settings.device_type = device_type


@contextlib.contextmanager
def using_device_type(device_type):
    prev_device_type = cuml.global_settings.device_type
    try:
        set_global_device_type(device_type)
        yield prev_device_type
    finally:
        cuml.global_settings.device_type = prev_device_type


def set_global_memory_type(memory_type):
    if (isinstance(memory_type, str)):
        memory_type = memory_type.lower()

    if memory_type not in ['global', 'host', 'managed', 'mirror', None]:
        raise ValueError('Parameter memory_type must be one of "global", '
                         '"host", "managed", "mirror" or None')

    cuml.global_settings.memory_type = memory_type


@contextlib.contextmanager
def using_memory_type(memory_type):
    prev_memory_type = cuml.global_settings.memory_type
    try:
        set_global_device_type(memory_type)
        yield prev_memory_type
    finally:
        cuml.global_settings.memory_type = prev_memory_type
