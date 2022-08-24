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


import pytest
import cuml
from cuml.common.device_selection import using_device_type, using_memory_type


@pytest.mark.parametrize('device_type', ['cpu', 'gpu', None])
def test_device_type(device_type):
    initial_device_type = cuml.global_settings.device_type
    with using_device_type(device_type):
        assert cuml.global_settings.device_type == device_type
    assert cuml.global_settings.device_type == initial_device_type


def test_device_type_exception():
    with pytest.raises(ValueError):
        with using_device_type('wrong_option'):
            assert True


@pytest.mark.parametrize('memory_type', ['global', 'host', 'managed',
                                         'mirror', None])
def test_memory_type(memory_type):
    initial_memory_type = cuml.global_settings.memory_type
    with using_memory_type(memory_type):
        assert cuml.global_settings.memory_type == memory_type
    assert cuml.global_settings.memory_type == initial_memory_type


def test_memory_type_exception():
    with pytest.raises(ValueError):
        with using_memory_type('wrong_option'):
            assert True
