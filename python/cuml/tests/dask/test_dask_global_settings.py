#
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
# pylint: disable=no-member

from time import sleep

import pytest
from dask import delayed

import cuml
from cuml import set_global_output_type, using_output_type
from cuml.internals.api_context_managers import _using_mirror_output_type
from cuml.internals.global_settings import (
    _global_settings_data,
    _GlobalSettingsData,
    GlobalSettings,
)

test_output_types_str = ("numpy", "numba", "cupy", "cudf")
test_global_settings_data_obj = _GlobalSettingsData()


def test_set_global_output_type():
    """Ensure that set_global_output_type is thread-safe"""

    def check_correct_type(index):
        output_type = test_output_types_str[index]
        # Force a race condition
        if index == 0:
            sleep(0.1)
        set_global_output_type(output_type)
        sleep(0.5)
        return cuml.global_settings.output_type == output_type

    results = [
        delayed(check_correct_type)(index)
        for index in range(len(test_output_types_str))
    ]

    assert (delayed(all)(results)).compute()


def test_using_output_type():
    """Ensure that using_output_type is thread-safe"""

    def check_correct_type(index):
        output_type = test_output_types_str[index]
        # Force a race condition
        if index == 0:
            sleep(0.1)
        with using_output_type(output_type):
            sleep(0.5)
            return cuml.global_settings.output_type == output_type

    results = [
        delayed(check_correct_type)(index)
        for index in range(len(test_output_types_str))
    ]

    assert (delayed(all)(results)).compute()


def test_using_mirror_output_type():
    """Ensure that _using_mirror_output_type is thread-safe"""

    def check_correct_type(index):
        # Force a race condition
        if index == 0:
            sleep(0.1)
        if index % 2 == 0:
            with _using_mirror_output_type():
                sleep(0.5)
                return cuml.global_settings.output_type == "mirror"
        else:
            output_type = test_output_types_str[index]
            with using_output_type(output_type):
                sleep(0.5)
                return cuml.global_settings.output_type == output_type

    results = [
        delayed(check_correct_type)(index)
        for index in range(len(test_output_types_str))
    ]

    assert (delayed(all)(results)).compute()


def test_global_settings_data():
    """Ensure that GlobalSettingsData objects are properly initialized
    per-thread"""

    def check_initialized(index):
        if index == 0:
            sleep(0.1)

        with pytest.raises(AttributeError):
            _global_settings_data.testing_index  # pylint: disable=W0104
        _global_settings_data.testing_index = index

        sleep(0.5)
        return (
            test_global_settings_data_obj.shared_state["_output_type"] is None
            and test_global_settings_data_obj.shared_state["root_cm"] is None
            and _global_settings_data.testing_index == index
        )

    results = [delayed(check_initialized)(index) for index in range(5)]

    assert (delayed(all)(results)).compute()


def test_global_settings():
    """Ensure that GlobalSettings acts as a proper thread-local borg"""

    def check_settings(index):
        # Force a race condition
        if index == 0:
            sleep(0.1)
        cuml.global_settings.index = index
        sleep(0.5)
        return (
            cuml.global_settings.index == index
            and cuml.global_settings.index == GlobalSettings().index
        )

    results = [delayed(check_settings)(index) for index in range(5)]

    assert (delayed(all)(results)).compute()
