#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

from time import sleep

from dask import delayed

import cuml
from cuml import set_global_output_type, using_output_type

test_output_types_str = ('numpy', 'numba', 'cupy', 'cudf')


def test_set_global_output_type():
    """Ensure that set_global_output_type is thread-safe"""
    def check_correct_type(index):
        output_type = test_output_types_str[index]
        # Force a race condition
        if index == 0:
            sleep(0.1)
        set_global_output_type(output_type)
        sleep(0.5)
        return cuml.global_output_type == output_type

    results = [
        delayed(check_correct_type)(index)
        for index in range(len(test_output_types_str))
    ]

    assert (delayed(all)(results)).compute()


def test_using_output_type():
    """Ensure that using_output_type is thread-safe"""
    def check_correct_type(index):
        output_type = test_output_types_str[index]
        with using_output_type(output_type):
            # Force a race condition
            if index == 0:
                sleep(0.1)
            sleep(0.5)
            return cuml.global_output_type == output_type

    results = [
        delayed(check_correct_type)(index)
        for index in range(len(test_output_types_str))
    ]

    assert (delayed(all)(results)).compute()
