# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
from dask.distributed import wait

from cuml.dask.common import raise_exception_from_futures


def _raise_exception():
    raise ValueError("intentional exception")


def test_dask_exceptions(client):
    fut = client.submit(_raise_exception)
    wait(fut)

    with pytest.raises(RuntimeError):
        raise_exception_from_futures([fut])
