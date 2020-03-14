#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

import numpy as np
import cupy as cp

from dask import delayed
from dask.distributed import Client
import pytest

from cuml.dask.common.func import tree_reduce


@pytest.mark.parametrize("n_parts", [1, 2, 10, 15])
def test_tree_reduce(n_parts, cluster):

    client = Client(cluster)

    a = [delayed(i) for i in range(n_parts)]
    b = tree_reduce(a)
    c = client.compute(b, sync=True)

    assert(sum(range(n_parts)) == c)








