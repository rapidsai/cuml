# Copyright (c) 2019, NVIDIA CORPORATION.
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

# Note: this isn't a very strict test, the goal is to test the Cython
# interface and cover all the parameters

import cuml
import pytest
import numpy as np


# Testing parameters

dtype = ['single', 'double']
n_samples = [100, 100000]
n_features = [10, 100]
n_informative = [5, 7]
n_targets = [1, 3]
shuffle = [True, False]
effective_rank = [None, 6]
random_state = [None, 1234]
bias = [-4.0]
noise = [3.5]


@pytest.mark.parametrize('dtype', dtype)
@pytest.mark.parametrize('n_samples', n_samples)
@pytest.mark.parametrize('n_features', n_features)
@pytest.mark.parametrize('n_informative', n_informative)
@pytest.mark.parametrize('n_targets', n_targets)
@pytest.mark.parametrize('shuffle', shuffle)
@pytest.mark.parametrize('effective_rank', effective_rank)
@pytest.mark.parametrize('random_state', random_state)
@pytest.mark.parametrize('bias', bias)
@pytest.mark.parametrize('noise', noise)
def test_make_blobs_scalar_parameters(dtype, n_samples, n_features,
                                      n_informative, n_targets, shuffle,
                                      effective_rank, random_state, bias,
                                      noise):

    out, values = cuml.make_regression(n_samples=n_samples,
                                       n_features=n_features,
                                       n_informative=n_informative,
                                       n_targets=n_targets, bias=bias,
                                       effective_rank=effective_rank,
                                       noise=noise, shuffle=shuffle,
                                       random_state=random_state, dtype=dtype)

    # we can use cupy in the future
    labels_np = values.copy_to_host()

    assert out.shape == (n_samples, n_features), "out shape mismatch"
    assert values.shape == (n_samples, n_targets), "values shape mismatch"
