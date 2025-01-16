#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
import numbers
import numpy as np


def check_random_seed(seed):
    """Turn a np.random.RandomState instance into a seed.
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return a random int as seed.
        If seed is an int, return it.
        If seed is a RandomState instance, derive a seed from it.
        Otherwise raise ValueError.
    """
    if seed is None:
        seed = np.random.RandomState(None)

    if isinstance(seed, numbers.Integral):
        return seed
    if isinstance(seed, np.random.RandomState):
        return seed.randint(
            low=0, high=np.iinfo(np.uint32).max, dtype=np.uint32
        )
    raise ValueError("%r cannot be used to create a seed." % seed)
