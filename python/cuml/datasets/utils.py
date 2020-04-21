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

import cupy as cp


def _create_rs_generator(random_state):
    """
    This is a utility function that returns an instance of CuPy RandomState
    Parameters
    ----------
    random_state : None, int, or CuPy RandomState
        The random_state from which the CuPy random state is generated
    """

    if hasattr(random_state, '__module__'):
        rs_type = random_state.__module__ + '.' + type(random_state).__name__
    else:
        rs_type = type(random_state).__name__

    rs = None
    if rs_type == "NoneType" or rs_type == "int":
        rs = cp.random.RandomState(seed=random_state)
    elif rs_type == "cupy.random.generator.RandomState":
        rs = rs_type
    else:
        raise ValueError('random_state type must be int or CuPy RandomState')
    return rs
