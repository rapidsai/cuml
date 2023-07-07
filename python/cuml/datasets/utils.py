# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")


def _create_rs_generator(random_state):
    """
    This is a utility function that returns an instance of CuPy RandomState
    Parameters
    ----------
    random_state : None, int, or CuPy RandomState
        The random_state from which the CuPy random state is generated
    """

    if isinstance(random_state, (type(None), int)):
        return cp.random.RandomState(seed=random_state)
    elif isinstance(random_state, cp.random.RandomState):
        return random_state
    else:
        raise ValueError("random_state type must be int or CuPy RandomState")
