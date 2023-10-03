#
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
from cuml.internals.device_support import GPU_ENABLED
from cuml.internals.safe_imports import gpu_only_import_from, UnavailableError

try:
    from functools import cache  # requires Python >= 3.9
except ImportError:
    from functools import lru_cache

    cache = lru_cache(maxsize=None)


def gpu_available_no_context_creation():
    """
    Function tries to check if GPUs are available in the system without
    creating a CUDA context. We check for CuPy presence as a proxy of that.
    """
    try:
        import cupy

        return True
    except ImportError:
        return False


@cache
def is_cuda_available():
    try:
        return GPU_ENABLED and gpu_available_no_context_creation()
    except UnavailableError:
        return False
