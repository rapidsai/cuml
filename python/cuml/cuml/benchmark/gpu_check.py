#
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
"""GPU and CUDA availability detection for benchmark tools.

This module provides utilities to detect whether cuML is available,
allowing the benchmark tools to run in CPU-only mode on systems 
without NVIDIA GPUs or cuML installed.
"""


def _check_cuml():
    """Check if cuML is available and functional."""
    try:
        import cuml  # noqa: F401
        return True
    except ImportError:
        return False


# Check if cuML is available - this implies all GPU dependencies are available
HAS_CUML = _check_cuml()


def is_gpu_available():
    """Check if GPU mode is available.
    
    Returns
    -------
    bool
        True if cuML is available.
    """
    return HAS_CUML


def get_available_input_types():
    """Get list of available input types based on GPU availability.
    
    Returns
    -------
    list
        List of available input type strings.
    """
    cpu_types = ["numpy", "pandas"]
    if is_gpu_available():
        return cpu_types + ["cupy", "cudf", "gpuarray", "gpuarray-c"]
    return cpu_types


def get_status_string():
    """Get a string describing the current GPU/CPU status.
    
    Returns
    -------
    str
        Human-readable status string.
    """
    if HAS_CUML:
        return "GPU mode (cuML available)"
    else:
        return "CPU-only mode (cuML not installed)"
