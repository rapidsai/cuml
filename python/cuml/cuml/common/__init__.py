#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

# from cuml.internals.array import CumlArray
# from cuml.internals.array_sparse import SparseCumlArray

from cuml.common.pointer_utils import device_of_gpu_matrix
from cuml.common.timing_utils import timed
from cuml.internals import logger
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.input_utils import (
    input_to_cuml_array,
    input_to_host_array,
    input_to_host_array_with_sparse_support,
    sparse_scipy_to_cp,
)
from cuml.internals.memory_utils import (
    rmm_cupy_ary,
    set_global_output_type,
    using_memory_type,
    using_output_type,
    with_cupy_rmm,
)

# utils


# legacy to be removed after complete CumlAray migration


__all__ = [
    "CumlArray",
    "SparseCumlArray",
    "device_of_gpu_matrix",
    "input_to_cuml_array",
    "input_to_host_array",
    "input_to_host_array_with_sparse_support",
    "rmm_cupy_ary",
    "set_global_output_type",
    "using_memory_type",
    "using_output_type",
    "with_cupy_rmm",
    "sparse_scipy_to_cp",
    "timed",
]
