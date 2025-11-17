#
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#


from enum import Enum, auto

import cudf
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import numpy as np
import pandas
import scipy.sparse as scipy_sparse


class MemoryTypeError(Exception):
    """An exception thrown to indicate inconsistent memory type selection"""


class MemoryType(Enum):
    device = auto()
    host = auto()

    @classmethod
    def from_str(cls, memory_type):
        if isinstance(memory_type, str):
            memory_type = memory_type.lower()
        elif isinstance(memory_type, cls):
            return memory_type

        try:
            return cls[memory_type]
        except KeyError:
            raise ValueError(
                "Parameter memory_type must be one of 'device', or 'host'"
            )

    @property
    def xpy(self):
        if self is MemoryType.host:
            return np
        else:
            return cp

    @property
    def xdf(self):
        if self is MemoryType.host:
            return pandas
        else:
            return cudf

    @property
    def xsparse(self):
        if self is MemoryType.host:
            return scipy_sparse
        else:
            return cpx_sparse

    @property
    def is_device_accessible(self):
        return self is MemoryType.device

    @property
    def is_host_accessible(self):
        return self is MemoryType.host
