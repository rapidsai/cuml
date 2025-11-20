#
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
from enum import Enum, auto

import cudf
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import numpy as np
import pandas as pd
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
        return np if self is MemoryType.host else cp

    @property
    def xdf(self):
        return pd if self is MemoryType.host else cudf

    @property
    def xsparse(self):
        return scipy_sparse if self is MemoryType.host else cpx_sparse
