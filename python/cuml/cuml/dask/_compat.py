# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import functools

import dask
import packaging.version


@functools.lru_cache
def DASK_2025_4_0():
    return packaging.version.parse(
        dask.__version__
    ) >= packaging.version.parse("2025.4.0")
