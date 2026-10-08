# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register and select supported benchmark execution backends."""

from ..registry import ACCEL_REGISTRY, CPU_REGISTRY, DASK_REGISTRY, SG_REGISTRY
from .accel import AccelBackend
from .base import Backend
from .cuml import CumlBackend
from .dask import DaskBackend
from .sklearn import SklearnBackend

BACKENDS: dict[str, Backend] = {
    "scikit-learn": SklearnBackend("scikit-learn", CPU_REGISTRY),
    "cuml": CumlBackend("cuml", SG_REGISTRY),
    "cuml.accel": AccelBackend("cuml.accel", ACCEL_REGISTRY),
    "cuml.dask": DaskBackend("cuml.dask", DASK_REGISTRY),
}


def get_backend(implementation: str) -> Backend:
    """Return the backend registered for an implementation.

    Parameters
    ----------
    implementation : str
        Registered benchmark implementation name.
    """
    return BACKENDS[implementation]
