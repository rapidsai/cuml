#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cuml.neighbors
from cuml.accel.estimator_proxy import ProxyBase

__all__ = (
    "NearestNeighbors",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "KernelDensity",
)


class NearestNeighbors(ProxyBase):
    _gpu_class = cuml.neighbors.NearestNeighbors
    _other_attributes = frozenset(("_fit_method", "_tree"))


class KNeighborsClassifier(ProxyBase):
    _gpu_class = cuml.neighbors.KNeighborsClassifier
    _other_attributes = frozenset(("_fit_method", "_tree"))


class KNeighborsRegressor(ProxyBase):
    _gpu_class = cuml.neighbors.KNeighborsRegressor
    _other_attributes = frozenset(("_fit_method", "_tree"))


class KernelDensity(ProxyBase):
    _gpu_class = cuml.neighbors.KernelDensity
