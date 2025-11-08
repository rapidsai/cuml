#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cuml.cluster
from cuml.accel.estimator_proxy import ProxyBase

__all__ = ("HDBSCAN",)


class HDBSCAN(ProxyBase):
    _gpu_class = cuml.cluster.HDBSCAN
    _not_implemented_attributes = frozenset(
        ("exemplars_", "outlier_scores_", "relative_validity_")
    )
