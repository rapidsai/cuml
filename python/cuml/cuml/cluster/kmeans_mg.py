#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from cuml.cluster import KMeans


class KMeansMG(KMeans):
    """
    A Multi-Node Multi-GPU implementation of KMeans
    """

    _multi_gpu = True

    def __init__(self, *, handle, **kwargs):
        self.handle = handle
        super().__init__(**kwargs)
