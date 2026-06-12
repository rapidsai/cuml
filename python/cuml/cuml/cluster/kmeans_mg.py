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

    def fit(self, X, y=None, sample_weight=None, *, convert_dtype=True):
        if isinstance(X, (list, tuple)):
            return self._fit_mg_parts(
                X,
                sample_weight_parts=sample_weight,
                convert_dtype=convert_dtype,
            )

        return super().fit(
            X,
            y=y,
            sample_weight=sample_weight,
            convert_dtype=convert_dtype,
        )
