#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cuml.ensemble
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.interop import UnsupportedOnGPU

__all__ = ("RandomForestRegressor", "RandomForestClassifier")


class RandomForestRegressor(ProxyBase):
    _gpu_class = cuml.ensemble.RandomForestRegressor

    def _gpu_fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise UnsupportedOnGPU("`sample_weight` is not supported")

        try:
            y = input_to_cuml_array(y, convert_to_mem_type=False)[0]
        except ValueError:
            raise
        else:
            if len(y.shape) > 1 and y.shape[1] > 1:
                raise UnsupportedOnGPU(
                    "Multi-output targets are not supported"
                )
        return self._gpu.fit(X, y)

    def _gpu_score(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise UnsupportedOnGPU("`sample_weight` is not supported")
        return self._gpu.score(X, y)

    def __len__(self):
        return self._call_method("__len__")

    def __iter__(self):
        return self._call_method("__iter__")

    def __getitem__(self, index):
        return self._call_method("__getitem__", index)


class RandomForestClassifier(ProxyBase):
    _gpu_class = cuml.ensemble.RandomForestClassifier

    def _gpu_fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise UnsupportedOnGPU("`sample_weight` is not supported")

        try:
            y = input_to_cuml_array(y, convert_to_mem_type=False)[0]
        except ValueError:
            raise
        else:
            if len(y.shape) > 1 and y.shape[1] > 1:
                if self.oob_score:
                    raise ValueError(
                        "The type of target cannot be used to compute OOB estimates"
                    )
                else:
                    raise UnsupportedOnGPU(
                        "Multi-output targets are not supported"
                    )
        return self._gpu.fit(X, y)

    def _gpu_score(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise UnsupportedOnGPU("`sample_weight` is not supported")
        return self._gpu.score(X, y)

    def __len__(self):
        return self._call_method("__len__")

    def __iter__(self):
        return self._call_method("__iter__")

    def __getitem__(self, index):
        return self._call_method("__getitem__", index)
