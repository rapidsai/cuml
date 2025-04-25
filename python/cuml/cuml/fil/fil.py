#
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import warnings

from cuml.experimental.fil.fil import (
    ForestInference as ExperimentalForestInference,
)
from cuml.internals.array import CumlArray
from cuml.internals.global_settings import GlobalSettings


class ForestInference(ExperimentalForestInference):
    def __init__(
        self,
        *,
        treelite_model=None,
        handle=None,
        output_type=None,
        verbose=False,
        is_classifier=False,
        output_class=None,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
        precision="single",
        device_id=0,
    ):
        super().__init__(
            treelite_model=treelite_model,
            handle=handle,
            output_type=output_type,
            verbose=verbose,
            is_classifier=is_classifier,
            output_class=output_class,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            precision=precision,
            device_id=device_id,
        )
        if treelite_model is not None and self.num_outputs() == 1:
            warning_msg = (
                "In RAPIDS 25.06, the output shape of ForestInference predict"
                " will include an extra dimension to accommodate multi-target"
                " regressors and classifiers."
            )
            if self.is_classifier:
                warning_msg = (
                    f"{warning_msg} For performance and memory reasons,"
                    " predict_proba will also return solely the positive class"
                    " probability for binary classifiers, consistent with "
                    " XGBoost."
                )
            warnings.warn(warning_msg, FutureWarning)

    def predict_proba(self, X, *, preds=None, chunk_size=None) -> CumlArray:
        results = super().predict_proba(X, preds=preds, chunk_size=chunk_size)
        if len(results.shape) == 2 and results.shape[-1] == 1:
            results = results.to_output("array").flatten()
            results = GlobalSettings().xpy.stack(
                [1 - results, results], axis=1
            )
        return results

    def predict(
        self, X, *, preds=None, chunk_size=None, threshold=None
    ) -> CumlArray:

        results = super().predict(
            X, preds=preds, chunk_size=chunk_size, threshold=threshold
        )
        if (
            self.is_classifier
            and len(results.shape) == 2
            and results.shape[-1] == 1
        ):
            results = results.to_output("array").flatten()
        return results
