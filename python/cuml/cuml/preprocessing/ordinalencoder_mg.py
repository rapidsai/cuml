#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
import cupy as cp
from cudf import DataFrame

from cuml.preprocessing.encoders import OrdinalEncoder


class OrdinalEncoderMG(OrdinalEncoder):
    def __init__(self, *, client=None, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def _check_input_fit(self, X, is_categories=False):
        """Helper function to check input of fit within the multi-gpu model"""
        import dask.array

        from cuml.dask.common.dask_arr_utils import to_dask_cudf

        if isinstance(X, (dask.array.core.Array, cp.ndarray)):
            self._set_input_type("array")
            if is_categories:
                X = X.transpose()
            if isinstance(X, cp.ndarray):
                return DataFrame(X)
            else:
                return to_dask_cudf(X, client=self.client)
        else:
            self._set_input_type("df")
            return X

    def _unique(self, inp):
        return inp.unique().compute()

    def _has_unknown(self, X_cat, encoder_cat):
        return not X_cat.isin(encoder_cat).all().compute()
