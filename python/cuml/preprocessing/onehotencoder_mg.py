#
# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

from cuml.dask.common.dask_arr_utils import to_dask_cudf
from cuml.internals.safe_imports import gpu_only_import_from
from cuml.preprocessing.encoders import OneHotEncoder
import dask
from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")
DataFrame = gpu_only_import_from("cudf", "DataFrame")


class OneHotEncoderMG(OneHotEncoder):
    """
    A Multi-Node Multi-GPU implementation of OneHotEncoder

    Refer to the Dask OneHotEncoder implementation
    in `cuml.dask.preprocessing.encoders`.
    """

    def __init__(self, *, client=None, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def _check_input_fit(self, X, is_categories=False):
        """Helper function to check input of fit within the multi-gpu model"""
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
