# Copyright (c) 2020, NVIDIA CORPORATION.
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

import numpy as np

from cuml.dask.common.base import BaseEstimator, DelayedTransformMixin
from cuml.dask.common.input_utils import DistributedDataHandler

from cuml.manifold.umap import UMAP as cumlUMAP


class UMAP(BaseEstimator,
           DelayedTransformMixin):
    def __init__(self, model, client=None, **kwargs):
        super(UMAP, self).__init__(client, **kwargs)
        self.local_model = model

    def fit_transform(self, X, y=None, convert_dtype=True):
        self.fit(X, y=y, convert_dtype=convert_dtype)
        return self.transform(X, convert_dtype=convert_dtype)

    def transform(self, X, convert_dtype=True):
        data = DistributedDataHandler.create(data=X, client=self.client)
        self.datatype = data.datatype
        return self._transform(X,
                               convert_dtype=convert_dtype)
