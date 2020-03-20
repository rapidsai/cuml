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

from cuml.dask.common.input_utils import DistributedDataHandler
from cuml.dask.common.base import DelayedTransformMixin
from cuml.dask.manifold.base import BaseManifold

from cuml.manifold.umap import UMAP as cumlUMAP


class UMAP(BaseManifold,
           DelayedTransformMixin):
    def __init__(self, client=None, verbose=False, n_sampling=500, **kwargs):
        super(UMAP, self).__init__(client, **kwargs)
        self.local_model = cumlUMAP(verbose=verbose, MNMG=True, **kwargs)
        self.n_sampling = n_sampling

    def fit(self, X, y=None, convert_dtype=True):
        selection = np.random.choice(len(X), self.n_sampling)
        X = X[selection]
        X = X.compute()
        if y is not None:
            y = y[selection]
            y = y.compute()
        self.local_model.fit(X, y=y, convert_dtype=convert_dtype)

    def fit_transform(self, X, y=None, convert_dtype=True):
        self.fit(X, y=y, convert_dtype=convert_dtype)
        return self.transform(X, convert_dtype=convert_dtype)

    def transform(self, X, convert_dtype=True):
        data = DistributedDataHandler.create(data=X, client=self.client)
        self.datatype = data.datatype
        return self._transform(X,
                               convert_dtype=convert_dtype)
