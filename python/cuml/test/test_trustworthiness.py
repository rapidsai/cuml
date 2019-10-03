# Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

from cuml.manifold import TSNE

from sklearn.manifold.t_sne import trustworthiness as sklearn_trustworthiness
from cuml.metrics import trustworthiness as cuml_trustworthiness

from sklearn import datasets
import pandas as pd
import numpy as np
import cudf
import pytest

dataset_names = ['digits', 'boston', 'iris', 'breast_cancer',
                 'diabetes']

@pytest.mark.parametrize('name', dataset_names)
def test_trustworthiness(name):
    datasets
    X = eval("datasets.load_{}".format(name))().data
    X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X))
    eps = 0.001
    
    for i in range(1):
        print("iteration = ", i)

        tsne = TSNE(2, random_state=i, verbose=0, learning_rate=2+i)

        Y = tsne.fit_transform(X_cudf).to_pandas().values
        nans = np.sum(np.isnan(Y))
        cuml_trust = cuml_trustworthiness(X, Y)
        sklearn_trust = sklearn_trustworthiness(X, Y)
        
        assert (sklearn_trust * (1 - eps) <= cuml_trust and
                cuml_trust <= sklearn_trust * (1 + eps))
        assert trust > 0.76
        assert nans == 0
        del Y

        # Reuse
        Y = tsne.fit_transform(X)
        nans = np.sum(np.isnan(Y))
        cuml_trust = cuml_trustworthiness(X, Y)
        sklearn_trust = sklearn_trustworthiness(X, Y)
        
        assert (sklearn_trust * (1 - eps) <= cuml_trust and
                cuml_trust <= sklearn_trust * (1 + eps))
        assert trust > 0.76
        assert nans == 0
        del Y

        # Again
        tsne = TSNE(2, random_state=i+2, verbose=1, learning_rate=2+i+2,
                    method="exact")

        Y = tsne.fit_transform(X_cudf).to_pandas().values
        nans = np.sum(np.isnan(Y))
        cuml_trust = cuml_trustworthiness(X, Y)
        sklearn_trust = sklearn_trustworthiness(X, Y)
        
        assert (sklearn_trust * (1 - eps) <= cuml_trust and
                cuml_trust <= sklearn_trust * (1 + eps))
        assert trust > 0.76
        assert nans == 0
        del Y

        # Reuse
        Y = tsne.fit_transform(X)
        nans = np.sum(np.isnan(Y))
        cuml_trust = cuml_trustworthiness(X, Y)
        sklearn_trust = sklearn_trustworthiness(X, Y)
        
        assert (sklearn_trust * (1 - eps) <= cuml_trust and
                cuml_trust <= sklearn_trust * (1 + eps))
        assert trust > 0.76
        assert nans == 0
        del Y
