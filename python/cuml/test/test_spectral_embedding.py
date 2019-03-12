# Copyright (c) 2018, NVIDIA CORPORATION.
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
import pytest

from cuml.manifold.spectral import SpectralEmbedding as cumlSpectral
from sklearn.manifold import SpectralEmbedding as skSpectral

import cudf
import pandas as pd
import numpy as np

from sklearn import datasets

def test_output_cuml_vs_sklearn():
    iris = datasets.load_iris()
    data = iris.data

    cumlEmbedding = cumlSpectral(n_neighbors = 10, n_components = 2)
    skEmbedding = skSpectral(n_neighbors = 10, n_components = 2)

    cumlOutput = cumlEmbedding.fit_transform(data)
    skOutput = skEmbedding.fit_transform(data)

    print(str(cumlOutput))
    print(str(skOutput))

    assert 1==0






