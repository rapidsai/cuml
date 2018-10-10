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
from cuSKL import DBSCAN as cuDBSCAN
from sklearn.cluster import DBSCAN as skDBSCAN
from test_utils import array_equal
import pygdf
import numpy as np


@pytest.mark.parametrize('datatype', [np.float32, np.float64])

def test_dbscan_predict(datatype):
    gdf = pygdf.DataFrame()
    gdf['0']=np.asarray([1,2,2,8,8,25],dtype=datatype)
    gdf['1']=np.asarray([2,2,3,7,8,80],dtype=datatype)

    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]], dtype = datatype)

    print("Calling fit_predict")
    cudbscan = cuDBSCAN(eps = 3, min_samples = 2)
    cu_labels = cudbscan.fit_predict(gdf)
    skdbscan = skDBSCAN(eps = 3, min_samples = 2)
    sk_labels = skdbscan.fit_predict(X)
    print(X.shape[0])
    for i in range(X.shape[0]):
        assert cu_labels[i] == sk_labels[i]


