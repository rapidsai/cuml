# Copyright (c) 2021, NVIDIA CORPORATION.
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

"""
The code is based on the similar utility function from SHAP:
https://github.com/slundberg/shap/blob/9411b68e8057a6c6f3621765b89b24d82bee13d4/shap/utils/_legacy.py
This version makes use of cuml kmeans instead of sklearn for speed.
"""

import cupy as np
from cuml import KMeans
from cuml.preprocessing import SimpleImputer
from scipy.sparse import issparse


def kmeans(X, k, round_values=True):
    group_names = [str(i) for i in range(X.shape[1])]
    if str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
        group_names = X.columns
        X = X.values

    # in case there are any missing values in data impute them
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

    if round_values:
        for i in range(k):
            for j in range(X.shape[1]):
                xj = X[:, j].toarray().flatten() if issparse(
                    X) else X[:, j]  # sparse support courtesy of @PrimozGodec
                ind = np.argmin(np.abs(xj - kmeans.cluster_centers_[i, j]))
                kmeans.cluster_centers_[i, j] = X[ind, j]
    return DenseData(
        kmeans.cluster_centers_,
        group_names,
        None,
        1.0 *
        cp.bincount(
            kmeans.labels_))


class Data:
    def __init__(self):
        pass


class DenseData(Data):
    def __init__(self, data, group_names, *args):
        self.groups = args[0] if len(args) > 0 and args[0] is not None else [
            np.array([i]) for i in range(len(group_names))]

        length = sum(len(g) for g in self.groups)
        num_samples = data.shape[0]
        t = False
        if length != data.shape[1]:
            t = True
            num_samples = data.shape[1]

        valid = (
            not t and length == data.shape[1]) or (
            t and length == data.shape[0])
        assert valid, "# of names must match data matrix!"

        self.weights = args[1] if len(args) > 1 else cp.ones(num_samples)
        self.weights /= cp.sum(self.weights)
        wl = len(self.weights)
        valid = (not t and wl == data.shape[0]) or (t and wl == data.shape[1])
        assert valid, "# weights must match data matrix!"

        self.transposed = t
        self.group_names = group_names
        self.data = data
        self.groups_size = len(self.groups)
