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
import cupy as cp
from cuml import KMeans
from cuml.preprocessing import SimpleImputer
from scipy.sparse import issparse
from numba import cuda
import cudf

def kmeans_sampling(X, k, round_values=True, detailed=False):
    """
    Adapted from :
    https://github.com/slundberg/shap/blob/9411b68e8057a6c6f3621765b89b24d82bee13d4/shap/utils/_legacy.py
    Summarize a dataset (X) using weighted k-means.
    
    Parameters
    ----------
    X : cuDF DataFrame or cuda_array_interface compliant device array
        Data to be summarized, shape (n_samples, n_features)
    k : int
        Number of means to use for approximation.
    round_values : bool; default=True
        For all i, round the ith dimension of each mean sample to match the nearest value
        from X[:,i]. This ensures discrete features always get a valid value.
    detailed: bool; default=False
        To return details of group names and cluster labels of all data points

    Returns
    -------
    summary : Summary of the data, shape (k, n_features)
    group_names : Names of the features
    labels : Cluster labels of the data points in the original dataset, shape (n_samples, 1)
    """
    if not hasattr(X, "__cuda_array_interface__") and not \
            isinstance(X, cudf.DataFrame):
        raise TypeError("X needs to be either a cuDF DataFrame, Series or \
                    a cuda_array_interface compliant array.")

    if isinstance(X, cudf.DataFrame):
        group_names = X.columns
        X = X.values
        output_dtype = "DataFrame"
    elif isinstance(X, cudf.Series):
        group_names = X.name
        X = X.values.reshape(-1, 1)
        output_dtype = "Series"
    else:
        output_dtype, X = ["numba", cp.array(X)] if cuda.devicearray.is_cuda_ndarray(X) else ["cupy", X]
        try:
            # more than one column
            group_names = [str(i) for i in range(X.shape[1])]
        except IndexError as e:
            # one column
            X = X.reshape(-1, 1)
            group_names = ['0']

    # in case there are any missing values in data impute them
    imp = SimpleImputer(missing_values=cp.nan, strategy='mean')
    X = imp.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

    if round_values:
        for i in range(k):
            for j in range(X.shape[1]):
                xj = X[:, j].toarray().flatten() if issparse(
                    X) else X[:, j]  # sparse support courtesy of @PrimozGodec
                ind = cp.argmin(cp.abs(xj - kmeans.cluster_centers_[i, j]))
                kmeans.cluster_centers_[i, j] = X[ind, j]
    summary = kmeans.cluster_centers_
    labels = kmeans.labels_

    if output_dtype == "DataFrame":
        summary = cudf.DataFrame(summary)
    elif output_dtype == "Series":
        summary = cudf.Series(summary)
    elif output_dtype == "numba":
        summary = cuda.as_cuda_array(summary)
    if detailed:
        return summary , group_names, labels
    else:
        return summary
