# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
from cuml.preprocessing import SimpleImputer
from cuml.internals.input_utils import (
    determine_array_type,
    get_supported_input_type,
)
from cuml import KMeans
import cuml
from cuml.internals.safe_imports import cpu_only_import_from
from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")
issparse = cpu_only_import_from("scipy.sparse", "issparse")


@cuml.internals.api_return_generic()
def kmeans_sampling(X, k, round_values=True, detailed=False, random_state=0):
    """
    Adapted from :
    https://github.com/slundberg/shap/blob/9411b68e8057a6c6f3621765b89b24d82bee13d4/shap/utils/_legacy.py
    Summarize a dataset (X) using weighted k-means.

    Parameters
    ----------
    X : cuDF or Pandas DataFrame/Series, numpy arrays or cuda_array_interface
        compliant device array.
        Data to be summarized, shape (n_samples, n_features)
    k : int
        Number of means to use for approximation.
    round_values : bool; default=True
        For all i, round the ith dimension of each mean sample to match the
        nearest value from X[:,i]. This ensures discrete features always get
        a valid value.
    detailed: bool; default=False
        To return details of group names and cluster labels of all data points
    random_state: int; default=0
        Sets the random state.

    Returns
    -------
    summary : Summary of the data, shape (k, n_features)
    group_names : Names of the features
    labels : Cluster labels of the data points in the original dataset,
             shape (n_samples, 1)
    """
    output_dtype = get_supported_input_type(X)
    _output_dtype_str = determine_array_type(X)
    cuml.internals.set_api_output_type(_output_dtype_str)

    if output_dtype is None:
        raise TypeError(
            f"Type of input {type(X)} is not supported. Supported \
                        dtypes: cuDF DataFrame, cuDF Series, cupy, numba,\
                        numpy, pandas DataFrame, pandas Series"
        )

    if "DataFrame" in str(output_dtype):
        group_names = X.columns
        X = cp.array(X.values, copy=False)
    if "Series" in str(output_dtype):
        group_names = X.name
        X = cp.array(X.values.reshape(-1, 1), copy=False)
    else:
        # it's either numpy, cupy or numba
        X = cp.array(X, copy=False)
        try:
            # more than one column
            group_names = [str(i) for i in range(X.shape[1])]
        except IndexError:
            # one column
            X = X.reshape(-1, 1)
            group_names = ["0"]

    # in case there are any missing values in data impute them
    imp = SimpleImputer(
        missing_values=cp.nan, strategy="mean", output_type=_output_dtype_str
    )
    X = imp.fit_transform(X)

    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        output_type=_output_dtype_str,
        n_init="auto",
    ).fit(X)

    if round_values:
        for i in range(k):
            for j in range(X.shape[1]):
                xj = (
                    X[:, j].toarray().flatten() if issparse(X) else X[:, j]
                )  # sparse support courtesy of @PrimozGodec
                ind = cp.argmin(cp.abs(xj - kmeans.cluster_centers_[i, j]))
                kmeans.cluster_centers_[i, j] = X[ind, j]
    summary = kmeans.cluster_centers_
    labels = kmeans.labels_

    if detailed:
        return summary, group_names, labels
    else:
        return summary
