# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cudf
import cupy as cp
import pandas as pd

from cuml import KMeans
from cuml.internals.outputs import ArrayIndexPair, mlfunc, using_output_type
from cuml.internals.validation import check_array
from cuml.preprocessing import SimpleImputer


@mlfunc
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
    if isinstance(X, (cudf.DataFrame, pd.DataFrame)):
        group_names = [str(c) for c in X.columns]
    elif isinstance(X, (cudf.Series, pd.Series)):
        group_names = [str(X.name)]
    elif len(X.shape) == 2:
        group_names = [str(i) for i in range(X.shape[1])]
    else:
        group_names = ["0"]

    X, index = check_array(
        X, ensure_2d=False, ensure_all_finite=False, return_index=True
    )
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    with using_output_type("cupy"):
        # in case there are any missing values in data impute them
        imp = SimpleImputer(missing_values=cp.nan, strategy="mean")
        X = imp.fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        kmeans.fit(X)
        summary = kmeans.cluster_centers_

        if round_values:
            for i in range(k):
                for j in range(X.shape[1]):
                    xj = X[:, j]
                    ind = cp.argmin(cp.abs(xj - summary[i, j]))
                    summary[i, j] = X[ind, j]

        if detailed:
            return (
                summary,
                group_names,
                ArrayIndexPair(kmeans.labels_, index=index),
            )
        else:
            return summary
