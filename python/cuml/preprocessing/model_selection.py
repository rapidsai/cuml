# Copyright (c) 2019, NVIDIA CORPORATION.
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

import cudf
from typing import Union, Tuple
from numba import jit
import numpy as np


@jit
def _shuffle_idx(df: cudf.DataFrame) -> np.ndarray:
    """ Return indices which represent a randomly shuffled version of `df`
    """
    return np.random.permutation(len(df))


def train_test_split(
    X: cudf.DataFrame,
    y: Union[str, cudf.Series],
    train_size: float = 0.8,
    shuffle: bool = True,
) -> Tuple[cudf.DataFrame, cudf.DataFrame, cudf.DataFrame, cudf.DataFrame]:
    """
    Partitions the data into four collated dataframes, mimicing sklearn's
    `train_test_split`

    Parameters
    ----------
    X : cudf.DataFrame
        Data to split, has shape (n_samples, n_features)
    y : str or cudf.Series
        Set of labels for the data, either a series of shape (n_samples) or
        the string label of a column in X containing the labels
    train_size : float, optional
        Proportion [0, 1] of the data to be assigned to the training set
    shuffle : bool, optional
        Whether or not to shuffle inputs before splitting

    Returns
    -------
    X_train, X_test, y_train, y_test : cudf.DataFrame
        Partitioned dataframes. If `y` was provided as a column name, the
        column was dropped from the `X`s
    """
    if not 0 <= train_size <= 1:
        raise ValueError(
            "train_size should be between 0 and 1 (found {})".format(
                train_size
            )
        )

    if isinstance(y, str):
        # Use the column with name `str` as y
        name = y
        y = X[name]
        X = X.drop(name)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y must have the same first dimension"
            "(found {} and {})".format(X.shape[0], y.shape[0])
        )

    if shuffle:
        idxs = _shuffle_idx(X)
        X = X.iloc[idxs].reset_index(drop=True)
        y = y.iloc[idxs].reset_index(drop=True)

    split_idx = int(X.shape[0] * train_size)
    X_train = X.iloc[0:split_idx]
    y_train = y.iloc[0:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test
