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
    # TODO this is the bottleneck and should be a gpu operation,
    # when possible replace with the mlprim mentioned in cuml #659
    return np.random.permutation(len(df))


def train_test_split(
    X: cudf.DataFrame,
    y: Union[str, cudf.Series],
    train_size: Union[float, int] = 0.8,
    shuffle: bool = True,
    seed: int = None,
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
    train_size : float or int, optional
        If float, represents the proportion [0, 1] of the data
        to be assigned to the training set. If an int, represents the number
        of instances to be assigned to the training set. Defaults to 0.8
    shuffle : bool, optional
        Whether or not to shuffle inputs before splitting
    seed : int, optional
        If shuffle is true, seeds the generator. Unseeded by default

    Examples
    --------

    .. code-block:: python

        import cudf
        from cuml.preprocessing.model_selection import train_test_split

        # Generate some sample data
        df = cudf.DataFrame({'x': range(10),
                             'y': [0, 1] * 5})
        print(df)

        # Suppose we want an 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(df, 'y',
                                                            train_size=0.8)
        print("X_train:")
        print(X_train)
        print("X_test:")
        print(X_test)
        print("y_train:")
        print(y_train)
        print("y_test:")
        print(y_test)

        # Alternatively, if our labels are stored separately
        labels = df['y']
        df = df.drop(['y'])

        # we can also do
        X_train, X_test, y_train, y_test = train_test_split(df, labels,
                                                            train_size=0.8)

    Output:

    .. code-block:: python

           x  y
        0  0  0
        1  1  1
        2  2  0
        3  3  1
        4  4  0
        5  5  1
        6  6  0
        7  7  1
        8  8  0
        9  9  1
        X_train:
        x
        0  3
        1  9
        2  2
        3  0
        4  7
        5  1
        6  8
        7  4
        X_test:
        x
        8  5
        9  6
        y_train:
        0    1
        1    1
        2    0
        3    0
        4    1
        5    1
        6    0
        7    0
        Name: y, dtype: int64
        y_test:
        8    1
        9    0
        Name: y, dtype: int64


    Returns
    -------
    X_train, X_test, y_train, y_test : cudf.DataFrame
        Partitioned dataframes. If `y` was provided as a column name, the
        column was dropped from the `X`s
    """
    # TODO Use cupy indexing to support non cudf input types for X, y
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

    if isinstance(train_size, float):
        if not 0 <= train_size <= 1:
            raise ValueError(
                "proportion train_size should be between"
                "0 and 1 (found {})".format(train_size)
            )
        split_idx = int(X.shape[0] * train_size)

    if isinstance(train_size, int):
        if not 0 <= train_size <= X.shape[0]:
            raise ValueError(
                "Number of instances train_size should be between 0 and the"
                "first dimension of X (found {})".format(train_size)
            )
        split_idx = train_size

    if seed is not None:
        np.random.seed(seed)

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
