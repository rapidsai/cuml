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
import cupy as cp
import numpy as np


def train_test_split(
    X, y,
    train_size: Union[float, int] = 0.8,
    shuffle: bool = True,
    seed: int = None,
) -> Tuple[cudf.DataFrame, cudf.DataFrame, cudf.DataFrame, cudf.DataFrame]:
    """
    Partitions cuDF data into four collated dataframes, mimicking
    Scikit-learn's `train_test_split`

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
        print(f'Original data: {df.shape[0]} elements')

        # Suppose we want an 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(df, 'y',
                                                            train_size=0.8)
        print(f'X_train: {X_train.shape[0]} elements')
        print(f'X_test: {X_test.shape[0]} elements')
        print(f'y_train: {y_train.shape[0]} elements')
        print(f'y_test: {y_test.shape[0]} elements')

        # Alternatively, if our labels are stored separately
        labels = df['y']
        df = df.drop(['y'])

        # we can also do
        X_train, X_test, y_train, y_test = train_test_split(df, labels,
                                                            train_size=0.8)

    Output:

    .. code-block:: python

        Original data: 10 elements
        X_train: 8 elements
        X_test: 2 elements
        y_train: 8 elements
        y_test: 2 elements

    Returns
    -------
    X_train, X_test, y_train, y_test : cudf.DataFrame
        Partitioned dataframes. If `y` was provided as a column name, the
        column was dropped from the `X`s
    """
    # TODO Use cupy indexing to support non cudf input types for X, y
    if isinstance(y, str):
        # Use the column with name `str` as y
        if isinstance(X, cudf.DataFrame):
            name = y
            y = X[name]
            X = X.drop(name)
        else:
            raise ValueError("X needs to be a cuDF Dataframe when y is a \
                             string")

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

    if shuffle:
        if isinstance(seed, cp.random.RandomState):
            idxs = cp.arange(X.shape[0])

        elif isinstance(seed, np.random.RandomState):
            idxs = np.arange(X.shape[0])

        elif isinstance(seed, int):
            idxs = cp.arange(X.shape[0])
            seed = cp.random.RandomState(seed=seed)

        seed.shuffle(idxs)
        X = X.iloc[idxs].reset_index(drop=True)
        y = y.iloc[idxs].reset_index(drop=True)

    split_idx = int(X.shape[0] * train_size)
    X_train = X.iloc[0:split_idx]
    y_train = y.iloc[0:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test
