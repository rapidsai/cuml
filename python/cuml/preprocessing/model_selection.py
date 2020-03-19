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
import cupy as cp
import numpy as np
import warnings

from numba import cuda
from typing import Union

from cuml.utils import rmm_cupy_ary


def train_test_split(
    X,
    y,
    test_size: Union[float, int] = None,
    train_size: Union[float, int] = None,
    shuffle: bool = True,
    random_state: Union[int, cp.random.RandomState,
                        np.random.RandomState] = None,
    seed: Union[int, cp.random.RandomState, np.random.RandomState] = None
):
    """
    Partitions device data into four collated objects, mimicking
    Scikit-learn's `train_test_split`

    Parameters
    ----------
    X : cudf.DataFrame or cuda_array_interface compliant device array
        Data to split, has shape (n_samples, n_features)
    y : str, cudf.Series or cuda_array_interface compliant device array
        Set of labels for the data, either a series of shape (n_samples) or
        the string label of a column in X (if it is a cuDF DataFrame)
        containing the labels
    train_size : float or int, optional
        If float, represents the proportion [0, 1] of the data
        to be assigned to the training set. If an int, represents the number
        of instances to be assigned to the training set. Defaults to 0.8
    shuffle : bool, optional
        Whether or not to shuffle inputs before splitting
    random_state : int, CuPy RandomState or NumPy RandomState optional
        If shuffle is true, seeds the generator. Unseeded by default
    seed: random_state : int, CuPy RandomState or NumPy RandomState optional
        Deprecated in favor of `random_state`.
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
    if isinstance(y, str):
        # Use the column with name `str` as y
        if isinstance(X, cudf.DataFrame):
            name = y
            y = X[name]
            X = X.drop(name)
        else:
            raise TypeError("X needs to be a cuDF Dataframe when y is a \
                             string")

    # todo: this check will be replaced with upcoming improvements
    # to input_utils with PR #1379
    if not cuda.is_cuda_array(X) and not isinstance(X, cudf.DataFrame) \
            and isinstance(y, cudf.Series):
        raise TypeError("X needs to be either a cuDF DataFrame, Series or \
                        a cuda_array_interface compliant array.")

    if not cuda.is_cuda_array(y) and not isinstance(y, cudf.DataFrame) \
            and isinstance(y, cudf.Series):
        raise TypeError("y needs to be either a cuDF DataFrame, Series or \
                        a cuda_array_interface compliant array.")

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

    if isinstance(train_size, int):
        if not 0 <= train_size <= X.shape[0]:
            raise ValueError(
                "Number of instances train_size should be between 0 and the"
                "first dimension of X (found {})".format(train_size)
            )

    if isinstance(test_size, float):
        if not 0 <= test_size <= 1:
            raise ValueError(
                "proportion test_size should be between"
                "0 and 1 (found {})".format(train_size)
            )

    if isinstance(test_size, int):
        if not 0 <= test_size <= X.shape[0]:
            raise ValueError(
                "Number of instances test_size should be between 0 and the"
                "first dimension of X (found {})".format(test_size)
            )

    x_numba = False
    y_numba = False

    if seed is not None:
        if random_state is None:
            warnings.warn("Parameter 'seed' is deprecated, please use \
                          'random_state' instead.")
            random_state = seed
        else:
            warnings.warn("Both 'seed' and 'random_state' parameters were \
                          set, using 'random_state' since 'seed' is \
                          deprecated. ")

    if shuffle:
        if random_state is None or isinstance(random_state, int):
            idxs = rmm_cupy_ary(cp.arange, X.shape[0])
            random_state = cp.random.RandomState(seed=random_state)

        elif isinstance(random_state, cp.random.RandomState):
            idxs = rmm_cupy_ary(cp.arange, X.shape[0])

        elif isinstance(random_state, np.random.RandomState):
            idxs = np.arange(X.shape[0])

        else:
            raise TypeError("`random_state` must be an int, NumPy RandomState \
                             or CuPy RandomState.")

        random_state.shuffle(idxs)

        if isinstance(X, cudf.DataFrame) or isinstance(X, cudf.Series):
            X = X.iloc[idxs].reset_index(drop=True)

        elif cuda.is_cuda_array(X):
            # numba (and therefore rmm device_array) does not support
            # fancy indexing
            if cuda.devicearray.is_cuda_ndarray(X):
                x_numba = True
            X = cp.asarray(X)[idxs]

        if isinstance(y, cudf.DataFrame) or isinstance(y, cudf.Series):
            y = y.iloc[idxs]

        elif cuda.is_cuda_array(y):
            if cuda.devicearray.is_cuda_ndarray(y):
                y_numba = True
            y = cp.asarray(y)[idxs]

    # Determining sizes of splits
    if isinstance(train_size, float):
        train_size = int(X.shape[0] * train_size)

    if test_size is None:
        if train_size is None:
            train_size = int(X.shape[0] * 0.75)

        test_size = X.shape[0] - train_size

    if isinstance(test_size, float):
        test_size = int(X.shape[0] * test_size)
        if train_size is None:
            train_size = X.shape[0] - test_size

    elif isinstance(test_size, int):
        if train_size is None:
            train_size = X.shape[0] - test_size

    if cuda.is_cuda_array(X) or isinstance(X, cp.sparse.csr_matrix):
        X_train = X[0:train_size]
        y_train = y[0:train_size]
    elif isinstance(X, cudf.DataFrame):
        X_train = X.iloc[0:train_size]
        y_train = y.iloc[0:train_size]

    if cuda.is_cuda_array(y) or isinstance(X, cp.sparse.csr_matrix):
        X_test = X[-1 * test_size:]
        y_test = y[-1 * test_size:]
    elif isinstance(y, cudf.DataFrame):
        X_test = X.iloc[-1 * test_size:]
        y_test = y.iloc[-1 * test_size:]

    if x_numba:
        X_train = cuda.as_cuda_array(X_train)
        X_test = cuda.as_cuda_array(X_test)

    if y_numba:
        y_train = cuda.as_cuda_array(y_train)
        y_test = cuda.as_cuda_array(y_test)

    return X_train, X_test, y_train, y_test
