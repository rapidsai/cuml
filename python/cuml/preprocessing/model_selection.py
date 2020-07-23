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

from cuml.common.memory_utils import _strides_to_order
from cuml.common.memory_utils import rmm_cupy_ary
from numba import cuda
from typing import Union


def slice_data(X, y, train_size, test_size, x_numba, y_numba):
    if hasattr(X, "__cuda_array_interface__"):
        x_order = _strides_to_order(X.__cuda_array_interface__['strides'],
                                    cp.dtype(X.dtype))

    if hasattr(y, "__cuda_array_interface__"):
        y_order = _strides_to_order(y.__cuda_array_interface__['strides'],
                                    cp.dtype(y.dtype))

    if hasattr(X, "__cuda_array_interface__") or \
            isinstance(X, cp.sparse.csr_matrix):
        X_train = cp.array(X[0:train_size], order=x_order)
        if y is not None:
            y_train = cp.array(y[0:train_size], order=y_order)
    elif isinstance(X, cudf.DataFrame):
        X_train = X.iloc[0:train_size]
        if y is not None:
            y_train = y.iloc[0:train_size]

    if hasattr(X, "__cuda_array_interface__") or \
            isinstance(X, cp.sparse.csr_matrix):
        X_test = cp.array(X[-1 * test_size:], order=x_order)
        if y is not None:
            y_test = cp.array(y[-1 * test_size:], order=y_order)
    elif isinstance(X, cudf.DataFrame):
        X_test = X.iloc[-1 * test_size:]
        if y is not None:
            y_test = y.iloc[-1 * test_size:]
    if x_numba:
        X_train = cuda.as_cuda_array(X_train)
        X_test = cuda.as_cuda_array(X_test)

    if y_numba:
        y_train = cuda.as_cuda_array(y_train)
        y_test = cuda.as_cuda_array(y_test)

    if y is not None:
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test


def stratify_split(X, y, n_train, n_test, x_numba, y_numba):
    """
    Function to perform a stratified split based on y.
    Identifies number of classes and samples per class, splices data within
    each class.

    Input:
    X, y: shuffled input data and labels
    """
    classes, y_indices = cp.unique(y, return_inverse=True)
    n_classes = classes.shape[0]
    class_counts = cp.bincount(y_indices)
    if n_train < n_classes:
        raise ValueError('The train_size = %d should be greater or '
                         'equal to the number of classes = %d' %
                         (n_train, n_classes))
    if n_test < n_classes:
        raise ValueError('The test_size = %d should be greater or '
                         'equal to the number of classes = %d' %
                         (n_test, n_classes))
    class_indices = cp.split(cp.argsort(y_indices),
                             cp.cumsum(class_counts)[:-1].item())

    X_train = None
    n_per_class = int(n_train / n_classes)

    for i in range(n_classes):
        class_idxs = class_indices[i]
        if hasattr(X, "__cuda_array_interface__") or \
           isinstance(X, cp.sparse.csr_matrix):
            X_i = X[class_idxs]
            y_i = y[class_idxs]
        else:
            X_i = X.iloc[class_idxs]
            y_i = y.iloc[class_idxs]

        train_size = n_per_class
        test_size = len(class_idxs) - train_size

        X_train_i, X_test_i, y_train_i, y_test_i = slice_data(X_i, y_i,
                                                              train_size,
                                                              test_size,
                                                              x_numba,
                                                              y_numba)

        if X_train is None:
            X_train = X_train_i.copy()
            y_train = y_train_i.copy()
            X_test = X_test_i.copy()
            y_test = y_test_i.copy()
        else:
            if hasattr(X, "__cuda_array_interface__") or \
               isinstance(X, cp.sparse.csr_matrix):
                X_train = cp.concatenate([X_train, X_train_i], axis=0)
                X_test = cp.concatenate([X_test, X_test_i], axis=0)
                y_train = cp.concatenate([y_train, y_train_i], axis=0)
                y_test = cp.concatenate([y_test, y_test_i], axis=0)
            else:
                X_train = X_train.append(X_train_i)
                X_test = X_test.append(X_test_i)
                y_train = y_train.append(y_train_i)
                y_test = y_test.append(y_test_i)
    return X_train, X_test, y_train, y_test


def train_test_split(
    X,
    y=None,
    test_size: Union[float, int] = None,
    train_size: Union[float, int] = None,
    shuffle: bool = True,
    random_state: Union[int, cp.random.RandomState,
                        np.random.RandomState] = None,
    seed: Union[int, cp.random.RandomState, np.random.RandomState] = None,
    stratify=None
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
    X_train, X_test, y_train, y_test : cudf.DataFrame or array-like objects
        Partitioned dataframes if X and y were cuDF objects. If `y` was
        provided as a column name, the column was dropped from the `X`s
        Partitioned numba device arrays if X and y were Numba device arrays.
        Partitioned CuPy arrays for any other input.
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
    # to input_utils
    #
    if y is not None:
        if not hasattr(X, "__cuda_array_interface__") and not \
                isinstance(X, cudf.DataFrame):
            raise TypeError("X needs to be either a cuDF DataFrame, Series or \
                            a cuda_array_interface compliant array.")

        if not hasattr(y, "__cuda_array_interface__") and not \
                isinstance(y, cudf.DataFrame):
            raise TypeError("y needs to be either a cuDF DataFrame, Series or \
                            a cuda_array_interface compliant array.")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must have the same first dimension"
                "(found {} and {})".format(X.shape[0], y.shape[0])
            )
    else:
        if not hasattr(X, "__cuda_array_interface__") and not \
                isinstance(X, cudf.DataFrame):
            raise TypeError("X needs to be either a cuDF DataFrame, Series or \
                            a cuda_array_interface compliant object.")

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

    x_numba = cuda.devicearray.is_cuda_ndarray(X)
    y_numba = cuda.devicearray.is_cuda_ndarray(y)

    if seed is not None:
        if random_state is None:
            warnings.warn("Parameter 'seed' is deprecated, please use \
                          'random_state' instead.")
            random_state = seed
        else:
            warnings.warn("Both 'seed' and 'random_state' parameters were \
                          set, using 'random_state' since 'seed' is \
                          deprecated. ")

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

    if shuffle:
        # Shuffle the data
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

        elif hasattr(X, "__cuda_array_interface__"):
            # numba (and therefore rmm device_array) does not support
            # fancy indexing
            X = cp.asarray(X)[idxs]

        if isinstance(y, cudf.DataFrame) or isinstance(y, cudf.Series):
            y = y.iloc[idxs]

        elif hasattr(y, "__cuda_array_interface__"):
            y = cp.asarray(y)[idxs]

        if stratify is not None:
            split_return = stratify_split(X, y, train_size, test_size,
                                          x_numba, y_numba)
            return split_return
    split_return = slice_data(X, y, train_size, test_size, x_numba, y_numba)
    return split_return
