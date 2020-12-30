
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
import cupyx
import numpy as np
import warnings

from cuml.common.memory_utils import _strides_to_order
from cuml.model_selection import \
        _stratify_split as _new_stratify_split, \
        _approximate_mode as _new_approximate_mode, \
        train_test_split as new_train_test_split
from numba import cuda
from typing import Union


def _stratify_split(X, y, n_train, n_test, x_numba, y_numba, random_state):
    """
    Function to perform a stratified split based on y lables.
    Based on scikit-learn stratified split implementation.

    Parameters
    ----------
    X, y: Shuffled input data and labels
    n_train: Number of samples in train set
    n_test: number of samples in test set
    x_numba: Determines whether the data should be converted to numba
    y_numba: Determines whether the labales should be converted to numba

    Returns
    -------
    X_train, X_test: Data X divided into train and test sets
    y_train, y_test: Labels divided into train and test sets
    """
    warnings.warn("Importing from cuml.preprocessing.model_selection is "
                  "deprecated and will be removed in 0.18. Instead, please "
                  "import from cuml.model_selection",
                  DeprecationWarning)
    return _new_stratify_split(X, y, n_train, n_test, x_numba, y_numba,
                               random_state)

def _approximate_mode(class_counts, n_draws, rng):
    """
    CuPy implementataiton based on scikit-learn approximate_mode method.
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/__init__.py#L984

    It is the mostly likely outcome of drawing n_draws many
    samples from the population given by class_counts.

    Parameters
    ----------
    class_counts : ndarray of int
        Population per class.
    n_draws : int
        Number of draws (samples to draw) from the overall population.
    rng : random state
        Used to break ties.

    Returns
    -------
    sampled_classes : cupy array of int
        Number of samples drawn from each class.
        np.sum(sampled_classes) == n_draws
    """
    warnings.warn("Importing from cuml.preprocessing.model_selection is "
                  "deprecated and will be removed in 0.18. Instead, please "
                  "import from cuml.model_selection",
                  DeprecationWarning)
    return _new_approximate_mode(class_counts, n_draws, rng)

def train_test_split(X,
                     y=None,
                     test_size: Union[float,
                                      int] = None,
                     train_size: Union[float,
                                       int] = None,
                     shuffle: bool = True,
                     random_state: Union[int,
                                         cp.random.RandomState,
                                         np.random.RandomState] = None,
                     seed: Union[int,
                                 cp.random.RandomState,
                                 np.random.RandomState] = None,
                     stratify=None):
    """
    Partitions device data into four collated objects, mimicking
    Scikit-learn's `train_test_split
    <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>`_.

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
        If shuffle is true, seeds the generator. Unseeded by default

        .. deprecated:: 0.11
           Parameter `seed` is deprecated and will be removed in 0.17. Please
           use `random_state` instead

    stratify: bool, optional
        Whether to stratify the input data based on class labels.
        None by default

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
        df = df.drop(['y'], axis=1)

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
        provided as a column name, the column was dropped from `X`.
        Partitioned numba device arrays if X and y were Numba device arrays.
        Partitioned CuPy arrays for any other input.

    """
    warnings.warn("Importing from cuml.preprocessing.model_selection is "
                  "deprecated and will be removed in 0.18. Instead, please "
                  "import from cuml.model_selection",
                  DeprecationWarning)
    return new_train_test_split(X, y, test_size, train_size, shuffle,
                                random_state, seed, stratify)
