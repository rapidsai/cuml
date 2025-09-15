# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cudf
import cupy as cp
import numpy as np

from cuml.common import input_to_cuml_array
from cuml.internals.array import CumlArray, array_to_memory_order
from cuml.internals.input_utils import (
    determine_array_type,
    determine_df_obj_type,
)
from cuml.internals.output_utils import output_to_df_obj_like


def _compute_stratify_split_indices(
    indices: cp.ndarray,
    stratify: CumlArray,
    n_train: int,
    n_test: int,
    random_state: cp.random.RandomState,
) -> Tuple[cp.ndarray]:
    """
    Compute the indices for stratified split based on stratify keys.
    Based on scikit-learn stratified split implementation.

    Parameters
    ----------
    indices: cupy array
        Indices used to shuffle input data
    stratify: CumlArray
        Keys used for stratification
    n_train: Number of samples in train set
    n_test: number of samples in test set
    random_state: cupy RandomState
        Random state used for shuffling stratify keys

    Returns
    -------
    train_indices, test_indices:
        Indices of inputs from which train and test sets are gathered
    """

    if indices.ndim != 1:
        raise ValueError(
            "Expected one one dimension for indices, but found array"
            "with shape = %d" % (indices.shape)
        )

    if stratify.ndim != 1:
        raise ValueError(
            "Expected one one dimension for stratify, but found array"
            "with shape = %d" % (stratify.shape)
        )

    # Converting to cupy array removes the need to add an if-else block
    # for startify column

    classes, stratify_indices = cp.unique(stratify, return_inverse=True)

    n_classes = classes.shape[0]
    class_counts = cp.bincount(stratify_indices)
    if cp.min(class_counts) < 2:
        raise ValueError(
            "The least populated class in y has only 1"
            " member, which is too few. The minimum"
            " number of groups for any class cannot"
            " be less than 2."
        )

    if n_train < n_classes:
        raise ValueError(
            "The train_size = %d should be greater or "
            "equal to the number of classes = %d" % (n_train, n_classes)
        )

    # List of length n_classes. Each element contains indices of that class.
    class_indices: List[cp.ndarray] = cp.split(
        cp.argsort(stratify_indices), cp.cumsum(class_counts)[:-1].tolist()
    )

    # Break ties
    n_i = _approximate_mode(class_counts, n_train, random_state)
    class_counts_remaining = class_counts - n_i
    t_i = _approximate_mode(class_counts_remaining, n_test, random_state)

    train_indices_partials = []
    test_indices_partials = []
    for i in range(n_classes):
        permutation = random_state.permutation(class_counts[i].item())
        perm_indices_class_i = class_indices[i].take(permutation)

        train_indices_partials.append(perm_indices_class_i[: n_i[i]])
        test_indices_partials.append(
            perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]]
        )

    train_indices = cp.concatenate(train_indices_partials, axis=0)
    test_indices = cp.concatenate(test_indices_partials, axis=0)

    return indices[train_indices], indices[test_indices]


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
    # this computes a bad approximation to the mode of the
    # multivariate hypergeometric given by class_counts and n_draws
    continuous = n_draws * class_counts / class_counts.sum()
    # floored means we don't overshoot n_samples, but probably undershoot
    floored = cp.floor(continuous)
    # we add samples according to how much "left over" probability
    # they had, until we arrive at n_samples
    need_to_add = int(n_draws - floored.sum())
    if need_to_add > 0:
        remainder = continuous - floored
        values = cp.sort(cp.unique(remainder))[::-1]
        # add according to remainder, but break ties
        # randomly to avoid biases
        for value in values:
            (inds,) = cp.where(remainder == value)
            # if we need_to_add less than what's in inds
            # we draw randomly from them.
            # if we need to add more, we add them all and
            # go to the next value
            add_now = min(len(inds), need_to_add)
            inds = rng.choice(inds, size=add_now, replace=False)
            floored[inds] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break
    return floored.astype(int)


def train_test_split(
    X,
    y=None,
    test_size: Optional[Union[float, int]] = None,
    train_size: Optional[Union[float, int]] = None,
    shuffle: bool = True,
    random_state: Optional[
        Union[int, cp.random.RandomState, np.random.RandomState]
    ] = None,
    stratify=None,
):
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

    stratify: cudf.Series or cuda_array_interface compliant device array,
            optional parameter. When passed, the input is split using this
            as column to startify on. Default=None

    Examples
    --------
    .. code-block:: python

        >>> import cudf
        >>> from cuml.model_selection import train_test_split
        >>> # Generate some sample data
        >>> df = cudf.DataFrame({'x': range(10),
        ...                      'y': [0, 1] * 5})
        >>> print(f'Original data: {df.shape[0]} elements')
        Original data: 10 elements
        >>> # Suppose we want an 80/20 split
        >>> X_train, X_test, y_train, y_test = train_test_split(df, 'y',
        ...                                                     train_size=0.8)
        >>> print(f'X_train: {X_train.shape[0]} elements')
        X_train: 8 elements
        >>> print(f'X_test: {X_test.shape[0]} elements')
        X_test: 2 elements
        >>> print(f'y_train: {y_train.shape[0]} elements')
        y_train: 8 elements
        >>> print(f'y_test: {y_test.shape[0]} elements')
        y_test: 2 elements

        >>> # Alternatively, if our labels are stored separately
        >>> labels = df['y']
        >>> df = df.drop(['y'], axis=1)
        >>> # we can also do
        >>> X_train, X_test, y_train, y_test = train_test_split(df, labels,
        ...                                                     train_size=0.8)

    Returns
    -------

    X_train, X_test, y_train, y_test : cudf.DataFrame or array-like objects
        Partitioned dataframes if X and y were cuDF objects. If `y` was
        provided as a column name, the column was dropped from `X`.
        Partitioned numba device arrays if X and y were Numba device arrays.
        Partitioned CuPy arrays for any other input.

    """
    if isinstance(y, str):
        # Use the column with name `str` as y
        if isinstance(X, cudf.DataFrame):
            name = y
            y = X[name]
            X = X.drop(name, axis=1)
        else:
            raise TypeError(
                "X needs to be a cuDF Dataframe when y is a \
                             string"
            )

    all_numeric = True
    if isinstance(X, cudf.DataFrame):
        all_numeric = all(
            cudf.api.types.is_numeric_dtype(X[col]) for col in X.columns
        )

    if all_numeric:
        x_order = array_to_memory_order(X)
        X_arr, X_row, *_ = input_to_cuml_array(X, order=x_order)
    else:
        x_order = "F"
        X_arr, X_row = X, X.shape[0]
    if y is not None:
        y_order = array_to_memory_order(y)
        y_arr, y_row, *_ = input_to_cuml_array(y, order=y_order)
        if X_row != y_row:
            raise ValueError(
                "X and y must have the same first dimension"
                f"(found {X_row} and {y_row})"
            )

    if isinstance(train_size, float):
        if not 0 <= train_size <= 1:
            raise ValueError(
                "proportion train_size should be between"
                f"0 and 1 (found {train_size})"
            )

    if isinstance(train_size, int):
        if not 0 <= train_size <= X_row:
            raise ValueError(
                "Number of instances train_size should be between 0 and the"
                f"first dimension of X (found {train_size})"
            )

    if isinstance(test_size, float):
        if not 0 <= test_size <= 1:
            raise ValueError(
                "proportion test_size should be between"
                f"0 and 1 (found {train_size})"
            )

    if isinstance(test_size, int):
        if not 0 <= test_size <= X_row:
            raise ValueError(
                "Number of instances test_size should be between 0 and the"
                f"first dimension of X (found {test_size})"
            )

    # Determining sizes of splits
    if isinstance(train_size, float):
        train_size = int(X_row * train_size)

    if test_size is None:
        if train_size is None:
            train_size = int(X_row * 0.75)

        test_size = X_row - train_size

    if isinstance(test_size, float):
        test_size = int(X_row * test_size)
        if train_size is None:
            train_size = X_row - test_size

    elif isinstance(test_size, int):
        if train_size is None:
            train_size = X_row - test_size

    # Compute training set and test set indices
    if shuffle:
        idxs = cp.arange(X_row)

        # Compute shuffle indices
        if random_state is None or isinstance(random_state, int):
            random_state = cp.random.RandomState(seed=random_state)

        elif isinstance(random_state, np.random.RandomState):
            random_state = cp.random.RandomState(
                seed=random_state.get_state()[1]
            )

        elif not isinstance(random_state, cp.random.RandomState):
            raise TypeError(
                "`random_state` must be an int, NumPy RandomState \
                             or CuPy RandomState."
            )

        random_state.shuffle(idxs)

        if stratify is not None:
            stratify, *_ = input_to_cuml_array(stratify)
            stratify = stratify[idxs]

            (train_indices, test_indices,) = _compute_stratify_split_indices(
                idxs,
                stratify,
                train_size,
                test_size,
                random_state,
            )

        else:
            train_indices = idxs[:train_size]
            test_indices = idxs[-1 * test_size :]
    else:
        train_indices = range(0, train_size)
        test_indices = range(-1 * test_size, 0)

    if all_numeric:
        # Gather from indices
        X_train = X_arr[train_indices]
        X_test = X_arr[test_indices]
        if y is not None:
            y_train = y_arr[train_indices]
            y_test = y_arr[test_indices]

        # Coerce output to original input type
        x_type = determine_df_obj_type(X) or determine_array_type(X)
        if y is not None:
            y_type = determine_df_obj_type(y) or determine_array_type(y)

        def _process_df_objs(
            df, df_type, df_train, df_test, train_indices, test_indices
        ):
            if df_type in {"series", "dataframe"}:
                df_train = output_to_df_obj_like(df_train, df, df_type)
                df_test = output_to_df_obj_like(df_test, df, df_type)

                if determine_array_type(df.index) == "pandas":
                    if isinstance(train_indices, cp.ndarray):
                        train_indices = train_indices.get()
                    if isinstance(test_indices, cp.ndarray):
                        test_indices = test_indices.get()

                df_train.index = df.index[train_indices]
                df_test.index = df.index[test_indices]
            else:
                df_train = df_train.to_output(df_type)
                df_test = df_test.to_output(df_type)
            return df_train, df_test

        X_train, X_test = _process_df_objs(
            X, x_type, X_train, X_test, train_indices, test_indices
        )
        if y is not None:
            y_train, y_test = _process_df_objs(
                y, y_type, y_train, y_test, train_indices, test_indices
            )

    else:
        X_train = X_arr.iloc[train_indices]
        X_test = X_arr.iloc[test_indices]
        if y is not None:
            y_train = y_arr[train_indices]
            y_test = y_arr[test_indices]

    if y is not None:
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test


class StratifiedKFold:
    """
    A cudf based implementation of Stratified K-Folds cross-validator.

    Provides train/test indices to split data into stratified K folds.
    The percentage of samples for each class are maintained in each
    fold.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : boolean, default=False
        Whether to shuffle each class's samples before splitting.
    random_state : int (default=None)
        Random seed

    Examples
    --------
    Splitting X,y into stratified K folds
    .. code-block:: python
        import cupy
        X = cupy.random.rand(12,10)
        y = cupy.arange(12)%4
        kf = StratifiedKFold(n_splits=3)
        for tr,te in kf.split(X,y):
            print(tr, te)
    Output:
    .. code-block:: python
        [ 4  5  6  7  8  9 10 11] [0 1 2 3]
        [ 0  1  2  3  8  9 10 11] [4 5 6 7]
        [0 1 2 3 4 5 6 7] [ 8  9 10 11]

    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2 or not isinstance(n_splits, int):
            raise ValueError(
                f"n_splits {n_splits} is not a integer at least 2"
            )

        if random_state is not None and not isinstance(random_state, int):
            raise ValueError(f"random_state {random_state} is not an integer")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = random_state

    def get_n_splits(self, X=None, y=None):
        return self.n_splits

    def split(self, x, y):
        if len(x) != len(y):
            raise ValueError("Expecting same length of x and y")
        y = input_to_cuml_array(y).array.to_output("cupy")
        if len(cp.unique(y)) < 2:
            raise ValueError("number of unique classes cannot be less than 2")
        df = cudf.DataFrame()
        ids = cp.arange(y.shape[0])

        if self.shuffle:
            cp.random.seed(self.seed)
            cp.random.shuffle(ids)
            y = y[ids]

        df["y"] = y
        df["ids"] = ids
        grpby = df.groupby(["y"])

        dg = grpby.agg({"y": "count"})
        col = dg.columns[0]
        msg = (
            f"n_splits={self.n_splits} cannot be greater "
            + "than the number of members in each class."
        )
        if self.n_splits > dg[col].min():
            raise ValueError(msg)

        got = grpby.apply(lambda df: df.assign(order=range(len(df))))
        got = got.sort_values("ids")

        for i in range(self.n_splits):
            mask = got["order"] % self.n_splits == i
            train = got.loc[~mask, "ids"].values
            test = got.loc[mask, "ids"].values
            if len(test) == 0:
                break
            yield train, test

    def _check_array_shape(self, y):
        if y is None:
            raise ValueError("Expecting 1D array, got None")
        elif hasattr(y, "shape") and len(y.shape) > 1 and y.shape[1] > 1:
            raise ValueError(f"Expecting 1D array, got {y.shape}")
        else:
            pass
