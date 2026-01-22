# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import cudf
import cupy as cp
import numpy as np
from numba import cuda as numba_cuda
from sklearn.model_selection import (
    train_test_split as sklearn_train_test_split,
)

from cuml.common import input_to_cuml_array
from cuml.internals.input_utils import input_to_host_array
from cuml.internals.utils import check_random_seed


def train_test_split(
    *arrays,
    y="deprecated",
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):
    """
    Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are cudf DataFrames/Series, cupy arrays, numba device
        arrays, numpy arrays, pandas DataFrames/Series, or any array-like
        objects with a shape attribute.

    y : str, default="deprecated"
        The name of the column that contains the target variable.

        .. deprecated:: 26.02
            The ``y`` parameter is deprecated and will be removed in 26.04.
            Extract the column manually:
            ``X, y = df.drop('col', axis=1), df['col']``

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If train_size is also None, it will
        be set to 0.25.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.

    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs. Output types match
        input types (cudf inputs return cudf outputs, cupy inputs return
        cupy outputs, etc.)

    Examples
    --------
    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.model_selection import train_test_split
        >>> X = cp.arange(10).reshape((5, 2))
        >>> y = cp.array([0, 0, 1, 1, 1])
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.2, random_state=42
        ... )

    Notes
    -----
    .. versionchanged:: 26.02
        Output types now consistently match input types. Previously, pandas
        inputs were converted to cudf outputs. Now pandas inputs return pandas
        outputs, cudf inputs return cudf outputs.

    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    # Handle deprecated y parameter usage
    # Case 1: y passed as keyword: train_test_split(df, y=...)
    # Case 2: column name passed as second positional arg: train_test_split(df, "col")
    y_is_column_name_positional = len(arrays) == 2 and isinstance(
        arrays[1], str
    )
    # Use isinstance check to avoid ambiguous truth value with array-like y
    y_was_passed = not (isinstance(y, str) and y == "deprecated")

    if y_was_passed or y_is_column_name_positional:
        warnings.warn(
            "The explicit 'y' parameter is deprecated and will be "
            "removed in 26.04. Extract the column manually: "
            "X, y = df.drop('col', axis=1), df['col']",
            FutureWarning,
            stacklevel=2,
        )

        if y_is_column_name_positional:
            # User passed: train_test_split(df, "colname")
            X = arrays[0]
            col_name = arrays[1]
            X, y = X.drop(col_name, axis=1), X[col_name]
            arrays = (X, y)
        elif isinstance(y, str):
            # User passed: train_test_split(df, y="colname")
            X = arrays[0]
            X, y = X.drop(y, axis=1), X[y]
            arrays = (X, y)
        else:
            # User passed: train_test_split(X, y=array)
            if len(arrays) > 1:
                raise ValueError(
                    "Cannot use deprecated 'y' parameter with multiple "
                    "positional arrays. Pass all arrays as positional "
                    "arguments instead: train_test_split(X, y, ...)"
                )
            arrays = (arrays[0], y)

    # Validate arrays have consistent first dimension
    n_samples = arrays[0].shape[0]
    for i, arr in enumerate(arrays[1:], 1):
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"Found input variables with inconsistent numbers of samples: "
                f"{[a.shape[0] for a in arrays]}"
            )

    # Normalize random_state for sklearn (extract seed from RandomState objects)
    sklearn_random_state = random_state
    if isinstance(random_state, cp.random.RandomState):
        # CuPy RandomState - extract underlying state as int
        sklearn_random_state = int(check_random_seed(random_state))
    elif isinstance(random_state, np.random.RandomState):
        sklearn_random_state = int(check_random_seed(random_state))

    # Convert stratify to numpy, other arrays types are not supported by scikit-learn
    if stratify is not None:
        stratify = input_to_host_array(stratify).array

    # Check for numba device arrays, scikit-learn can't handle them directly
    original_types = []
    converted_arrays = []
    for arr in arrays:
        if numba_cuda.devicearray.is_cuda_ndarray(arr):
            original_types.append("numba")
            converted_arrays.append(cp.asarray(arr))
        else:
            original_types.append(None)
            converted_arrays.append(arr)

    results = sklearn_train_test_split(
        *converted_arrays,
        test_size=test_size,
        train_size=train_size,
        random_state=sklearn_random_state,
        shuffle=shuffle,
        stratify=stratify,
    )

    # Convert numba arrays back to numba device arrays
    # There are two results for each original array.
    final_results = []
    for i, orig_type in enumerate(original_types):
        for n in (0, 1):
            result = results[i * 2 + n]
            if orig_type == "numba":
                final_results.append(numba_cuda.as_cuda_array(result))
            else:
                final_results.append(result)

    return final_results


class _KFoldBase(ABC):
    """Base class for k-fold split."""

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        if n_splits < 2 or not isinstance(n_splits, int):
            raise ValueError(
                f"n_splits {n_splits} is not a integer at least 2"
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = random_state

    @abstractmethod
    def split(self, X, y):
        raise NotImplementedError()

    @abstractmethod
    def get_n_splits(self, X=None, y=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        raise NotImplementedError()


class KFold(_KFoldBase):
    """K-Folds cross-validator.

    Provides train/test indices to split data in train/test sets. Split dataset into k
    consecutive folds (without shuffling by default).

    Each fold is then used once as a validation set while the k - 1 remaining folds form
    the training set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=False
        Whether to shuffle the samples before splitting. Note that the samples within
        each split will not be shuffled.

    random_state : int, CuPy RandomState, NumPy RandomState, or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.

    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(self, X, y=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        Yields
        ------
        train : CuPy ndarray
            The training set indices for that split.

        test : CuPy ndarray
            The testing set indices for that split.
        """
        n_samples = X.shape[0]
        if y is not None and n_samples != len(y):
            raise ValueError("Expecting same length of x and y")
        if n_samples < self.n_splits:
            raise ValueError(
                f"n_splits: {self.n_splits} must be smaller than the number of samples: {n_samples}."
            )

        indices = cp.arange(n_samples)

        if self.shuffle:
            cp.random.RandomState(check_random_seed(self.seed)).shuffle(
                indices
            )

        fold_sizes = cp.full(
            self.n_splits, n_samples // self.n_splits, dtype=cp.int64
        )
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test = indices[start:stop]

            mask = cp.zeros(n_samples, dtype=cp.bool_)
            mask[start:stop] = True

            train = indices[cp.logical_not(mask)]
            yield train, test
            current = stop

    def get_n_splits(self, X=None, y=None):
        return self.n_splits


class StratifiedKFold(_KFoldBase):
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
        super().__init__(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

        if random_state is not None and not isinstance(random_state, int):
            raise ValueError(f"random_state {random_state} is not an integer")

    def get_n_splits(self, X=None, y=None):
        return self.n_splits

    def split(self, X, y):
        if len(X) != len(y):
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
