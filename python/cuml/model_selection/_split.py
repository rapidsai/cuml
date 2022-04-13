# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from abc import ABCMeta, abstractmethod
from inspect import signature
import numbers
import warnings
import cudf
import cupy as cp
import cupyx
import numpy as np

from cuml.common.memory_utils import _strides_to_order
from numba import cuda
from typing import Union

from python.cuml._thirdparty.sklearn.utils.validation import (
    _num_samples,
    check_random_state,
    indexable,
)
from python.cuml.thirdparty_adapters.adapters import check_array


def _stratify_split(
    X, stratify, labels, n_train, n_test, x_numba, y_numba, random_state
):
    """
    Function to perform a stratified split based on stratify column.
    Based on scikit-learn stratified split implementation.

    Parameters
    ----------
    X, y: Shuffled input data and labels
    stratify: column to be stratified on.
    n_train: Number of samples in train set
    n_test: number of samples in test set
    x_numba: Determines whether the data should be converted to numba
    y_numba: Determines whether the labales should be converted to numba

    Returns
    -------
    X_train, X_test: Data X divided into train and test sets
    y_train, y_test: Labels divided into train and test sets
    """
    x_cudf = False
    labels_cudf = False

    if isinstance(X, cudf.DataFrame):
        x_cudf = True
    elif hasattr(X, "__cuda_array_interface__"):
        X = cp.asarray(X)
        x_order = _strides_to_order(
            X.__cuda_array_interface__["strides"], cp.dtype(X.dtype)
        )

    # labels and stratify will be only cp arrays
    if isinstance(labels, cudf.Series):
        labels_cudf = True
        labels = labels.values
    elif hasattr(labels, "__cuda_array_interface__"):
        labels = cp.asarray(labels)
    elif isinstance(stratify, cudf.DataFrame):
        # ensuring it has just one column
        if labels.shape[1] != 1:
            raise ValueError(
                "Expected one column for labels, but found df"
                "with shape = %d" % (labels.shape)
            )
        labels_cudf = True
        labels = labels[0].values

    labels_order = _strides_to_order(
        labels.__cuda_array_interface__["strides"], cp.dtype(labels.dtype)
    )

    # Converting to cupy array removes the need to add an if-else block
    # for startify column
    if isinstance(stratify, cudf.Series):
        stratify = stratify.values
    elif hasattr(stratify, "__cuda_array_interface__"):
        stratify = cp.asarray(stratify)
    elif isinstance(stratify, cudf.DataFrame):
        # ensuring it has just one column
        if stratify.shape[1] != 1:
            raise ValueError(
                "Expected one column, but found column"
                "with shape = %d" % (stratify.shape)
            )
        stratify = stratify[0].values

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

    class_indices = cp.split(
        cp.argsort(stratify_indices), cp.cumsum(class_counts)[:-1].tolist()
    )

    X_train = None

    # random_state won't be None or int, that's handled earlier
    if isinstance(random_state, np.random.RandomState):
        random_state = cp.random.RandomState(seed=random_state.get_state()[1])

    # Break ties
    n_i = _approximate_mode(class_counts, n_train, random_state)
    class_counts_remaining = class_counts - n_i
    t_i = _approximate_mode(class_counts_remaining, n_test, random_state)

    for i in range(n_classes):
        permutation = random_state.permutation(class_counts[i].item())
        perm_indices_class_i = class_indices[i].take(permutation)

        y_train_i = cp.array(labels[perm_indices_class_i[: n_i[i]]], order=labels_order)
        y_test_i = cp.array(
            labels[perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]]], order=labels_order
        )
        if hasattr(X, "__cuda_array_interface__") or isinstance(
            X, cupyx.scipy.sparse.csr_matrix
        ):

            X_train_i = cp.array(X[perm_indices_class_i[: n_i[i]]], order=x_order)
            X_test_i = cp.array(
                X[perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]]], order=x_order
            )

            if X_train is None:
                X_train = cp.array(X_train_i, order=x_order)
                y_train = cp.array(y_train_i, order=labels_order)
                X_test = cp.array(X_test_i, order=x_order)
                y_test = cp.array(y_test_i, order=labels_order)
            else:
                X_train = cp.concatenate([X_train, X_train_i], axis=0)
                X_test = cp.concatenate([X_test, X_test_i], axis=0)
                y_train = cp.concatenate([y_train, y_train_i], axis=0)
                y_test = cp.concatenate([y_test, y_test_i], axis=0)

        elif x_cudf:
            X_train_i = X.iloc[perm_indices_class_i[: n_i[i]]]
            X_test_i = X.iloc[perm_indices_class_i[n_i[i] : n_i[i] + t_i[i]]]

            if X_train is None:
                X_train = X_train_i
                y_train = y_train_i
                X_test = X_test_i
                y_test = y_test_i
            else:
                X_train = cudf.concat([X_train, X_train_i], ignore_index=False)
                X_test = cudf.concat([X_test, X_test_i], ignore_index=False)
                y_train = cp.concatenate([y_train, y_train_i], axis=0)
                y_test = cp.concatenate([y_test, y_test_i], axis=0)

    if x_numba:
        X_train = cuda.as_cuda_array(X_train)
        X_test = cuda.as_cuda_array(X_test)
    elif x_cudf:
        X_train = cudf.DataFrame(X_train)
        X_test = cudf.DataFrame(X_test)

    if y_numba:
        y_train = cuda.as_cuda_array(y_train)
        y_test = cuda.as_cuda_array(y_test)
    elif labels_cudf:
        y_train = cudf.Series(y_train)
        y_test = cudf.Series(y_test)

    return X_train, X_test, y_train, y_test


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
    test_size: Union[float, int] = None,
    train_size: Union[float, int] = None,
    shuffle: bool = True,
    random_state: Union[int, cp.random.RandomState, np.random.RandomState] = None,
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

    # todo: this check will be replaced with upcoming improvements
    # to input_utils
    #
    if y is not None:
        if not hasattr(X, "__cuda_array_interface__") and not isinstance(
            X, cudf.DataFrame
        ):
            raise TypeError(
                "X needs to be either a cuDF DataFrame, Series or \
                            a cuda_array_interface compliant array."
            )

        if not hasattr(y, "__cuda_array_interface__") and not isinstance(
            y, cudf.DataFrame
        ):
            raise TypeError(
                "y needs to be either a cuDF DataFrame, Series or \
                            a cuda_array_interface compliant array."
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must have the same first dimension"
                "(found {} and {})".format(X.shape[0], y.shape[0])
            )
    else:
        if not hasattr(X, "__cuda_array_interface__") and not isinstance(
            X, cudf.DataFrame
        ):
            raise TypeError(
                "X needs to be either a cuDF DataFrame, Series or \
                            a cuda_array_interface compliant object."
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

    x_numba = cuda.devicearray.is_cuda_ndarray(X)
    y_numba = cuda.devicearray.is_cuda_ndarray(y)

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
            idxs = cp.arange(X.shape[0])
            random_state = cp.random.RandomState(seed=random_state)

        elif isinstance(random_state, cp.random.RandomState):
            idxs = cp.arange(X.shape[0])

        elif isinstance(random_state, np.random.RandomState):
            idxs = np.arange(X.shape[0])

        else:
            raise TypeError(
                "`random_state` must be an int, NumPy RandomState \
                             or CuPy RandomState."
            )

        random_state.shuffle(idxs)

        if isinstance(X, cudf.DataFrame) or isinstance(X, cudf.Series):
            X = X.iloc[idxs]

        elif hasattr(X, "__cuda_array_interface__"):
            # numba (and therefore rmm device_array) does not support
            # fancy indexing
            X = cp.asarray(X)[idxs]

        if isinstance(y, cudf.DataFrame) or isinstance(y, cudf.Series):
            y = y.iloc[idxs]

        elif hasattr(y, "__cuda_array_interface__"):
            y = cp.asarray(y)[idxs]

        if stratify is not None:
            if isinstance(stratify, cudf.DataFrame) or isinstance(
                stratify, cudf.Series
            ):
                stratify = stratify.iloc[idxs]

            elif hasattr(stratify, "__cuda_array_interface__"):
                stratify = cp.asarray(stratify)[idxs]

            split_return = _stratify_split(
                X, stratify, y, train_size, test_size, x_numba, y_numba, random_state
            )
            return split_return

    # If not stratified, perform train_test_split splicing
    if hasattr(X, "__cuda_array_interface__"):
        x_order = _strides_to_order(
            X.__cuda_array_interface__["strides"], cp.dtype(X.dtype)
        )

    if hasattr(y, "__cuda_array_interface__"):
        y_order = _strides_to_order(
            y.__cuda_array_interface__["strides"], cp.dtype(y.dtype)
        )

    if hasattr(X, "__cuda_array_interface__") or isinstance(
        X, cupyx.scipy.sparse.csr_matrix
    ):
        X_train = cp.array(X[0:train_size], order=x_order)
        X_test = cp.array(X[-1 * test_size :], order=x_order)
        if y is not None:
            y_train = cp.array(y[0:train_size], order=y_order)
            y_test = cp.array(y[-1 * test_size :], order=y_order)
    elif isinstance(X, cudf.DataFrame):
        X_train = X.iloc[0:train_size]
        X_test = X.iloc[-1 * test_size :]
        if y is not None:
            if isinstance(y, cudf.Series):
                y_train = y.iloc[0:train_size]
                y_test = y.iloc[-1 * test_size :]
            elif hasattr(y, "__cuda_array_interface__") or isinstance(
                y, cupyx.scipy.sparse.csr_matrix
            ):
                y_train = cp.array(y[0:train_size], order=y_order)
                y_test = cp.array(y[-1 * test_size :], order=y_order)

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


class BaseCrossValidator(metaclass=ABCMeta):
    """Base class for all cross-validators
    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[cp.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Generates boolean masks corresponding to test sets.
        By default, delegates to _iter_test_indices(X, y, groups)
        """
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = cp.zeros(_num_samples(X), dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        pass

    def __repr__(self):
        return _build_repr(self)


class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    """Base class for KFold, GroupKFold, and StratifiedKFold"""

    @abstractmethod
    def __init__(self, n_splits, *, shuffle, random_state):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(
                "The number of folds must be of Integral type. "
                "%s of type %s was passed." % (n_splits, type(n_splits))
            )
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits)
            )

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False; got {0}".format(shuffle))

        if not shuffle and random_state is not None:  # None is the default
            raise ValueError(
                "Setting a random_state has no effect since shuffle is "
                "False. You should leave "
                "random_state to its default (None), or set shuffle=True.",
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                (
                    "Cannot have number of splits n_splits={0} greater"
                    " than the number of samples: n_samples={1}."
                ).format(self.n_splits, n_samples)
            )

        for train, test in super().split(X, y, groups):
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class KFold(_BaseKFold):
    """K-Folds cross-validator
    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds (without shuffling by default).
    Each fold is then used once as a validation while the k - 1 remaining
    folds form the training set.
    Read more in the :ref:`User Guide <k_fold>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import KFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> kf = KFold(n_splits=2)
    >>> kf.get_n_splits(X)
    2
    >>> print(kf)
    KFold(n_splits=2, random_state=None, shuffle=False)
    >>> for train_index, test_index in kf.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [2 3] TEST: [0 1]
    TRAIN: [0 1] TEST: [2 3]
    Notes
    -----
    The first ``n_samples % n_splits`` folds have size
    ``n_samples // n_splits + 1``, other folds have size
    ``n_samples // n_splits``, where ``n_samples`` is the number of samples.
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.
    See Also
    --------
    StratifiedKFold : Takes group information into account to avoid building
        folds with imbalanced class distributions (for binary or multiclass
        classification tasks).
    GroupKFold : K-fold iterator variant with non-overlapping groups.
    RepeatedKFold : Repeats K-Fold n times.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = cp.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = cp.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop


class GroupKFold(_BaseKFold):
    """K-fold iterator variant with non-overlapping groups.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    The folds are approximately balanced in the sense that the number of
    distinct groups is approximately the same in each fold.
    Read more in the :ref:`User Guide <group_k_fold>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupKFold
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 3, 4])
    >>> groups = np.array([0, 0, 2, 2])
    >>> group_kfold = GroupKFold(n_splits=2)
    >>> group_kfold.get_n_splits(X, y, groups)
    2
    >>> print(group_kfold)
    GroupKFold(n_splits=2)
    >>> for train_index, test_index in group_kfold.split(X, y, groups):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    ...     print(X_train, X_test, y_train, y_test)
    ...
    TRAIN: [0 1] TEST: [2 3]
    [[1 2]
     [3 4]] [[5 6]
     [7 8]] [1 2] [3 4]
    TRAIN: [2 3] TEST: [0 1]
    [[5 6]
     [7 8]] [[1 2]
     [3 4]] [3 4] [1 2]
    See Also
    --------
    LeaveOneGroupOut : For splitting the data according to explicit
        domain-specific stratification of the dataset.
    """

    def __init__(self, n_splits=5):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)

        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError(
                "Cannot have number of splits n_splits=%d greater"
                " than the number of groups: %d." % (self.n_splits, n_groups)
            )

        # Weight groups by their number of occurrences
        n_samples_per_group = cp.bincount(groups)

        # Distribute the most frequent groups first
        indices = cp.argsort(n_samples_per_group)[::-1]
        n_samples_per_group = n_samples_per_group[indices]

        # Total weight of each fold
        n_samples_per_fold = cp.zeros(self.n_splits)

        # Mapping from group index to fold index
        group_to_fold = cp.zeros(len(unique_groups))

        # Distribute samples by adding the largest weight to the lightest fold
        for group_index, weight in enumerate(n_samples_per_group):
            lightest_fold = cp.argmin(n_samples_per_fold)
            n_samples_per_fold[lightest_fold] += weight
            group_to_fold[indices[group_index]] = lightest_fold

        indices = group_to_fold[groups]

        for f in range(self.n_splits):
            yield cp.where(indices == f)[0]

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        return super().split(X, y, groups)


class StratifiedKFold(_BaseKFold):
    """Stratified K-Folds cross-validator.
    Provides train/test indices to split data in train/test sets.
    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class.
    Read more in the :ref:`User Guide <stratified_k_fold>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> skf = StratifiedKFold(n_splits=2)
    >>> skf.get_n_splits(X, y)
    2
    >>> print(skf)
    StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    >>> for train_index, test_index in skf.split(X, y):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [1 3] TEST: [0 2]
    TRAIN: [0 2] TEST: [1 3]
    Notes
    -----
    The implementation is designed to:
    * Generate test sets such that all contain the same distribution of
      classes, or as close as possible.
    * Be invariant to class label: relabelling ``y = ["Happy", "Sad"]`` to
      ``y = [1, 0]`` should not change the indices generated.
    * Preserve order dependencies in the dataset ordering, when
      ``shuffle=False``: all samples from class k in some test set were
      contiguous in y, or separated in y by samples from classes other than k.
    * Generate test sets where the smallest and largest differ by at most one
      sample.
    .. versionchanged:: 0.22
        The previous implementation did not follow the last constraint.
    See Also
    --------
    RepeatedStratifiedKFold : Repeats Stratified K-Fold n times.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _make_test_folds(self, X, y=None):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ("binary", "multiclass")
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                "Supported target types are: {}. Got {!r} instead.".format(
                    allowed_target_types, type_of_target_y
                )
            )

        y = column_or_1d(y)

        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        # y_inv encodes y according to lexicographic order. We invert y_idx to
        # map the classes so that they are encoded by order of appearance:
        # 0 represents the first label appearing in y, 1 the second, etc.
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]

        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError(
                "n_splits=%d cannot be greater than the"
                " number of members in each class." % (self.n_splits)
            )
        if self.n_splits > min_groups:
            warnings.warn(
                "The least populated class in y has only %d"
                " members, which is less than n_splits=%d."
                % (min_groups, self.n_splits),
                UserWarning,
            )

        # Determine the optimal number of samples from each class in each fold,
        # using round robin over the sorted y. (This can be done direct from
        # counts, but that code is unreadable.)
        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [
                np.bincount(y_order[i :: self.n_splits], minlength=n_classes)
                for i in range(self.n_splits)
            ]
        )

        # To maintain the data order dependencies as best as possible within
        # the stratification constraint, we assign samples from each class in
        # blocks (and then mess that up when shuffle=True).
        test_folds = np.empty(len(y), dtype="i")
        for k in range(n_classes):
            # since the kth column of allocation stores the number of samples
            # of class k in each test set, this generates blocks of fold
            # indices corresponding to the allocation for class k.
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                rng.shuffle(folds_for_class)
            test_folds[y_encoded == k] = folds_for_class
        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.
        groups : object
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y, groups)


def _build_repr(self):
    # XXX This is copied from BaseEstimator's get_params
    cls = self.__class__
    init = getattr(cls.__init__, "deprecated_original", cls.__init__)
    # Ignore varargs, kw and default values and pop self
    init_signature = signature(init)
    # Consider the constructor parameters excluding 'self'
    if init is object.__init__:
        args = []
    else:
        args = sorted(
            [
                p.name
                for p in init_signature.parameters.values()
                if p.name != "self" and p.kind != p.VAR_KEYWORD
            ]
        )
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        # We need deprecation warnings to always be on in order to
        # catch deprecated param values.
        # This is set in utils/__init__.py but it gets overwritten
        # when running under python3 somehow.
        warnings.simplefilter("always", FutureWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(self, key, None)
                if value is None and hasattr(self, "cvargs"):
                    value = self.cvargs.get(key, None)
            if len(w) and w[0].category == FutureWarning:
                # if the parameter is deprecated, don't show it
                continue
        finally:
            warnings.filters.pop(0)
        params[key] = value

    return "%s(%s)" % (class_name, _pprint(params, offset=len(class_name)))
