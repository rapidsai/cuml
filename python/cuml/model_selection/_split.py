# Author: Jianlin Du <jianlind@andrew.cmu.edu>
from collections.abc import Iterable
import warnings
from itertools import chain, combinations
from math import ceil, floor
import numbers
from abc import ABCMeta, abstractmethod
from inspect import signature

import scipy.sparse as sp
import numpy as np
import cupy as cp
import cudf

import utils


NSPLIT_WARNING = (
    "The default value of n_split will change from 3 to 5 "
    "in version 0.22. Specify it explicitly to silence this warning.")


class BaseCrossValidator(metaclass=ABCMeta):
    """Base class for all cross-validators
    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : cudf.DataFrame, cudf.Series or a cupy.ndarray, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : cudf dataframe, of length n_samples
            The target variable for supervised learning problems.
        groups : cudf dataframe, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : cudf Series
            The training set indices for that split.
        test : cudf Series
            The testing set indices for that split.
        """

        ############## do we need to do this for cudf, CHECK LATER
        #X, y, groups = indexable(X, y, groups)

        if isinstance(X, cp.ndarray):
            n_samples = len(X)
        if isinstance(X, (cudf.DataFrame, cudf.Series)):
            n_samples = len(X.index)

        indices = cp.arange(n_samples)
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[cp.logical_not(test_index)]
            train_index = cudf.from_dlpack(train_index.toDlpack())  # convert to cudf Series
            test_index = indices[test_index]
            test_index = cudf.from_dlpack(test_index.toDlpack())    # convert to cudf Series
            yield train_index, test_index

    # Since subclasses must implement either _iter_test_masks or
    # _iter_test_indices, neither can be abstract.
    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Generates boolean masks corresponding to test sets.
        By default, delegates to _iter_test_indices(X, y, groups)
        """
        if isinstance(X, cp.ndarray):
            n_samples = len(X)
        if isinstance(X, (cudf.DataFrame, cudf.Series)):
            n_samples = len(X.index)

        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = cp.zeros(n_samples, dtype=cp.bool_)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError

    @abstractmethod
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""

    def __repr__(self):
        return _build_repr(self)


class LeaveOneOut(BaseCrossValidator):
    """Leave-One-Out cross-validator
    Provides train/test indices to split data in train/test sets. Each
    sample is used once as a test set (singleton) while the remaining
    samples form the training set.
    Note: ``LeaveOneOut()`` is equivalent to ``KFold(n_splits=n)`` and
    ``LeavePOut(p=1)`` where ``n`` is the number of samples.
    Due to the high number of test sets (which is the same as the
    number of samples) this cross-validation method can be very costly.
    For large datasets one should favor :class:`KFold`, :class:`ShuffleSplit`
    or :class:`StratifiedKFold`.
    Read more in the :ref:`User Guide <cross_validation>`.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import LeaveOneOut
    >>> X = np.array([[1, 2], [3, 4]])
    >>> y = np.array([1, 2])
    >>> loo = LeaveOneOut()
    >>> loo.get_n_splits(X)
    2
    >>> print(loo)
    LeaveOneOut()
    >>> for train_index, test_index in loo.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    ...    print(X_train, X_test, y_train, y_test)
    TRAIN: [1] TEST: [0]
    [[3 4]] [[1 2]] [2] [1]
    TRAIN: [0] TEST: [1]
    [[1 2]] [[3 4]] [1] [2]
    See also
    --------
    LeaveOneGroupOut
        For splitting the data according to explicit, domain-specific
        stratification of the dataset.
    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = utils._num_samples(X)
        if n_samples <= 1:
            raise ValueError(
                'Cannot perform LeaveOneOut with n_samples={}.'.format(
                    n_samples)
            )
        return range(n_samples)

    def get_n_splits(self, X, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return utils._num_samples(X)


class LeavePOut(BaseCrossValidator):
    """Leave-P-Out cross-validator
    Provides train/test indices to split data in train/test sets. This results
    in testing on all distinct samples of size p, while the remaining n - p
    samples form the training set in each iteration.
    Note: ``LeavePOut(p)`` is NOT equivalent to
    ``KFold(n_splits=n_samples // p)`` which creates non-overlapping test sets.
    Due to the high number of iterations which grows combinatorically with the
    number of samples this cross-validation method can be very costly. For
    large datasets one should favor :class:`KFold`, :class:`StratifiedKFold`
    or :class:`ShuffleSplit`.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    p : int
        Size of the test sets. Must be strictly greater than the number of
        samples.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import LeavePOut
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 3, 4])
    >>> lpo = LeavePOut(2)
    >>> lpo.get_n_splits(X)
    6
    >>> print(lpo)
    LeavePOut(p=2)
    >>> for train_index, test_index in lpo.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [2 3] TEST: [0 1]
    TRAIN: [1 3] TEST: [0 2]
    TRAIN: [1 2] TEST: [0 3]
    TRAIN: [0 3] TEST: [1 2]
    TRAIN: [0 2] TEST: [1 3]
    TRAIN: [0 1] TEST: [2 3]
    """

    def __init__(self, p):
        self.p = p

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = utils._num_samples(X)
        if n_samples <= self.p:
            raise ValueError(
                'p={} must be strictly less than the number of '
                'samples={}'.format(self.p, n_samples)
            )
        for combination in combinations(range(n_samples), self.p):
            yield cp.array(combination)

    def get_n_splits(self, X, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        """
        if X is None:
            raise ValueError("The 'X' parameter should not be None.")
        return int(utils.comb(utils._num_samples(X), self.p, exact=True))



class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    """Base class for KFold, GroupKFold, and StratifiedKFold"""

    @abstractmethod
    def __init__(self, n_splits, shuffle, random_state):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError('The number of folds must be of Integral type. '
                             '%s of type %s was passed.'
                             % (n_splits, type(n_splits)))
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits))

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be True or False;"
                            " got {0}".format(shuffle))

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        ######## CHECK LATER
        #X, y, groups = indexable(X, y, groups)
        if isinstance(X, cp.ndarray):
            n_samples = len(X)
        if isinstance(X, (cudf.DataFrame, cudf.Series)):
            n_samples = len(X.index)

        if self.n_splits > n_samples:
            raise ValueError(
                ("Cannot have number of splits n_splits={0} greater"
                 " than the number of samples: n_samples={1}.")
                .format(self.n_splits, n_samples))

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
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
        .. versionchanged:: 0.20
            ``n_splits`` default value will change from 3 to 5 in v0.22.
    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `cp.random`. Used when ``shuffle`` == True.
    Examples
    --------
    >>> import cupy as cp
    >>> from sklearn.model_selection import KFold
    >>> X = cp.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = cp.array([1, 2, 3, 4])
    >>> kf = KFold(n_splits=2)
    >>> kf.get_n_splits(X)
    2
    >>> print(kf)  # doctest: +NORMALIZE_WHITESPACE
    KFold(n_splits=2, random_state=None, shuffle=False)
    >>> for train_index, test_index in kf.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [2 3] TEST: [0 1]
    TRAIN: [0 1] TEST: [2 3]
    Notes
    -----
    The first ``n_samples % n_splits`` folds have size
    ``n_samples // n_splits + 1``, other folds have size
    ``n_samples // n_splits``, where ``n_samples`` is the number of samples.
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting ``random_state``
    to an integer.
    See also
    --------
    StratifiedKFold
        Takes group information into account to avoid building folds with
        imbalanced class distributions (for binary or multiclass
        classification tasks).
    GroupKFold: K-fold iterator variant with non-overlapping groups.
    RepeatedKFold: Repeats K-Fold n times.
    """

    def __init__(self, n_splits='warn', shuffle=False,
                 random_state=None):
        if n_splits == 'warn':
            warnings.warn(NSPLIT_WARNING, FutureWarning)
            n_splits = 3
        super().__init__(n_splits, shuffle, random_state)

    def _iter_test_indices(self, X, y=None, groups=None):
        if isinstance(X, cp.ndarray):
            n_samples = len(X)
        if isinstance(X, (cudf.DataFrame, cudf.Series)):
            n_samples = len(X.index)

        indices = cp.arange(n_samples)
        
        if self.shuffle:
            utils.check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = cp.full(n_splits, n_samples // n_splits, dtype=cp.int)
        fold_sizes[:n_samples % n_splits] += 1
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
    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
        .. versionchanged:: 0.20
            ``n_splits`` default value will change from 3 to 5 in v0.22.
    Examples
    --------
    >>> import cupy as cp
    >>> from sklearn.model_selection import GroupKFold
    >>> X = cp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = cp.array([1, 2, 3, 4])
    >>> groups = cp.array([0, 0, 2, 2])
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
    See also
    --------
    LeaveOneGroupOut
        For splitting the data according to explicit domain-specific
        stratification of the dataset.
    """
    def __init__(self, n_splits='warn'):
        if n_splits == 'warn':
            warnings.warn(NSPLIT_WARNING, FutureWarning)
            n_splits = 3
        super().__init__(n_splits, shuffle=False, random_state=None)

    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        ##########################
        #groups = check_array(groups, ensure_2d=False, dtype=None)

        unique_groups, groups = cp.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError("Cannot have number of splits n_splits=%d greater"
                             " than the number of groups: %d."
                             % (self.n_splits, n_groups))

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
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,), optional
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,)
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
    """Stratified K-Folds cross-validator
    Provides train/test indices to split data in train/test sets.
    This cross-validation object is a variation of KFold that returns
    stratified folds. The folds are made by preserving the percentage of
    samples for each class.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.
        .. versionchanged:: 0.20
            ``n_splits`` default value will change from 3 to 5 in v0.22.
    shuffle : boolean, optional
        Whether to shuffle each class's samples before splitting into batches.
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `cp.random`. Used when ``shuffle`` == True.
    Examples
    --------
    >>> import cupy as cp
    >>> from sklearn.model_selection import StratifiedKFold
    >>> X = cp.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = cp.array([0, 0, 1, 1])
    >>> skf = StratifiedKFold(n_splits=2)
    >>> skf.get_n_splits(X, y)
    2
    >>> print(skf)  # doctest: +NORMALIZE_WHITESPACE
    StratifiedKFold(n_splits=2, random_state=None, shuffle=False)
    >>> for train_index, test_index in skf.split(X, y):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [1 3] TEST: [0 2]
    TRAIN: [0 2] TEST: [1 3]
    Notes
    -----
    Train and test sizes may be different in each fold, with a difference of at
    most ``n_classes``.
    See also
    --------
    RepeatedStratifiedKFold: Repeats Stratified K-Fold n times.
    """

    def __init__(self, n_splits='warn', shuffle=False, random_state=None):
        if n_splits == 'warn':
            warnings.warn(NSPLIT_WARNING, FutureWarning)
            n_splits = 3
        super().__init__(n_splits, shuffle, random_state)

    def _make_test_folds(self, X, y=None):
        rng = utils.check_random_state(self.random_state)
        y = cp.asarray(y)
        # type_of_target_y = type_of_target(y)
        # allowed_target_types = ('binary', 'multiclass')
        # if type_of_target_y not in allowed_target_types:
        #     raise ValueError(
        #         'Supported target types are: {}. Got {!r} instead.'.format(
        #             allowed_target_types, type_of_target_y))

        y = utils.column_or_1d(y)
        n_samples = y.shape[0]
        unique_y, y_inversed = cp.unique(y, return_inverse=True)
        y_counts = cp.bincount(y_inversed)
        min_groups = cp.min(y_counts)
        if cp.all(self.n_splits > y_counts):
            raise ValueError("n_splits=%d cannot be greater than the"
                             " number of members in each class."
                             % (self.n_splits))
        if self.n_splits > min_groups:
            warnings.warn(("The least populated class in y has only %d"
                           " members, which is too few. The minimum"
                           " number of members in any class cannot"
                           " be less than n_splits=%d."
                           % (min_groups, self.n_splits)), Warning)

        # pre-assign each sample to a test fold index using individual KFold
        # splitting strategies for each class so as to respect the balance of
        # classes
        # NOTE: Passing the data corresponding to ith class say X[y==class_i]
        # will break when the data is not 100% stratifiable for all classes.
        # So we pass cp.zeroes(max(c, n_splits)) as data to the KFold
        per_cls_cvs = [
            KFold(self.n_splits, shuffle=self.shuffle,
                  random_state=rng).split(cp.zeros(max(int(count), self.n_splits)))
            for count in y_counts]
        test_folds = cp.zeros(n_samples, dtype=cp.int)
        for test_fold_indices, per_cls_splits in enumerate(zip(*per_cls_cvs)):
            for cls, (_, test_split) in zip(unique_y, per_cls_splits):
                #############################################
                # In the future, delete the following conversion code after cudf supports cupy indexing
                test_split = cp.asarray(test_split)

                cls_test_folds = test_folds[y == cls]
                # the test split can be too big because we used
                # KFold(...).split(X[:max(c, n_splits)]) when data is not 100%
                # stratifiable for all the classes
                # (we use a warning instead of raising an exception)
                # If this is the case, let's trim it:
                test_split = test_split[test_split < len(cls_test_folds)]
                cls_test_folds[test_split] = test_fold_indices
                test_folds[y == cls] = cls_test_folds

        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            Note that providing ``y`` is sufficient to generate the splits and
            hence ``cp.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.
        y : array-like, shape (n_samples,)
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
        split. You can make the results identical by setting ``random_state``
        to an integer.
        """
        #y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y, groups)




class LeaveOneGroupOut(BaseCrossValidator):
    """Leave One Group Out cross-validator
    Provides train/test indices to split data according to a third-party
    provided group. This group information can be used to encode arbitrary
    domain specific stratifications of the samples as integers.
    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.
    Read more in the :ref:`User Guide <cross_validation>`.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import LeaveOneGroupOut
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 1, 2])
    >>> groups = np.array([1, 1, 2, 2])
    >>> logo = LeaveOneGroupOut()
    >>> logo.get_n_splits(X, y, groups)
    2
    >>> logo.get_n_splits(groups=groups)  # 'groups' is always required
    2
    >>> print(logo)
    LeaveOneGroupOut()
    >>> for train_index, test_index in logo.split(X, y, groups):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    ...    print(X_train, X_test, y_train, y_test)
    TRAIN: [2 3] TEST: [0 1]
    [[5 6]
     [7 8]] [[1 2]
     [3 4]] [1 2] [1 2]
    TRAIN: [0 1] TEST: [2 3]
    [[1 2]
     [3 4]] [[5 6]
     [7 8]] [1 2] [1 2]
    """

    def _iter_test_masks(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        # We make a copy of groups to avoid side-effects during iteration
        ############################################3
        # groups = check_array(groups, copy=True, ensure_2d=False, dtype=None)
        unique_groups = cp.unique(groups)
        if len(unique_groups) <= 1:
            raise ValueError(
                "The groups parameter contains fewer than 2 unique groups "
                "(%s). LeaveOneGroupOut expects at least 2." % unique_groups)
        for i in unique_groups:
            yield groups == i

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set. This 'groups' parameter must always be specified to
            calculate the number of splits, though the other parameters can be
            omitted.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        ################################################
        # groups = check_array(groups, ensure_2d=False, dtype=None)
        return len(cp.unique(groups))

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples, optional
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,)
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



class LeavePGroupsOut(BaseCrossValidator):
    """Leave P Group(s) Out cross-validator
    Provides train/test indices to split data according to a third-party
    provided group. This group information can be used to encode arbitrary
    domain specific stratifications of the samples as integers.
    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.
    The difference between LeavePGroupsOut and LeaveOneGroupOut is that
    the former builds the test sets with all the samples assigned to
    ``p`` different values of the groups while the latter uses samples
    all assigned the same groups.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_groups : int
        Number of groups (``p``) to leave out in the test split.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import LeavePGroupsOut
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 1])
    >>> groups = np.array([1, 2, 3])
    >>> lpgo = LeavePGroupsOut(n_groups=2)
    >>> lpgo.get_n_splits(X, y, groups)
    3
    >>> lpgo.get_n_splits(groups=groups)  # 'groups' is always required
    3
    >>> print(lpgo)
    LeavePGroupsOut(n_groups=2)
    >>> for train_index, test_index in lpgo.split(X, y, groups):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    ...    print(X_train, X_test, y_train, y_test)
    TRAIN: [2] TEST: [0 1]
    [[5 6]] [[1 2]
     [3 4]] [1] [1 2]
    TRAIN: [1] TEST: [0 2]
    [[3 4]] [[1 2]
     [5 6]] [2] [1 1]
    TRAIN: [0] TEST: [1 2]
    [[1 2]] [[3 4]
     [5 6]] [1] [2 1]
    See also
    --------
    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_groups):
        self.n_groups = n_groups

    def _iter_test_masks(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        #######################################
        # groups = check_array(groups, copy=True, ensure_2d=False, dtype=None)
        unique_groups = cp.unique(groups)
        if self.n_groups >= len(unique_groups):
            raise ValueError(
                "The groups parameter contains fewer than (or equal to) "
                "n_groups (%d) numbers of unique groups (%s). LeavePGroupsOut "
                "expects that at least n_groups + 1 (%d) unique groups be "
                "present" % (self.n_groups, unique_groups, self.n_groups + 1))
        combi = combinations(range(len(unique_groups)), self.n_groups)
        for indices in combi:
            test_index = cp.zeros(utils._num_samples(X), dtype=cp.bool)
            for l in unique_groups[cp.array(indices)]:
                test_index[groups == l] = True
            yield test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set. This 'groups' parameter must always be specified to
            calculate the number of splits, though the other parameters can be
            omitted.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        # groups = check_array(groups, ensure_2d=False, dtype=None)
        return int(utils.comb(len(cp.unique(groups)), self.n_groups, exact=True))

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples, optional
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,)
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


class _RepeatedSplits(metaclass=ABCMeta):
    """Repeated splits for an arbitrary randomized CV splitter.
    Repeats splits for cross-validators n times with different randomization
    in each repetition.
    Parameters
    ----------
    cv : callable
        Cross-validator class.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    **cvargs : additional params
        Constructor parameters for cv. Must not contain random_state
        and shuffle.
    """
    def __init__(self, cv, n_repeats=10, random_state=None, **cvargs):
        if not isinstance(n_repeats, (np.integer, numbers.Integral)):
            raise ValueError("Number of repetitions must be of Integral type.")

        if n_repeats <= 0:
            raise ValueError("Number of repetitions must be greater than 0.")

        if any(key in cvargs for key in ('random_state', 'shuffle')):
            raise ValueError(
                "cvargs must not contain random_state or shuffle.")

        self.cv = cv
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.cvargs = cvargs

    def split(self, X, y=None, groups=None):
        """Generates indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_repeats = self.n_repeats
        rng = check_random_state(self.random_state)

        for idx in range(n_repeats):
            cv = self.cv(random_state=rng, shuffle=True,
                         **self.cvargs)
            for train_index, test_index in cv.split(X, y, groups):
                yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.
        y : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        rng = check_random_state(self.random_state)
        cv = self.cv(random_state=rng, shuffle=True,
                     **self.cvargs)
        return cv.get_n_splits(X, y, groups) * self.n_repeats



class _RepeatedSplits(metaclass=ABCMeta):
    """Repeated splits for an arbitrary randomized CV splitter.
    Repeats splits for cross-validators n times with different randomization
    in each repetition.
    Parameters
    ----------
    cv : callable
        Cross-validator class.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    **cvargs : additional params
        Constructor parameters for cv. Must not contain random_state
        and shuffle.
    """
    def __init__(self, cv, n_repeats=10, random_state=None, **cvargs):
        if not isinstance(n_repeats, (cp.integer, np.integer, numbers.Integral)):
            raise ValueError("Number of repetitions must be of Integral type.")

        if n_repeats <= 0:
            raise ValueError("Number of repetitions must be greater than 0.")

        if any(key in cvargs for key in ('random_state', 'shuffle')):
            raise ValueError(
                "cvargs must not contain random_state or shuffle.")

        self.cv = cv
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.cvargs = cvargs

    def split(self, X, y=None, groups=None):
        """Generates indices to split data into training and test set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_repeats = self.n_repeats
        rng = utils.check_random_state(self.random_state)

        for idx in range(n_repeats):
            cv = self.cv(random_state=rng, shuffle=True,
                         **self.cvargs)
            for train_index, test_index in cv.split(X, y, groups):
                yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.
        y : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        rng = utils.check_random_state(self.random_state)
        cv = self.cv(random_state=rng, shuffle=True,
                     **self.cvargs)
        return cv.get_n_splits(X, y, groups) * self.n_repeats


class RepeatedKFold(_RepeatedSplits):
    """Repeated K-Fold cross validator.
    Repeats K-Fold n times with different randomization in each repetition.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import RepeatedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=2652124)
    >>> for train_index, test_index in rkf.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    ...
    TRAIN: [0 1] TEST: [2 3]
    TRAIN: [2 3] TEST: [0 1]
    TRAIN: [1 2] TEST: [0 3]
    TRAIN: [0 3] TEST: [1 2]
    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting ``random_state``
    to an integer.
    See also
    --------
    RepeatedStratifiedKFold: Repeats Stratified K-Fold n times.
    """
    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            KFold, n_repeats, random_state, n_splits=n_splits)



class RepeatedStratifiedKFold(_RepeatedSplits):
    """Repeated Stratified K-Fold cross validator.
    Repeats Stratified K-Fold n times with different randomization in each
    repetition.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : None, int or RandomState, default=None
        Random state to be used to generate random state for each
        repetition.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import RepeatedStratifiedKFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 1, 1])
    >>> rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,
    ...     random_state=36851234)
    >>> for train_index, test_index in rskf.split(X, y):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    ...
    TRAIN: [1 2] TEST: [0 3]
    TRAIN: [0 3] TEST: [1 2]
    TRAIN: [1 3] TEST: [0 2]
    TRAIN: [0 2] TEST: [1 3]
    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting ``random_state``
    to an integer.
    See also
    --------
    RepeatedKFold: Repeats K-Fold n times.
    """
    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            StratifiedKFold, n_repeats, random_state, n_splits=n_splits)



def _build_repr(self):
    # XXX This is copied from BaseEstimator's get_params
    cls = self.__class__
    init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
    # Ignore varargs, kw and default values and pop self
    init_signature = signature(init)
    # Consider the constructor parameters excluding 'self'
    if init is object.__init__:
        args = []
    else:
        args = sorted([p.name for p in init_signature.parameters.values()
                       if p.name != 'self' and p.kind != p.VAR_KEYWORD])
    class_name = self.__class__.__name__
    params = dict()
    for key in args:
        # We need deprecation warnings to always be on in order to
        # catch deprecated param values.
        # This is set in utils/__init__.py but it gets overwritten
        # when running under python3 somehow.
        warnings.simplefilter("always", DeprecationWarning)
        try:
            with warnings.catch_warnings(record=True) as w:
                value = getattr(self, key, None)
            if len(w) and w[0].category == DeprecationWarning:
                # if the parameter is deprecated, don't show it
                continue
        finally:
            warnings.filters.pop(0)
        params[key] = value

    return '%s(%s)' % (class_name, _pprint(params, offset=len(class_name)))