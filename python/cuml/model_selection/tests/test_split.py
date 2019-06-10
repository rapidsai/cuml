"""Test the split module"""
import warnings
import pytest
import cudf
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy import stats
from itertools import combinations
from itertools import combinations_with_replacement

from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_raises_regexp
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_greater_equal
from sklearn.utils.testing import assert_not_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_warns_message
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import assert_no_warnings
from sklearn.utils.validation import _num_samples
from sklearn.utils.mocking import MockDataFrame

# from cuml.model_selection import cross_val_score
from cuml.model_selection import KFold
from cuml.model_selection import StratifiedKFold
from cuml.model_selection import GroupKFold
from cuml.model_selection import TimeSeriesSplit
from cuml.model_selection import LeaveOneOut
from cuml.model_selection import LeaveOneGroupOut
from cuml.model_selection import LeavePOut
from cuml.model_selection import LeavePGroupsOut
from cuml.model_selection import ShuffleSplit
from cuml.model_selection import GroupShuffleSplit
from cuml.model_selection import StratifiedShuffleSplit
from cuml.model_selection import PredefinedSplit
from cuml.model_selection import check_cv
from cuml.model_selection import train_test_split
# from cuml.model_selection import GridSearchCV
from cuml.model_selection import RepeatedKFold
from cuml.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import Ridge

from sklearn.model_selection._split import _validate_shuffle_split
from cuml.model_selection._split import _build_repr
from cuml.model_selection._split import NSPLIT_WARNING

from sklearn.datasets import load_digits
from sklearn.datasets import make_classification

from .utils import (safe_indexing, indexable, _num_samples, comb,
    check_random_state, np_check_random_state, column_or_1d, in1d,
    _approximate_mode, type_of_target, _pprint)
from cuml.model_selection.utils import comb

X = cudf.DataFrame({'a': range(10)})
y = cp.arange(10) // 2
test_groups = (
    cp.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3]),
    cp.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
    cp.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2]),
    cp.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]),
    [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3])
digits = load_digits()


# helpre: convert a tuple of cupy array to tuple of numpy array
def cp2np(tup):
    return (cp.asnumpy(tup[0]), cp.asnumpy(tup[1]))


# Input: a cv
# Output: list of tuples (train idx, test idx (both are in numpy))
def conv2np(cv, X, y=None):
    if y is not None:
        return [(e[0].to_array(), e[1].to_array()) for e in list(cv.split(X, y))]
    else:
        return [(e[0].to_array(), e[1].to_array()) for e in list(cv.split(X))]


# helper: check intersection between two cupy array
def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    ar1 = cp.asanyarray(ar1)
    ar2 = cp.asanyarray(ar2)

    if not assume_unique:
        if return_indices:
            ar1, ind1 = cp.unique(ar1, return_index=True)
            ar2, ind2 = cp.unique(ar2, return_index=True)
        else:
            ar1 = cp.unique(ar1)
            ar2 = cp.unique(ar2)
    else:
        ar1 = ar1.ravel()
        ar2 = ar2.ravel()

    aux = cp.concatenate((ar1, ar2))
    if return_indices:
        aux_sort_indices = cp.argsort(aux, kind='mergesort')
        aux = aux[aux_sort_indices]
    else:
        aux.sort()

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - ar1.size
        if not assume_unique:
            ar1_indices = ind1[ar1_indices]
            ar2_indices = ind2[ar2_indices]

        return int1d, ar1_indices, ar2_indices
    else:
        return int1d


class MockClassifier:
    """Dummy classifier to test the cross-validation"""

    def __init__(self, a=0, allow_nd=False):
        self.a = a
        self.allow_nd = allow_nd

    def fit(self, X, Y=None, sample_weight=None, class_prior=None,
            sparse_sample_weight=None, sparse_param=None, dummy_int=None,
            dummy_str=None, dummy_obj=None, callback=None):
        """The dummy arguments are to test that this fit function can
        accept non-array arguments through cross-validation, such as:
            - int
            - str (this is actually array-like)
            - object
            - function
        """
        self.dummy_int = dummy_int
        self.dummy_str = dummy_str
        self.dummy_obj = dummy_obj
        if callback is not None:
            callback(self)

        if self.allow_nd:
            X = X.reshape(len(X), -1)
        if X.ndim >= 3 and not self.allow_nd:
            raise ValueError('X cannot be d')
        if sample_weight is not None:
            assert sample_weight.shape[0] == X.shape[0], (
                'MockClassifier extra fit_param sample_weight.shape[0]'
                ' is {0}, should be {1}'.format(sample_weight.shape[0],
                                                X.shape[0]))
        if class_prior is not None:
            assert class_prior.shape[0] == len(np.unique(y)), (
                        'MockClassifier extra fit_param class_prior.shape[0]'
                        ' is {0}, should be {1}'.format(class_prior.shape[0],
                                                        len(np.unique(y))))
        if sparse_sample_weight is not None:
            fmt = ('MockClassifier extra fit_param sparse_sample_weight'
                   '.shape[0] is {0}, should be {1}')
            assert sparse_sample_weight.shape[0] == X.shape[0], \
                fmt.format(sparse_sample_weight.shape[0], X.shape[0])
        if sparse_param is not None:
            fmt = ('MockClassifier extra fit_param sparse_param.shape '
                   'is ({0}, {1}), should be ({2}, {3})')
            assert sparse_param.shape == P_sparse.shape, (
                fmt.format(sparse_param.shape[0],
                           sparse_param.shape[1],
                           P_sparse.shape[0], P_sparse.shape[1]))
        return self

    def predict(self, T):
        if self.allow_nd:
            T = T.reshape(len(T), -1)
        return T[:, 0]

    def score(self, X=None, Y=None):
        return 1. / (1 + np.abs(self.a))

    def get_params(self, deep=False):
        return {'a': self.a, 'allow_nd': self.allow_nd}


# OK
@ignore_warnings
def test_cross_validator_with_default_params():
    n_samples = 4
    n_unique_groups = 4
    n_splits = 2
    p = 2
    n_shuffle_splits = 10  # (the default value)

    X = cudf.DataFrame({'a': list(range(4)), 'b': list(range(4))})
    X_1d = cudf.DataFrame([('a', list(range(4)))])
    y = cp.array([1, 1, 2, 2])
    groups = cp.array([1, 2, 3, 4])
    loo = LeaveOneOut()
    lpo = LeavePOut(p)
    kf = KFold(n_splits)
    skf = StratifiedKFold(n_splits)
    lolo = LeaveOneGroupOut()
    lopo = LeavePGroupsOut(p)
    ss = ShuffleSplit(random_state=0)
    ps = PredefinedSplit([1, 1, 2, 2])  # n_splits = np of unique folds = 2

    loo_repr = "LeaveOneOut()"
    lpo_repr = "LeavePOut(p=2)"
    kf_repr = "KFold(n_splits=2, random_state=None, shuffle=False)"
    skf_repr = "StratifiedKFold(n_splits=2, random_state=None, shuffle=False)"
    lolo_repr = "LeaveOneGroupOut()"
    lopo_repr = "LeavePGroupsOut(n_groups=2)"
    ss_repr = ("ShuffleSplit(n_splits=10, random_state=0, "
               "test_size=None, train_size=None)")
    ps_repr = "PredefinedSplit(test_fold=array([1, 1, 2, 2]))"

    n_splits_expected = [n_samples, comb(n_samples, p), n_splits, n_splits,
                         n_unique_groups, comb(n_unique_groups, p),
                         n_shuffle_splits, 2]

    for i, (cv, cv_repr) in enumerate(zip(
            [loo, lpo, kf, skf, lolo, lopo, ss, ps],
            [loo_repr, lpo_repr, kf_repr, skf_repr, lolo_repr, lopo_repr,
             ss_repr, ps_repr])):
        # Test if get_n_splits works correctly
        assert_equal(n_splits_expected[i], cv.get_n_splits(X, y, groups))

        # Test if the cross-validator works as expected even if
        # the data is 1d
        l1 = [cp2np(tup) for tup in list(cv.split(X, y, groups))]
        l2 = [cp2np(tup) for tup in list(cv.split(X_1d, y, groups))]
        np.testing.assert_equal(l1, l2)
        # Test that train, test indices returned are integers
        for train, test in cv.split(X, y, groups):
            assert_equal(train.to_array().dtype.kind, 'i')
            assert_equal(train.to_array().dtype.kind, 'i')

        # Test if the repr works without any errors
        assert_equal(cv_repr, repr(cv))

    # ValueError for get_n_splits methods
    msg = "The 'X' parameter should not be None."
    assert_raise_message(ValueError, msg,
                         loo.get_n_splits, None, y, groups)
    assert_raise_message(ValueError, msg,
                         lpo.get_n_splits, None, y, groups)


# OK
@pytest.mark.filterwarnings('ignore: The default value of n_split')  # 0.22
def test_2d_y():
    # smoke test for 2d y and multi-label
    n_samples = 30
    rng = cp.random.generator.RandomState(1)
    X = rng.randint(0, 3, size=(n_samples, 2))
    y = rng.randint(0, 3, size=(n_samples,))
    y_2d = y.reshape(-1, 1)
    y_multilabel = rng.randint(0, 2, size=(n_samples, 3))
    groups = rng.randint(0, 3, size=(n_samples,))
    splitters = [LeaveOneOut(), LeavePOut(p=2), KFold(), StratifiedKFold(),
                 RepeatedKFold(), RepeatedStratifiedKFold(),
                 ShuffleSplit(), StratifiedShuffleSplit(test_size=.5),
                 GroupShuffleSplit(), LeaveOneGroupOut(),
                 LeavePGroupsOut(n_groups=2), GroupKFold(), TimeSeriesSplit(),
                 PredefinedSplit(test_fold=groups)]
    for splitter in splitters:
        list(splitter.split(X, y, groups))
        list(splitter.split(X, y_2d, groups))
        try:
            list(splitter.split(X, y_multilabel, groups))
        except ValueError as e:
            allowed_target_types = ('binary', 'multiclass')
            msg = "Supported target types are: {}. Got 'multilabel".format(
                allowed_target_types)
            assert msg in str(e)


def check_valid_split(train, test, n_samples=None):
    # Use python sets to get more informative assertion failure messages
    train, test = set(train), set(test)

    # Train and test split should not overlap
    assert_equal(train.intersection(test), set())

    if n_samples is not None:
        # Check that the union of train an test split cover all the indices
        assert_equal(train.union(test), set(range(n_samples)))


def check_cv_coverage(cv, X, y, groups, expected_n_splits=None):
    n_samples = _num_samples(X)
    # Check that a all the samples appear at least once in a test fold
    if expected_n_splits is not None:
        assert_equal(cv.get_n_splits(X, y, groups), expected_n_splits)
    else:
        expected_n_splits = cv.get_n_splits(X, y, groups)

    collected_test_samples = set()
    iterations = 0
    for train, test in cv.split(X, y, groups):
        check_valid_split(train, test, n_samples=n_samples)
        iterations += 1
        collected_test_samples.update(test)

    # Check that the accumulated test samples cover the whole dataset
    assert_equal(iterations, expected_n_splits)
    if n_samples is not None:
        assert_equal(collected_test_samples, set(range(n_samples)))


# OK
def test_kfold_valueerrors():
    X1 = cudf.DataFrame([('a', list(range(3)))])
    X2 = cudf.DataFrame([('a', list(range(5)))])
    # Check that errors are raised if there is not enough samples
    (ValueError, next, KFold(4).split(X1))

    # Check that a warning is raised if the least populated class has too few
    # members.
    y = cp.array([3, 3, -1, -1, 3])

    skf_3 = StratifiedKFold(3)
    assert_warns_message(Warning, "The least populated class",
                         next, skf_3.split(X2, y))

    # Check that despite the warning the folds are still computed even
    # though all the classes are not necessarily represented at on each
    # side of the split at each split
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_cv_coverage(skf_3, X2, y, groups=None, expected_n_splits=3)

    # Check that errors are raised if all n_groups for individual
    # classes are less than n_splits.
    y = cp.array([3, 3, -1, -1, 2])

    assert_raises(ValueError, next, skf_3.split(X2, y))

    # Error when number of folds is <= 1
    assert_raises(ValueError, KFold, 0)
    assert_raises(ValueError, KFold, 1)
    error_string = ("k-fold cross-validation requires at least one"
                    " train/test split")
    assert_raise_message(ValueError, error_string,
                         StratifiedKFold, 0)
    assert_raise_message(ValueError, error_string,
                         StratifiedKFold, 1)

    # When n_splits is not integer:
    assert_raises(ValueError, KFold, 1.5)
    assert_raises(ValueError, KFold, 2.0)
    assert_raises(ValueError, StratifiedKFold, 1.5)
    assert_raises(ValueError, StratifiedKFold, 2.0)

    # When shuffle is not  a bool:
    assert_raises(TypeError, KFold, n_splits=4, shuffle=None)


# OK
def test_kfold_indices():
    # Check all indices are returned in the test folds
    X1 = cudf.DataFrame([('a', list(range(18)))])
    kf = KFold(3)
    check_cv_coverage(kf, X1, y=None, groups=None, expected_n_splits=3)

    # Check all indices are returned in the test folds even when equal-sized
    # folds are not possible
    X2 = cudf.DataFrame([('a', list(range(17)))])
    kf = KFold(3)
    check_cv_coverage(kf, X2, y=None, groups=None, expected_n_splits=3)

    # Check if get_n_splits returns the number of folds
    assert_equal(5, KFold(5).get_n_splits(X2))


# OK
def test_kfold_no_shuffle():
    # Manually check that KFold preserves the data ordering on toy datasets
    X2 = cudf.DataFrame([('a', list(range(5)))])

    splits = KFold(2).split(X2[:-1])
    train, test = next(splits)
    assert_array_equal(test, [0, 1])
    assert_array_equal(train, [2, 3])

    train, test = next(splits)
    assert_array_equal(test, [2, 3])
    assert_array_equal(train, [0, 1])

    splits = KFold(2).split(X2)
    train, test = next(splits)
    assert_array_equal(test, [0, 1, 2])
    assert_array_equal(train, [3, 4])

    train, test = next(splits)
    assert_array_equal(test, [3, 4])
    assert_array_equal(train, [0, 1, 2])


# OK, !!! but string label not supported
def test_stratified_kfold_no_shuffle():
    # Manually check that StratifiedKFold preserves the data ordering as much
    # as possible on toy datasets in order to avoid hiding sample dependencies
    # when possible
    X, y = cudf.DataFrame([('a', list(range(4)))]), [1, 1, 0, 0]
    splits = StratifiedKFold(2).split(X, y)
    train, test = next(splits)
    assert_array_equal(test, [0, 2])
    assert_array_equal(train, [1, 3])

    train, test = next(splits)
    assert_array_equal(test, [1, 3])
    assert_array_equal(train, [0, 2])

    X, y = cudf.DataFrame([('a', list(range(7)))]), [1, 1, 1, 0, 0, 0, 0]
    splits = StratifiedKFold(2).split(X, y)
    train, test = next(splits)
    assert_array_equal(test, [0, 1, 3, 4])
    assert_array_equal(train, [2, 5, 6])

    train, test = next(splits)
    assert_array_equal(test, [2, 5, 6])
    assert_array_equal(train, [0, 1, 3, 4])

    # Check if get_n_splits returns the number of folds
    assert_equal(5, StratifiedKFold(5).get_n_splits(X, y))


# OK
def test_stratified_kfold_ratios():
    # Check that stratified kfold preserves class ratios in individual splits
    # Repeat with shuffling turned off and on
    n_samples = 1000
    X = cudf.DataFrame([('a', list(range(n_samples)))])
    y = cp.array([4] * int(0.10 * n_samples) +
                 [0] * int(0.89 * n_samples) +
                 [1] * int(0.01 * n_samples))

    for shuffle in (False, True):
        for train, test in StratifiedKFold(5, shuffle=shuffle).split(X, y):
            train, test = train.to_array(), test.to_array()
            assert_almost_equal(np.sum(y[train] == 4) / len(train), 0.10, 2)
            assert_almost_equal(np.sum(y[train] == 0) / len(train), 0.89, 2)
            assert_almost_equal(np.sum(y[train] == 1) / len(train), 0.01, 2)
            assert_almost_equal(np.sum(y[test] == 4) / len(test), 0.10, 2)
            assert_almost_equal(np.sum(y[test] == 0) / len(test), 0.89, 2)
            assert_almost_equal(np.sum(y[test] == 1) / len(test), 0.01, 2)


# OK
def test_kfold_balance():
    # Check that KFold returns folds with balanced sizes
    for i in range(11, 17):
        kf = KFold(5).split(X=cudf.DataFrame([('a', list(range(i)))]))
        sizes = [len(test) for _, test in kf]

        assert (np.max(sizes) - np.min(sizes)) <= 1
        assert_equal(np.sum(sizes), i)


# OK
def test_stratifiedkfold_balance():
    # Check that KFold returns folds with balanced sizes (only when
    # stratification is possible)
    # Repeat with shuffling turned off and on
    X = cudf.DataFrame([('a', list(range(17)))])
    y = [0] * 3 + [1] * 14

    for shuffle in (True, False):
        cv = StratifiedKFold(3, shuffle=shuffle)
        for i in range(11, 17):
            skf = cv.split(X[:i], y[:i])
            sizes = [len(test) for _, test in skf]

            assert (np.max(sizes) - np.min(sizes)) <= 1
            assert_equal(np.sum(sizes), i)


# OK
def test_shuffle_kfold():
    # Check the indices are shuffled properly
    kf = KFold(3)
    kf2 = KFold(3, shuffle=True, random_state=0)
    kf3 = KFold(3, shuffle=True, random_state=1)

    X = cudf.DataFrame([('a', list(range(300)))])

    all_folds = np.zeros(300)
    for (tr1, te1), (tr2, te2), (tr3, te3) in zip(
            kf.split(X), kf2.split(X), kf3.split(X)):
        tr1, te1 = tr1.to_array(), te1.to_array()
        tr2, te2 = tr2.to_array(), te2.to_array()
        tr3, te3 = tr3.to_array(), te3.to_array()
        for tr_a, tr_b in combinations((tr1, tr2, tr3), 2):
            # Assert that there is no complete overlap
            assert_not_equal(len(np.intersect1d(tr_a, tr_b)), len(tr1))

        # Set all test indices in successive iterations of kf2 to 1
        all_folds[te2] = 1

    # Check that all indices are returned in the different test folds
    assert_equal(sum(all_folds), 300)


# OK
def test_shuffle_kfold_stratifiedkfold_reproducibility():
    X = cudf.DataFrame([('a', list(range(15)))])  # Divisible by 3
    y = [0] * 7 + [1] * 8
    X2 = cudf.DataFrame([('a', list(range(16)))])  # Not divisible by 3
    y2 = [0] * 8 + [1] * 8

    # Check that when the shuffle is True, multiple split calls produce the
    # same split when random_state is int
    kf = KFold(3, shuffle=True, random_state=0)
    skf = StratifiedKFold(3, shuffle=True, random_state=0)

    # help convert a tuple of cupy array to tuple of numpy array
    def cp2np(tup):
        return (cp.asnumpy(tup[0]), cp.asnumpy(tup[1]))

    for cv in (kf, skf):
        l1 = [cp2np(tup) for tup in list(cv.split(X, y))]
        l2 = [cp2np(tup) for tup in list(cv.split(X, y))]
        np.testing.assert_equal(l1, l2)
        l1 = [cp2np(tup) for tup in list(cv.split(X2, y2))]
        l2 = [cp2np(tup) for tup in list(cv.split(X2, y2))]
        np.testing.assert_equal(l1, l2)

    # Check that when the shuffle is True, multiple split calls often
    # (not always) produce different splits when random_state is
    # RandomState instance or None
    kf = KFold(3, shuffle=True, random_state=cp.random.generator.RandomState(0))
    skf = StratifiedKFold(3, shuffle=True,
                          random_state=cp.random.generator.RandomState(0))

    for cv in (kf, skf):
        for data in zip((X, X2), (y, y2)):
            # Test if the two splits are different cv
            for (_, test_a), (_, test_b) in zip(cv.split(*data),
                                                cv.split(*data)):
                # cv.split(...) returns an array of tuples, each tuple
                # consisting of an array with train indices and test indices
                # Ensure that the splits for data are not same
                # when random state is not set
                with pytest.raises(AssertionError):
                    np.testing.assert_array_equal(test_a, test_b)


# OK
def test_shuffle_stratifiedkfold():
    # Check that shuffling is happening when requested, and for proper
    # sample coverage
    X_40 = cudf.DataFrame([('a', list(range(40)))])
    y = [0] * 20 + [1] * 20
    kf0 = StratifiedKFold(5, shuffle=True, random_state=0)
    kf1 = StratifiedKFold(5, shuffle=True, random_state=1)
    for (_, test0), (_, test1) in zip(kf0.split(X_40, y),
                                      kf1.split(X_40, y)):
        assert_not_equal(set(test0), set(test1))
    check_cv_coverage(kf0, X_40, y, groups=None, expected_n_splits=5)

    # Ensure that we shuffle each class's samples with different
    # random_state in StratifiedKFold
    # See https://github.com/scikit-learn/scikit-learn/pull/13124
    X = cudf.DataFrame([('a', list(range(10)))])
    y = [0] * 5 + [1] * 5
    kf1 = StratifiedKFold(5, shuffle=True, random_state=0)
    kf2 = StratifiedKFold(5, shuffle=True, random_state=1)
    test_set1 = sorted([tuple(s[1]) for s in kf1.split(X, y)])
    test_set2 = sorted([tuple(s[1]) for s in kf2.split(X, y)])
    assert test_set1 != test_set2


# OK
def test_shuffle_split():
    ss1 = ShuffleSplit(test_size=0.2, random_state=0).split(X)
    ss2 = ShuffleSplit(test_size=2, random_state=0).split(X)
    ss3 = ShuffleSplit(test_size=np.int32(2), random_state=0).split(X)
    ss4 = ShuffleSplit(test_size=int(2), random_state=0).split(X)
    for t1, t2, t3, t4 in zip(ss1, ss2, ss3, ss4):
        assert_array_equal(t1[0], t2[0])
        assert_array_equal(t2[0], t3[0])
        assert_array_equal(t3[0], t4[0])
        assert_array_equal(t1[1], t2[1])
        assert_array_equal(t2[1], t3[1])
        assert_array_equal(t3[1], t4[1])


# OK
@pytest.mark.parametrize("split_class", [ShuffleSplit,
                                         StratifiedShuffleSplit])
@pytest.mark.parametrize("train_size, exp_train, exp_test",
                         [(None, 9, 1),
                          (8, 8, 2),
                          (0.8, 8, 2)])
def test_shuffle_split_default_test_size(split_class, train_size, exp_train,
                                         exp_test):
    # Check that the default value has the expected behavior, i.e. 0.1 if both
    # unspecified or complement train_size unless both are specified.
    X = cudf.DataFrame([('a', list(range(10)))])
    y = np.ones(10)

    X_train, X_test = next(split_class(train_size=train_size).split(X, y))

    assert len(X_train) == exp_train
    assert len(X_test) == exp_test


# OK
@pytest.mark.parametrize("train_size, exp_train, exp_test",
                         [(None, 8, 2),
                          (7, 7, 3),
                          (0.7, 7, 3)])
def test_group_shuffle_split_default_test_size(train_size, exp_train,
                                               exp_test):
    # Check that the default value has the expected behavior, i.e. 0.2 if both
    # unspecified or complement train_size unless both are specified.
    X = cudf.DataFrame([('a', list(range(10)))])
    y = np.ones(10)
    groups = range(10)

    X_train, X_test = next(GroupShuffleSplit(train_size=train_size)
                           .split(X, y, groups))

    assert len(X_train) == exp_train
    assert len(X_test) == exp_test


# OK
@ignore_warnings
def test_stratified_shuffle_split_init():
    X = cudf.DataFrame([('a', list(range(7)))])
    y = np.asarray([0, 1, 1, 1, 2, 2, 2])
    # Check that error is raised if there is a class with only one sample
    assert_raises(ValueError, next,
                  StratifiedShuffleSplit(3, 0.2).split(X, y))

    # Check that error is raised if the test set size is smaller than n_classes
    assert_raises(ValueError, next, StratifiedShuffleSplit(3, 2).split(X, y))
    # Check that error is raised if the train set size is smaller than
    # n_classes
    assert_raises(ValueError, next,
                  StratifiedShuffleSplit(3, 3, 2).split(X, y))

    X = cudf.DataFrame([('a', list(range(9)))])
    y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])

    # Train size or test size too small
    assert_raises(ValueError, next,
                  StratifiedShuffleSplit(train_size=2).split(X, y))
    assert_raises(ValueError, next,
                  StratifiedShuffleSplit(test_size=2).split(X, y))


# OK
def test_stratified_shuffle_split_respects_test_size():
    y = cp.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2])
    test_size = 5
    train_size = 10
    X = cudf.DataFrame([('a', list(range(len(y))))])
    sss = StratifiedShuffleSplit(6, test_size=test_size, train_size=train_size,
                                 random_state=0).split(X, y)
    for train, test in sss:
        assert_equal(len(train), train_size)
        assert_equal(len(test), test_size)


# OK
def test_stratified_shuffle_split_iter():
    ys = [cp.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3]),
          cp.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
          cp.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2] * 2),
          cp.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]),
          cp.array([-1] * 800 + [1] * 50),
          cp.array(np.concatenate([[i] * (100 + i) for i in range(11)])),
          [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
          #['1', '1', '1', '1', '2', '2', '2', '3', '3', '3', '3', '3'],
          ]

    for y in ys:
        sss = StratifiedShuffleSplit(6, test_size=0.33,
                                     random_state=0).split(cp.ones(len(y)), y)
        y = cp.asanyarray(y)  # To make it indexable for y[train]
        # this is how test-size is computed internally
        # in _validate_shuffle_split
        test_size = cp.ceil(0.33 * len(y))
        train_size = len(y) - test_size
        for train, test in sss:
            train, test = cp.array(train), cp.array(test)
            assert_array_equal(cp.unique(y[train]).tolist(), cp.unique(y[test]).tolist())
            # Checks if folds keep classes proportions
            p_train = (cp.bincount(cp.unique(y[train],
                                   return_inverse=True)[1]) /
                       float(len(y[train])))
            p_test = (cp.bincount(cp.unique(y[test],
                                  return_inverse=True)[1]) /
                      float(len(y[test])))
            assert_array_almost_equal(p_train.tolist(), p_test.tolist(), 1)
            assert_equal(len(train) + len(test), y.size)
            assert_equal(len(train), train_size)
            assert_equal(len(test), test_size)
            assert_array_equal(intersect1d(train, test).tolist(), [])


# OK
def test_stratified_shuffle_split_even():
    # Test the StratifiedShuffleSplit, indices are drawn with a
    # equal chance
    n_folds = 5
    n_splits = 1000

    def assert_counts_are_ok(idx_counts, p):
        # Here we test that the distribution of the counts
        # per index is close enough to a binomial
        threshold = 0.05 / n_splits
        bf = stats.binom(n_splits, p)
        for count in idx_counts:
            prob = bf.pmf(count)
            assert prob > threshold, \
                "An index is not drawn with chance corresponding to even draws"

    for n_samples in (6, 22):
        groups = cp.array((n_samples // 2) * [0, 1])
        splits = StratifiedShuffleSplit(n_splits=n_splits,
                                        test_size=1. / n_folds,
                                        random_state=0)

        train_counts = [0] * n_samples
        test_counts = [0] * n_samples
        n_splits_actual = 0
        for train, test in splits.split(X=np.ones(n_samples), y=groups):
            n_splits_actual += 1
            for counter, ids in [(train_counts, train), (test_counts, test)]:
                for id in ids:
                    counter[id] += 1
        assert_equal(n_splits_actual, n_splits)

        n_train, n_test = _validate_shuffle_split(
            n_samples, test_size=1. / n_folds, train_size=1. - (1. / n_folds))

        assert_equal(len(train), n_train)
        assert_equal(len(test), n_test)
        assert_equal(len(set(train).intersection(test)), 0)

        group_counts = cp.unique(groups)
        assert_equal(splits.test_size, 1.0 / n_folds)
        assert_equal(n_train + n_test, len(groups))
        assert_equal(len(group_counts), 2)
        ex_test_p = float(n_test) / n_samples
        ex_train_p = float(n_train) / n_samples

        assert_counts_are_ok(train_counts, ex_train_p)
        assert_counts_are_ok(test_counts, ex_test_p)


# OK
def test_stratified_shuffle_split_overlap_train_test_bug():
    # See https://github.com/scikit-learn/scikit-learn/issues/6121 for
    # the original bug report
    y = [0, 1, 2, 3] * 3 + [4, 5] * 5
    X = cudf.DataFrame([('a', list(range(len(y))))])

    sss = StratifiedShuffleSplit(n_splits=1,
                                 test_size=0.5, random_state=0)

    train, test = next(sss.split(X=X, y=y))
    train, test = train.to_array(), test.to_array()
    # no overlap
    assert_array_equal(np.intersect1d(train, test), [])

    # complete partition
    assert_array_equal(np.union1d(train, test), np.arange(len(y)))


# OK
def test_stratified_shuffle_split_multilabel():
    for y in [cp.array([[0, 1], [1, 0], [1, 0], [0, 1]]),
              cp.array([[0, 1], [1, 1], [1, 1], [0, 1]])]:
        X = cp.ones_like(y)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        train, test = next(sss.split(X=X, y=y))
        train, test = train.to_array(), test.to_array()
        y_train = y[train]
        y_test = y[test]

        # no overlap
        assert_array_equal(np.intersect1d(train, test), [])

        # complete partition
        print(type(train), type(test))
        assert_array_equal(np.unique(np.concatenate((train, test))), np.arange(len(y)))

        # correct stratification of entire rows
        # (by design, here y[:, 0] uniquely determines the entire row of y)
        expected_ratio = np.mean(y[:, 0])
        assert_equal(expected_ratio, np.mean(y_train[:, 0]))
        assert_equal(expected_ratio, np.mean(y_test[:, 0]))


# OK (involves many conversions, may not be the fastest, plus random_state
#     can only be none or int)
def test_stratified_shuffle_split_multilabel_many_labels():
    # fix in PR #9922: for multilabel data with > 1000 labels, str(row)
    # truncates with an ellipsis for elements in positions 4 through
    # len(row) - 4, so labels were not being correctly split using the powerset
    # method for transforming a multilabel problem to a multiclass one; this
    # test checks that this problem is fixed.
    row_with_many_zeros = [1, 0, 1] + [0] * 1000 + [1, 0, 1]
    row_with_many_ones = [1, 0, 1] + [1] * 1000 + [1, 0, 1]
    y = cp.array([row_with_many_zeros] * 10 + [row_with_many_ones] * 100)
    X = cp.ones_like(y)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    train, test = next(sss.split(X=X, y=y))
    train, test = train.to_array(), test.to_array()
    y_train = y[train]
    y_test = y[test]

    # correct stratification of entire rows
    # (by design, here y[:, 4] uniquely determines the entire row of y)
    expected_ratio = cp.mean(y[:, 4])
    assert_equal(expected_ratio, cp.mean(y_train[:, 4]))
    assert_equal(expected_ratio, cp.mean(y_test[:, 4]))


# OK
def test_predefinedsplit_with_kfold_split():
    # Check that PredefinedSplit can reproduce a split generated by Kfold.
    folds = cp.full(10, -1.)
    kf_train = []
    kf_test = []
    for i, (train_ind, test_ind) in enumerate(KFold(5, shuffle=True).split(X)):
        kf_train.append(train_ind)
        kf_test.append(test_ind)
        test_ind = test_ind.to_array()
        folds[test_ind] = i
    ps = PredefinedSplit(folds)
    # n_splits is simply the no of unique folds
    assert_equal(len(cp.unique(folds)), ps.get_n_splits())
    ps_train, ps_test = zip(*ps.split())
    assert_array_equal(ps_train, kf_train)
    assert_array_equal(ps_test, kf_test)


# OK !!!!!! but does not support array of strings as group
def test_group_shuffle_split():
    for groups_i in test_groups:
        X = cudf.DataFrame([('a', list(range(len(groups_i))))])
        y = cp.ones(len(groups_i))
        n_splits = 6
        test_size = 1. / 3
        slo = GroupShuffleSplit(n_splits, test_size=test_size, random_state=0)

        # Make sure the repr works
        repr(slo)

        # Test that the length is correct
        assert_equal(slo.get_n_splits(X, y, groups=groups_i), n_splits)

        l_unique = cp.unique(groups_i)
        l = cp.array(groups_i)

        for train, test in slo.split(X, y, groups=groups_i):
            # First test: no train group is in the test set and vice versa
            train, test = cp.array(train.to_array()), cp.array(test.to_array())
            l_train_unique = cp.unique(l[train])
            l_test_unique = cp.unique(l[test])
            assert not cp.any(in1d(l[train], l_test_unique))
            assert not cp.any(in1d(l[test], l_train_unique))

            # Second test: train and test add up to all the data
            assert_equal(l[train].size + l[test].size, l.size)

            # Third test: train and test are disjoint
            # !!!!!!!!!!!!!!! we have to transfer it back to numpy, since 
            # cupy array's __array__() is intentionally blocked for now.....
            assert_array_equal(cp.asnumpy(intersect1d(train, test)), [])

            # Fourth test:
            # unique train and test groups are correct, +- 1 for rounding error
            assert abs(len(l_test_unique) -
                       round(test_size * len(l_unique))) <= 1
            assert abs(len(l_train_unique) -
                       round((1.0 - test_size) * len(l_unique))) <= 1


# OK, !! but does not support string groups
def test_leave_one_p_group_out():
    logo = LeaveOneGroupOut()
    lpgo_1 = LeavePGroupsOut(n_groups=1)
    lpgo_2 = LeavePGroupsOut(n_groups=2)

    # Make sure the repr works
    assert_equal(repr(logo), 'LeaveOneGroupOut()')
    assert_equal(repr(lpgo_1), 'LeavePGroupsOut(n_groups=1)')
    assert_equal(repr(lpgo_2), 'LeavePGroupsOut(n_groups=2)')
    assert_equal(repr(LeavePGroupsOut(n_groups=3)),
                 'LeavePGroupsOut(n_groups=3)')

    for j, (cv, p_groups_out) in enumerate(((logo, 1), (lpgo_1, 1),
                                            (lpgo_2, 2))):
        for i, groups_i in enumerate(test_groups):
            n_groups = len(cp.unique(groups_i))
            n_splits = (n_groups if p_groups_out == 1
                        else n_groups * (n_groups - 1) / 2)
            X = cudf.DataFrame([('a', list(range(len(groups_i))))])
            y = cp.ones(len(groups_i))

            # Test that the length is correct
            assert_equal(cv.get_n_splits(X, y, groups=groups_i), n_splits)

            groups_arr = cp.asarray(groups_i)

            # Split using the original list / array / list of string groups_i
            for train, test in cv.split(X, y, groups=groups_i):
                # First test: no train group is in the test set and vice versa
                train = train.to_array()
                test = test.to_array()
                assert_array_equal(intersect1d(groups_arr[train],
                                                  groups_arr[test]).tolist(),
                                   [])

                # Second test: train and test add up to all the data
                assert_equal(len(train) + len(test), len(groups_i))

                # Third test:
                # The number of groups in test must be equal to p_groups_out
                assert cp.unique(groups_arr[test]).shape[0], p_groups_out

    # check get_n_splits() with dummy parameters
    # assert_equal(logo.get_n_splits(None, None, ['a', 'b', 'c', 'b', 'c']), 3)
    assert_equal(logo.get_n_splits(groups=[1.0, 1.1, 1.0, 1.2]), 3)
    assert_equal(lpgo_2.get_n_splits(None, None, cp.arange(4)), 6)
    assert_equal(lpgo_1.get_n_splits(groups=cp.arange(4)), 4)

    # raise ValueError if a `groups` parameter is illegal
    with assert_raises(ValueError):
        logo.get_n_splits(None, None, [0.0, cp.nan, 0.0])
    with assert_raises(ValueError):
        lpgo_2.get_n_splits(None, None, [0.0, cp.inf, 0.0])

    msg = "The 'groups' parameter should not be None."
    assert_raise_message(ValueError, msg,
                         logo.get_n_splits, None, None, None)
    assert_raise_message(ValueError, msg,
                         lpgo_1.get_n_splits, None, None, None)


# OK
def test_leave_group_out_changing_groups():
    # Check that LeaveOneGroupOut and LeavePGroupsOut work normally if
    # the groups variable is changed before calling split
    groups = cp.array([0, 1, 2, 1, 1, 2, 0, 0])
    X = cudf.DataFrame([('a', list(range(len(groups))))])
    groups_changing = cp.array(groups, copy=True)
    lolo = LeaveOneGroupOut().split(X, groups=groups)
    lolo_changing = LeaveOneGroupOut().split(X, groups=groups)
    lplo = LeavePGroupsOut(n_groups=2).split(X, groups=groups)
    lplo_changing = LeavePGroupsOut(n_groups=2).split(X, groups=groups)
    groups_changing[:] = 0
    for llo, llo_changing in [(lolo, lolo_changing), (lplo, lplo_changing)]:
        for (train, test), (train_chan, test_chan) in zip(llo, llo_changing):
            assert_array_equal(train, train_chan)
            assert_array_equal(test, test_chan)

    # n_splits = no of 2 (p) group combinations of the unique groups = 3C2 = 3
    assert_equal(
        3, LeavePGroupsOut(n_groups=2).get_n_splits(X, y=X,
                                                    groups=groups))
    # n_splits = no of unique groups (C(uniq_lbls, 1) = n_unique_groups)
    assert_equal(3, LeaveOneGroupOut().get_n_splits(X, y=X,
                                                    groups=groups))


# OK
def test_leave_one_p_group_out_error_on_fewer_number_of_groups():
    X = cudf.DataFrame([('a', list(range(0)))])
    # X = cp.array([])
    y = groups = cp.ones(0)
    assert_raise_message(ValueError, "Found array with 0 sample(s)", next,
                         LeaveOneGroupOut().split(X, y, groups))
    X = cudf.DataFrame([('a', list(range(1)))])
    y = groups = cp.ones(1)
    msg = ("The groups parameter contains fewer than 2 unique groups ({}). "
           "LeaveOneGroupOut expects at least 2.").format(groups)
    assert_raise_message(ValueError, msg, next,
                         LeaveOneGroupOut().split(X, y, groups))
    X = cudf.DataFrame([('a', list(range(1)))])
    y = groups = cp.ones(1)
    msg = ("The groups parameter contains fewer than (or equal to) n_groups "
           "(3) numbers of unique groups ({}). LeavePGroupsOut expects "
           "that at least n_groups + 1 (4) unique groups "
           "be present").format(groups)
    assert_raise_message(ValueError, msg, next,
                         LeavePGroupsOut(n_groups=3).split(X, y, groups))
    X = cudf.DataFrame([('a', list(range(3)))])
    y = groups = cp.arange(3)
    msg = ("The groups parameter contains fewer than (or equal to) n_groups "
           "(3) numbers of unique groups ({}). LeavePGroupsOut expects "
           "that at least n_groups + 1 (4) unique groups "
           "be present").format(groups)
    assert_raise_message(ValueError, msg, next,
                         LeavePGroupsOut(n_groups=3).split(X, y, groups))


# OK
@ignore_warnings
def test_repeated_cv_value_errors():
    # n_repeats is not integer or <= 0
    for cv in (RepeatedKFold, RepeatedStratifiedKFold):
        assert_raises(ValueError, cv, n_repeats=0)
        assert_raises(ValueError, cv, n_repeats=1.5)


# OK
def test_repeated_kfold_determinstic_split():
    X = cudf.DataFrame([('a', list(range(5)))])
    random_state = 258173307
    rkf = RepeatedKFold(
        n_splits=2,
        n_repeats=2,
        random_state=random_state)

    # split should produce same and deterministic splits on
    # each call
    for _ in range(3):
        splits = rkf.split(X)
        train, test = next(splits)
        assert_array_equal(train, [0, 3])
        assert_array_equal(test, [1, 2, 4])

        train, test = next(splits)
        assert_array_equal(train, [1, 2, 4])
        assert_array_equal(test, [0, 3])

        train, test = next(splits)
        assert_array_equal(train, [0, 4])
        assert_array_equal(test, [1, 2, 3])

        train, test = next(splits)
        assert_array_equal(train, [1, 2, 3])
        assert_array_equal(test, [0, 4])
        assert_raises(StopIteration, next, splits)


# OK
def test_get_n_splits_for_repeated_kfold():
    n_splits = 3
    n_repeats = 4
    rkf = RepeatedKFold(n_splits, n_repeats)
    expected_n_splits = n_splits * n_repeats
    assert_equal(expected_n_splits, rkf.get_n_splits())


# OK
def test_get_n_splits_for_repeated_stratified_kfold():
    n_splits = 3
    n_repeats = 4
    rskf = RepeatedStratifiedKFold(n_splits, n_repeats)
    expected_n_splits = n_splits * n_repeats
    assert_equal(expected_n_splits, rskf.get_n_splits())


# OK
def test_repeated_stratified_kfold_determinstic_split():
    X = cudf.DataFrame([('a', list(range(5)))])
    y = [1, 1, 1, 0, 0]
    random_state = 1944695409
    rskf = RepeatedStratifiedKFold(
        n_splits=2,
        n_repeats=2,
        random_state=random_state)

    # split should produce same and deterministic splits on
    # each call
    for _ in range(3):
        splits = rskf.split(X, y)
        train, test = next(splits)
        assert_array_equal(train, [2, 4])
        assert_array_equal(test, [0, 1, 3])

        train, test = next(splits)
        assert_array_equal(train, [0, 1, 3])
        assert_array_equal(test, [2, 4])

        train, test = next(splits)
        assert_array_equal(train, [0, 3])
        assert_array_equal(test, [1, 2, 4])

        train, test = next(splits)
        assert_array_equal(train, [1, 2, 4])
        assert_array_equal(test, [0, 3])

        assert_raises(StopIteration, next, splits)


# OK
@pytest.mark.parametrize("test_size, train_size",
                         [(2.0, None),
                          (1.0, None),
                          (0.1, 0.95),
                          (None, 1j),
                          (11, None),
                          (10, None),
                          (8, 3)])
def test_shufflesplit_errors(test_size, train_size):
    with pytest.raises(ValueError):
        next(ShuffleSplit(test_size=test_size, train_size=train_size).split(X))


# OK
def test_shufflesplit_reproducible():
    # Check that iterating twice on the ShuffleSplit gives the same
    # sequence of train-test when the random_state is given
    ss = ShuffleSplit(random_state=21)
    assert_array_equal(list(a for a, b in ss.split(X)),
                       list(a for a, b in ss.split(X)))


# OK, list is acceptable but elements cannot be string
def test_stratifiedshufflesplit_list_input():
    # Check that when y is a list / list of labels, it works.
    sss = StratifiedShuffleSplit(test_size=2, random_state=42)
    X = cudf.DataFrame([('a', list(range(7)))])
    y1 = [1] * 4 + [0] * 3
    y2 = cp.hstack((cp.ones(4), cp.zeros(3)))
    y3 = y2.tolist()

    np.testing.assert_equal(conv2np(sss, X, y=y1),
                            conv2np(sss, X, y=y2))
    np.testing.assert_equal(conv2np(sss, X, y=y3),
                            conv2np(sss, X, y=y2))


# OK
def test_check_cv():
    X = cudf.DataFrame([('a', list(range(9)))])
    cv = check_cv(3, classifier=False)
    # Use numpy.testing.assert_equal which recursively compares
    # lists of lists
    np.testing.assert_equal(conv2np(KFold(3), X),
                            conv2np(cv, X))

    y_binary = cp.array([0, 1, 0, 1, 0, 0, 1, 1, 1])
    cv = check_cv(3, y_binary, classifier=True)
    np.testing.assert_equal(conv2np(StratifiedKFold(3), X, y=y_binary),
                            conv2np(cv, X, y=y_binary))

    y_multiclass = np.array([0, 1, 0, 1, 2, 1, 2, 0, 2])
    cv = check_cv(3, y_multiclass, classifier=True)
    np.testing.assert_equal(conv2np(StratifiedKFold(3), X, y=y_multiclass),
                            conv2np(cv, X, y=y_multiclass))
    # also works with 2d multiclass
    y_multiclass_2d = y_multiclass.reshape(-1, 1)
    cv = check_cv(3, y_multiclass_2d, classifier=True)
    np.testing.assert_equal(conv2np(StratifiedKFold(3), X, y=y_multiclass_2d),
                            conv2np(cv, X, y=y_multiclass_2d))

    assert not cp.all(
        cp.array(next(StratifiedKFold(3).split(X, y_multiclass_2d))[0]) ==
        cp.array(next(KFold(3).split(X, y_multiclass_2d))[0]))

    X = cudf.DataFrame([('a', list(range(5)))])
    y_multilabel = cp.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1],
                             [1, 1, 0, 1], [0, 0, 1, 0]])
    cv = check_cv(3, y_multilabel, classifier=True)
    np.testing.assert_equal(conv2np(KFold(3), X), conv2np(cv, X))

    y_multioutput = cp.array([[1, 2], [0, 3], [0, 0], [3, 1], [2, 0]])
    cv = check_cv(3, y_multioutput, classifier=True)
    np.testing.assert_equal(conv2np(KFold(3), X), conv2np(cv, X))

    assert_raises(ValueError, check_cv, cv="lolo")


# OK
def test_cv_iterable_wrapper():
    kf_iter = KFold(n_splits=5).split(X, y)
    kf_iter_wrapped = check_cv(kf_iter)
    # Since the wrapped iterable is enlisted and stored,
    # split can be called any number of times to produce
    # consistent results.
    one = [(e[0].to_array(), e[1].to_array()) for e in list(kf_iter_wrapped.split(X, y))]
    two = [(e[0].to_array(), e[1].to_array()) for e in list(kf_iter_wrapped.split(X, y))]
    np.testing.assert_equal(one, two)
    # If the splits are randomized, successive calls to split yields different
    # results
    kf_randomized_iter = KFold(n_splits=5, shuffle=True).split(X, y)
    kf_randomized_iter_wrapped = check_cv(kf_randomized_iter)
    # numpy's assert_array_equal properly compares nested lists
    rand1 = [(e[0].to_array(), e[1].to_array()) for e in list(kf_randomized_iter_wrapped.split(X, y))]
    rand2 = [(e[0].to_array(), e[1].to_array()) for e in list(kf_randomized_iter_wrapped.split(X, y))]
    np.testing.assert_equal(rand1, rand2)

    try:
        rand = [(e[0].to_array(), e[1].to_array()) for e in list(kf_randomized_iter_wrapped.split(X, y))]
        nonrand = [(e[0].to_array(), e[1].to_array()) for e in list(kf_iter_wrapped.split(X, y))]
        np.testing.assert_equal(nonrand, rand)
        splits_are_equal = True
    except AssertionError:
        splits_are_equal = False
    assert not splits_are_equal, (
        "If the splits are randomized, "
        "successive calls to split should yield different results")


# OK
def test_group_kfold():
    rng = cp.random.generator.RandomState(0)

    # Parameters of the test
    n_groups = 15
    n_samples = 1000
    n_splits = 5

    X = cudf.DataFrame([('a', list(range(n_samples)))])
    y = cp.ones(n_samples)

    # Construct the test data
    tolerance = 0.05 * n_samples  # 5 percent error allowed
    groups = rng.randint(0, n_groups, n_samples)
    ideal_n_groups_per_fold = n_samples // n_splits

    len(cp.unique(groups))
    # Get the test fold indices from the test set indices of each fold
    folds = cp.zeros(n_samples)
    lkf = GroupKFold(n_splits=n_splits)
    for i, (_, test) in enumerate(lkf.split(X, y, groups)):
        test = cp.array(test)
        folds[test] = i

    # Check that folds have approximately the same size
    assert_equal(len(folds), len(groups))
    for i in cp.unique(folds):
        assert_greater_equal(tolerance,
                             abs(sum(folds == i) - ideal_n_groups_per_fold))

    # Check that each group appears only in 1 fold
    for group in cp.unique(groups):
        assert_equal(len(cp.unique(folds[groups == group])), 1)

    # Check that no group is on both sides of the split
    for train, test in lkf.split(X, y, groups):
        train, test = cp.array(train), cp.array(test)
        assert_equal(len(intersect1d(groups[train], groups[test])), 0)

    # Construct the test data
    groups = cp.array([0, 1, 2, 3, 1,
                       4, 5, 3, 6, 7,
                       8, 9, 10, 11, 1,
                       6, 12, 13, 14, 15, 16,
                       5, 10, 17, 18, 19, 20,
                       21, 22, 23, 24, 17,
                       4, 25, 26, 27, 28, 29])

    n_groups = len(cp.unique(groups))
    n_samples = len(groups)
    n_splits = 5
    tolerance = 0.05 * n_samples  # 5 percent error allowed
    ideal_n_groups_per_fold = n_samples // n_splits

    X = cudf.DataFrame([('a', list(range(n_samples)))])
    y = cp.ones(n_samples)

    # Get the test fold indices from the test set indices of each fold
    folds = cp.zeros(n_samples)
    for i, (_, test) in enumerate(lkf.split(X, y, groups)):
        test = cp.array(test)
        folds[test] = i

    # Check that folds have approximately the same size
    assert_equal(len(folds), len(groups))
    for i in cp.unique(folds):
        assert_greater_equal(tolerance,
                             abs(sum(folds == i) - ideal_n_groups_per_fold))

    # Check that each group appears only in 1 fold
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for group in cp.unique(groups):
            assert_equal(len(cp.unique(folds[groups == group])), 1)

    # Check that no group is on both sides of the split
    for train, test in lkf.split(X, y, groups):
        train, test = cp.array(train), cp.array(test)
        assert_equal(len(intersect1d(groups[train], groups[test])), 0)

    # groups can also be a list
    cv_iter = list(lkf.split(X, y, groups.tolist()))
    for (train1, test1), (train2, test2) in zip(lkf.split(X, y, groups),
                                                cv_iter):
        train1, test1 = train1.to_array(), test1.to_array()
        train2, test2 = train2.to_array(), test2.to_array()
        assert_array_equal(train1, train2)
        assert_array_equal(test1, test2)

    # Should fail if there are more folds than groups
    groups = cp.array([1, 1, 1, 2, 2])
    X = cudf.DataFrame([('a', list(range(len(groups))))])
    y = cp.ones(len(groups))
    assert_raises_regexp(ValueError, "Cannot have number of splits.*greater",
                         next, GroupKFold(n_splits=3).split(X, y, groups))


# OK
def test_time_series_cv():
    X = cudf.DataFrame([('a', list(range(7)))])

    # Should fail if there are more folds than samples
    assert_raises_regexp(ValueError, "Cannot have number of folds.*greater",
                         next,
                         TimeSeriesSplit(n_splits=7).split(X))

    tscv = TimeSeriesSplit(2)

    # Manually check that Time Series CV preserves the data
    # ordering on toy datasets
    splits = tscv.split(X[:-1])
    train, test = next(splits)
    assert_array_equal(train, [0, 1])
    assert_array_equal(test, [2, 3])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3])
    assert_array_equal(test, [4, 5])

    splits = TimeSeriesSplit(2).split(X)

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2])
    assert_array_equal(test, [3, 4])

    train, test = next(splits)
    assert_array_equal(train, [0, 1, 2, 3, 4])
    assert_array_equal(test, [5, 6])

    # Check get_n_splits returns the correct number of splits
    splits = TimeSeriesSplit(2).split(X)
    n_splits_actual = len(list(splits))
    assert_equal(n_splits_actual, tscv.get_n_splits())
    assert_equal(n_splits_actual, 2)


# OK
def _check_time_series_max_train_size(splits, check_splits, max_train_size):
    for (train, test), (check_train, check_test) in zip(splits, check_splits):
        assert_array_equal(test, check_test)
        assert len(check_train) <= max_train_size
        suffix_start = max(len(train) - max_train_size, 0)
        assert_array_equal(check_train, train[suffix_start:])


# OK
def test_time_series_max_train_size():
    X = cudf.DataFrame([('a', list(range(6)))])
    splits = TimeSeriesSplit(n_splits=3).split(X)
    check_splits = TimeSeriesSplit(n_splits=3, max_train_size=3).split(X)
    _check_time_series_max_train_size(splits, check_splits, max_train_size=3)

    # Test for the case where the size of a fold is greater than max_train_size
    check_splits = TimeSeriesSplit(n_splits=3, max_train_size=2).split(X)
    _check_time_series_max_train_size(splits, check_splits, max_train_size=2)

    # Test for the case where the size of each fold is less than max_train_size
    check_splits = TimeSeriesSplit(n_splits=3, max_train_size=5).split(X)
    _check_time_series_max_train_size(splits, check_splits, max_train_size=2)


# OK
def test_nsplit_default_warn():
    # Test that warnings are raised. Will be removed in 0.22
    assert_warns_message(FutureWarning, NSPLIT_WARNING, KFold)
    assert_warns_message(FutureWarning, NSPLIT_WARNING, GroupKFold)
    assert_warns_message(FutureWarning, NSPLIT_WARNING, StratifiedKFold)
    assert_warns_message(FutureWarning, NSPLIT_WARNING, TimeSeriesSplit)

    assert_no_warnings(KFold, n_splits=5)
    assert_no_warnings(GroupKFold, n_splits=5)
    assert_no_warnings(StratifiedKFold, n_splits=5)
    assert_no_warnings(TimeSeriesSplit, n_splits=5)


# OK
def test_build_repr():
    class MockSplitter:
        def __init__(self, a, b=0, c=None):
            self.a = a
            self.b = b
            self.c = c

        def __repr__(self):
            return _build_repr(self)

    assert_equal(repr(MockSplitter(5, 6)), "MockSplitter(a=5, b=6, c=None)")


# OK
@pytest.mark.parametrize('CVSplitter', (ShuffleSplit, GroupShuffleSplit,
                                        StratifiedShuffleSplit))
def test_shuffle_split_empty_trainset(CVSplitter):
    cv = CVSplitter(test_size=.99)
    X = cudf.DataFrame([('a', list(range(1)))])
    y = cp.array([0]) # 1 sample
    with pytest.raises(
            ValueError,
            match='With n_samples=1, test_size=0.99 and train_size=None, '
            'the resulting train set will be empty'):
        next(cv.split(X, y, groups=[1]))


# OK
def test_leave_one_out_empty_trainset():
    # LeaveOneGroup out expect at least 2 groups so no need to check
    cv = LeaveOneOut()
    X = cudf.DataFrame([('a', list(range(1)))])
    y = cp.array([0]) # 1 sample
    with pytest.raises(
            ValueError,
            match='Cannot perform LeaveOneOut with n_samples=1'):
        next(cv.split(X, y))


# OK
def test_leave_p_out_empty_trainset():
    # No need to check LeavePGroupsOut
    cv = LeavePOut(p=2)

    X = cudf.DataFrame([('a', list(range(2)))])
    y = cp.array([0, 3])  # 2 samples
    with pytest.raises(
            ValueError,
            match='p=2 must be strictly less than the number of samples=2'):
        next(cv.split(X, y, groups=[1, 2]))
