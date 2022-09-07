# Original authors from Sckit-Learn:
#          Nicolas Tresegnie <nicolas.tresegnie@gmail.com>
#          Sergey Feldman <sergeyfeldman@gmail.com>
# License: BSD 3 clause


# This code originates from the Scikit-Learn library,
# it was since modified to allow GPU acceleration.
# This code is under BSD 3 clause license.
# Authors mentioned above do not endorse or promote this production.


import numbers
import warnings

import numpy
import cupy as cp
import cuml
from cupyx.scipy import sparse

from ....thirdparty_adapters import (_get_mask,
                                     _masked_column_median,
                                     _masked_column_mean,
                                     _masked_column_mode)
from ..utils.skl_dependencies import BaseEstimator, TransformerMixin
from cuml.common.mixins import AllowNaNTagMixin, SparseInputTagMixin, \
    StringInputTagMixin
from ..utils.validation import (check_is_fitted, FLOAT_DTYPES)
from ....common.array_sparse import SparseCumlArray
from ....common.array_descriptor import CumlArrayDescriptor
from ....internals import _deprecate_pos_args
from cuml._thirdparty.sklearn.neighbors._base import (
    _check_weights, _get_weights)
from cuml.metrics.pairwise_distances import pairwise_distances

_NAN_METRICS = ["nan_euclidean"]


def is_scalar_nan(x):
    return bool(isinstance(x, numbers.Real) and cp.isnan(x))


def _check_inputs_dtype(X, missing_values):
    if (X.dtype.kind in ("f", "i", "u") and
            not isinstance(missing_values, numbers.Real)):
        raise ValueError("'X' and 'missing_values' types are expected to be"
                         " both numerical. Got X.dtype={} and "
                         " type(missing_values)={}."
                         .format(X.dtype, type(missing_values)))


def _get_elem_at_rank(rank, data, n_negative, n_zeros):
    """Find the value in data augmented with n_zeros for the given rank"""
    if rank < n_negative:
        return data[rank]
    if rank - n_negative < n_zeros:
        return 0
    return data[rank - n_zeros]


def _get_median(data, n_zeros):
    """Compute the median of data with n_zeros additional zeros.
    This function is used to support sparse matrices; it modifies data in-place
    """
    n_elems = len(data) + n_zeros
    if not n_elems:
        return cp.nan
    n_negative = (data < 0).sum()
    middle, is_odd = divmod(n_elems, 2)
    data = cp.sort(data)
    if is_odd:
        return _get_elem_at_rank(middle, data,
                                 n_negative, n_zeros)
    elm1 = _get_elem_at_rank(middle - 1, data,
                             n_negative, n_zeros)
    elm2 = _get_elem_at_rank(middle, data,
                             n_negative, n_zeros)
    return (elm1 + elm2) / 2.


def _most_frequent(array, extra_value, n_repeat):
    """Compute the most frequent value in a 1d array extended with
       [extra_value] * n_repeat, where extra_value is assumed to be not part
       of the array."""
    values, counts = cp.unique(array,
                               return_counts=True)
    most_frequent_count = counts.max()
    if most_frequent_count > n_repeat:
        value = values[counts == most_frequent_count].min()
    elif n_repeat > most_frequent_count:
        value = extra_value
    else:
        value = min(extra_value, values[counts == most_frequent_count].min())
    return value


class _BaseImputer(TransformerMixin):
    """Base class for all imputers.

    It adds automatically support for `add_indicator`.
    """

    def __init__(self, *, missing_values=cp.nan, add_indicator=False):
        self.missing_values = missing_values
        self.add_indicator = add_indicator

    def _fit_indicator(self, X):
        """Fit a MissingIndicator."""
        if self.add_indicator:
            with cuml.using_output_type("cupy"):
                self.indicator_ = MissingIndicator(
                    missing_values=self.missing_values, error_on_new=False
                )
                self.indicator_.fit(X)
        else:
            self.indicator_ = None

    def _transform_indicator(self, X):
        """Compute the indicator mask.'

        Note that X must be the original data as passed to the imputer before
        any imputation, since imputation may be done inplace in some cases.
        """
        if self.add_indicator:
            if not hasattr(self, 'indicator_'):
                raise ValueError(
                    "Make sure to call _fit_indicator before "
                    "_transform_indicator"
                )
            return self.indicator_.transform(X)

    def _concatenate_indicator(self, X_imputed, X_indicator):
        """Concatenate indicator mask with the imputed data."""
        if not self.add_indicator:
            return X_imputed

        hstack = sparse.hstack if sparse.issparse(X_imputed) else cp.hstack
        if X_indicator is None:
            raise ValueError(
                "Data from the missing indicator are not provided. Call "
                "_fit_indicator and _transform_indicator in the imputer "
                "implementation."
            )

        return hstack((X_imputed, X_indicator))

    def _more_tags(self):
        return {'allow_nan': is_scalar_nan(self.missing_values)}


class SimpleImputer(_BaseImputer, BaseEstimator,
                    SparseInputTagMixin, AllowNaNTagMixin):
    """Imputation transformer for completing missing values.

    Parameters
    ----------
    missing_values : number, string, cp.nan (default) or None
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `cp.nan`, since `pd.NA` will be converted to `cp.nan`.

    strategy : string, default='mean'
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.

        strategy="constant" for fixed value imputation.

    fill_value : string or numerical value, default=None
        When strategy == "constant", fill_value is used to replace all
        occurrences of missing_values.
        If left to the default, fill_value will be 0 when imputing numerical
        data and "missing_value" for strings or object data types.

    verbose : integer, default=0
        Controls the verbosity of the imputer.

    copy : boolean, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If X is not an array of floating values;
        - If X is encoded as a CSR matrix;
        - If add_indicator=True.

    add_indicator : boolean, default=False
        If True, a :class:`MissingIndicator` transform will stack onto output
        of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on
        the missing indicator even if there are missing values at
        transform/test time.

    Attributes
    ----------
    statistics_ : array of shape (n_features,)
        The imputation fill value for each feature.
        Computing statistics can result in `cp.nan` values.
        During :meth:`transform`, features corresponding to `cp.nan`
        statistics will be discarded.

    See also
    --------
    IterativeImputer : Multivariate imputation of missing values.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.preprocessing import SimpleImputer
    >>> imp_mean = SimpleImputer(missing_values=cp.nan, strategy='mean')
    >>> imp_mean.fit(cp.asarray([[7, 2, 3], [4, cp.nan, 6], [10, 5, 9]]))
    SimpleImputer()
    >>> X = [[cp.nan, 2, 3], [4, cp.nan, 6], [10, cp.nan, 9]]
    >>> print(imp_mean.transform(cp.asarray(X)))
    [[ 7.   2.   3. ]
     [ 4.   3.5  6. ]
     [10.   3.5  9. ]]

    Notes
    -----
    Columns which only contained missing values at :meth:`fit` are discarded
    upon :meth:`transform` if strategy is not "constant".

    """

    statistics_ = CumlArrayDescriptor()

    @_deprecate_pos_args(version="21.06")
    def __init__(self, *, missing_values=cp.nan, strategy="mean",
                 fill_value=None, copy=True, add_indicator=False):
        super().__init__(
            missing_values=missing_values,
            add_indicator=add_indicator
        )
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy

    def get_param_names(self):
        return super().get_param_names() + [
            "strategy",
            "fill_value",
            "verbose",
            "copy"
        ]

    def _validate_input(self, X, in_fit):
        allowed_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.strategy not in allowed_strategies:
            raise ValueError("Can only use these strategies: {0} "
                             " got strategy={1}".format(allowed_strategies,
                                                        self.strategy))

        if self.strategy in ("most_frequent", "constant"):
            dtype = None
        else:
            dtype = FLOAT_DTYPES

        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"

        try:
            X = self._validate_data(X, reset=in_fit,
                                    accept_sparse='csc', dtype=dtype,
                                    force_all_finite=force_all_finite,
                                    copy=self.copy)
        except ValueError as ve:
            if "could not convert" in str(ve):
                new_ve = ValueError("Cannot use {} strategy with non-numeric "
                                    "data:\n{}".format(self.strategy, ve))
                raise new_ve from None
            else:
                raise ve

        _check_inputs_dtype(X, self.missing_values)
        if X.dtype.kind not in ("i", "u", "f", "O"):
            raise ValueError("SimpleImputer does not support data with dtype "
                             "{0}. Please provide either a numeric array (with"
                             " a floating point or integer dtype) or "
                             "categorical data represented either as an array "
                             "with integer dtype or an array of string values "
                             "with an object dtype.".format(X.dtype))

        return X

    def fit(self, X, y=None) -> "SimpleImputer":
        """Fit the imputer on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : SimpleImputer
        """

        if isinstance(X, list):
            X = cp.asarray(X)

        X = self._validate_input(X, in_fit=True)
        super()._fit_indicator(X)

        # default fill_value is 0 for numerical input and "missing_value"
        # otherwise
        if self.fill_value is None:
            if X.dtype.kind in ("i", "u", "f"):
                fill_value = 0
            else:
                fill_value = "missing_value"
        else:
            fill_value = self.fill_value

        # fill_value should be numerical in case of numerical input
        if (self.strategy == "constant" and
                X.dtype.kind in ("i", "u", "f") and
                not isinstance(fill_value, numbers.Real)):
            raise ValueError("'fill_value'={0} is invalid. Expected a "
                             "numerical value when imputing numerical "
                             "data".format(fill_value))

        if sparse.issparse(X):
            # missing_values = 0 not allowed with sparse data as it would
            # force densification
            if self.missing_values == 0:
                raise ValueError("Imputation not possible when missing_values "
                                 "== 0 and input is sparse. Provide a dense "
                                 "array instead.")
            else:
                self.statistics_ = self._sparse_fit(X,
                                                    self.strategy,
                                                    self.missing_values,
                                                    fill_value)
        else:
            self.statistics_ = self._dense_fit(X,
                                               self.strategy,
                                               self.missing_values,
                                               fill_value)
        return self

    def _sparse_fit(self, X, strategy, missing_values, fill_value):
        """Fit the transformer on sparse data."""
        mask_data = _get_mask(X.data, missing_values)
        n_implicit_zeros = X.shape[0] - cp.diff(X.indptr)

        statistics = cp.empty(X.shape[1])

        if strategy == "constant":
            # for constant strategy, self.statistcs_ is used to store
            # fill_value in each column
            statistics.fill(fill_value)
        else:
            for i in range(X.shape[1]):
                column = X.data[X.indptr[i]:X.indptr[i + 1]]
                mask_column = mask_data[X.indptr[i]:X.indptr[i + 1]]
                column = column[~mask_column]

                # combine explicit and implicit zeros
                mask_zeros = _get_mask(column, 0)
                column = column[~mask_zeros]
                n_explicit_zeros = mask_zeros.sum()
                n_zeros = n_implicit_zeros[i] + n_explicit_zeros

                if strategy == "mean":
                    s = column.size + n_zeros
                    statistics[i] = cp.nan if s == 0 else column.sum() / s

                elif strategy == "median":
                    statistics[i] = _get_median(column,
                                                n_zeros)

                elif strategy == "most_frequent":
                    statistics[i] = _most_frequent(column,
                                                   0,
                                                   n_zeros)
        return statistics

    def _dense_fit(self, X, strategy, missing_values, fill_value):
        """Fit the transformer on dense data."""
        # Mean
        if strategy == "mean":
            return _masked_column_mean(X, missing_values)

        # Median
        elif strategy == "median":
            return _masked_column_median(X, missing_values)

        # Most frequent
        elif strategy == "most_frequent":
            return _masked_column_mode(X, missing_values)

        # Constant
        elif strategy == "constant":
            return cp.full(X.shape[1], fill_value, dtype=X.dtype)

    def transform(self, X) -> SparseCumlArray:
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.
        """
        check_is_fitted(self)

        X = self._validate_input(X, in_fit=False)
        X_indicator = super()._transform_indicator(X)

        statistics = self.statistics_

        if X.shape[1] != statistics.shape[0]:
            raise ValueError("X has %d features per sample, expected %d"
                             % (X.shape[1], self.statistics_.shape[0]))

        # Delete the invalid columns if strategy is not constant
        if self.strategy == "constant":
            valid_statistics = statistics
        else:
            # same as cp.isnan but also works for object dtypes
            invalid_mask = _get_mask(statistics, cp.nan)
            valid_mask = cp.logical_not(invalid_mask)
            valid_statistics = statistics[valid_mask]
            valid_statistics_indexes = cp.flatnonzero(valid_mask)

            if invalid_mask.any():
                missing = cp.arange(X.shape[1])[invalid_mask]
                if self.verbose:
                    warnings.warn("Deleting features without "
                                  "observed values: %s" % missing)
                X = X[:, valid_statistics_indexes]

        # Do actual imputation
        if sparse.issparse(X):
            if self.missing_values == 0:
                raise ValueError("Imputation not possible when missing_values "
                                 "== 0 and input is sparse. Provide a dense "
                                 "array instead.")
            else:
                mask = _get_mask(X.data, self.missing_values)
                indexes = cp.repeat(
                    cp.arange(len(X.indptr) - 1, dtype=int),
                    cp.diff(X.indptr).tolist())[mask]

                X.data[mask] = valid_statistics[indexes].astype(X.dtype,
                                                                copy=False)
        else:
            mask = _get_mask(X, self.missing_values)
            if self.strategy == "constant":
                X[mask] = valid_statistics[0]
            else:
                for i, vi in enumerate(valid_statistics_indexes):
                    feature_idxs = cp.flatnonzero(mask[:, vi])
                    X[feature_idxs, vi] = valid_statistics[i]

        X = super()._concatenate_indicator(X, X_indicator)
        return X


class MissingIndicator(TransformerMixin,
                       BaseEstimator,
                       AllowNaNTagMixin,
                       SparseInputTagMixin,
                       StringInputTagMixin):
    """Binary indicators for missing values.

    Note that this component typically should not be used in a vanilla
    :class:`Pipeline` consisting of transformers and a classifier, but rather
    could be added using a :class:`FeatureUnion` or :class:`ColumnTransformer`.

    Parameters
    ----------
    missing_values : number, string, cp.nan (default) or None
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to `cp.nan`, since `pd.NA` will be converted to `cp.nan`.

    features : str, default=None
        Whether the imputer mask should represent all or a subset of
        features.

        - If "missing-only" (default), the imputer mask will only represent
          features containing missing values during fit time.
        - If "all", the imputer mask will represent all features.

    sparse : boolean or "auto", default=None
        Whether the imputer mask format should be sparse or dense.

        - If "auto" (default), the imputer mask will be of same type as
          input.
        - If True, the imputer mask will be a sparse matrix.
        - If False, the imputer mask will be a numpy array.

    error_on_new : boolean, default=None
        If True (default), transform will raise an error when there are
        features with missing values in transform that have no missing values
        in fit. This is applicable only when ``features="missing-only"``.

    Attributes
    ----------
    features_ : ndarray, shape (n_missing_features,) or (n_features,)
        The features indices which will be returned when calling ``transform``.
        They are computed during ``fit``. For ``features='all'``, it is
        to ``range(n_features)``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.impute import MissingIndicator
    >>> X1 = np.array([[np.nan, 1, 3],
    ...                [4, 0, np.nan],
    ...                [8, 1, 0]])
    >>> X2 = np.array([[5, 1, np.nan],
    ...                [np.nan, 2, 3],
    ...                [2, 4, 0]])
    >>> indicator = MissingIndicator()
    >>> indicator.fit(X1)
    MissingIndicator()
    >>> X2_tr = indicator.transform(X2)
    >>> X2_tr
    array([[False,  True],
           [ True, False],
           [False, False]])

    """
    features_ = CumlArrayDescriptor()

    @_deprecate_pos_args(version="21.06")
    def __init__(self, *, missing_values=cp.nan, features="missing-only",
                 sparse="auto", error_on_new=True):
        self.missing_values = missing_values
        self.features = features
        self.sparse = sparse
        self.error_on_new = error_on_new

    def get_param_names(self):
        return super().get_param_names() + [
            "missing_values",
            "features",
            "sparse",
            "error_on_new"
        ]

    def _get_missing_features_info(self, X):
        """Compute the imputer mask and the indices of the features
        containing missing values.

        Parameters
        ----------
        X : {ndarray or sparse matrix}, shape (n_samples, n_features)
            The input data with missing values. Note that ``X`` has been
            checked in ``fit`` and ``transform`` before to call this function.

        Returns
        -------
        imputer_mask : {ndarray or sparse matrix}, shape \
        (n_samples, n_features)
            The imputer mask of the original data.

        features_with_missing : ndarray, shape (n_features_with_missing)
            The features containing missing values.

        """
        if sparse.issparse(X):
            mask = _get_mask(X.data, self.missing_values)

            # The imputer mask will be constructed with the same sparse format
            # as X.
            sparse_constructor = (sparse.csr_matrix if X.format == 'csr'
                                  else sparse.csc_matrix)
            imputer_mask = sparse_constructor(
                (mask, X.indices.copy(), X.indptr.copy()),
                shape=X.shape, dtype=cp.float32)
            # temporarly switch to using float32 as
            # cupy cannot operate with bool as of now

            if self.features == 'missing-only':
                n_missing = imputer_mask.sum(axis=0)

            if self.sparse is False:
                imputer_mask = imputer_mask.toarray()
            elif imputer_mask.format == 'csr':
                imputer_mask = imputer_mask.tocsc()
        else:
            imputer_mask = _get_mask(X, self.missing_values)

            if self.features == 'missing-only':
                n_missing = imputer_mask.sum(axis=0)

            if self.sparse is True:
                imputer_mask = sparse.csc_matrix(imputer_mask)

        if self.features == 'all':
            features_indices = cp.arange(X.shape[1])
        else:
            features_indices = cp.flatnonzero(n_missing)

        return imputer_mask, features_indices

    def _validate_input(self, X, in_fit):
        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"
        X = self._validate_data(X, reset=in_fit,
                                accept_sparse=('csc', 'csr'), dtype=None,
                                force_all_finite=force_all_finite)
        _check_inputs_dtype(X, self.missing_values)
        if X.dtype.kind not in ("i", "u", "f", "O"):
            raise ValueError("MissingIndicator does not support data with "
                             "dtype {0}. Please provide either a numeric array"
                             " (with a floating point or integer dtype) or "
                             "categorical data represented either as an array "
                             "with integer dtype or an array of string values "
                             "with an object dtype.".format(X.dtype))

        if sparse.issparse(X) and self.missing_values == 0:
            # missing_values = 0 not allowed with sparse data as it would
            # force densification
            raise ValueError("Sparse input with missing_values=0 is "
                             "not supported. Provide a dense "
                             "array instead.")

        return X

    def _fit(self, X, y=None):
        """Fit the transformer on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        imputer_mask : {ndarray or sparse matrix}, shape (n_samples, \
        n_features)
            The imputer mask of the original data.

        """
        X = self._validate_input(X, in_fit=True)
        self._n_features = X.shape[1]

        if self.features not in ('missing-only', 'all'):
            raise ValueError("'features' has to be either 'missing-only' or "
                             "'all'. Got {} instead.".format(self.features))

        if not ((isinstance(self.sparse, str) and
                self.sparse == "auto") or isinstance(self.sparse, bool)):
            raise ValueError("'sparse' has to be a boolean or 'auto'. "
                             "Got {!r} instead.".format(self.sparse))

        missing_features_info = self._get_missing_features_info(X)
        self.features_ = missing_features_info[1]

        return missing_features_info[0]

    def fit(self, X, y=None) -> "MissingIndicator":
        """Fit the transformer on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns self.
        """
        self._fit(X, y)

        return self

    def transform(self, X) -> SparseCumlArray:
        """Generate missing values indicator for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features) \
        or (n_samples, n_features_with_missing)
            The missing indicator for input data. The data type of ``Xt``
            will be boolean.

        """
        check_is_fitted(self)
        X = self._validate_input(X, in_fit=False)

        if X.shape[1] != self._n_features:
            raise ValueError("X has a different number of features "
                             "than during fitting.")

        imputer_mask, features = self._get_missing_features_info(X)

        if self.features == "missing-only":
            with cuml.using_output_type("numpy"):
                np_features = cp.asnumpy(features)
                features_diff_fit_trans = numpy.setdiff1d(np_features,
                                                          self.features_)
                if (self.error_on_new and features_diff_fit_trans.size > 0):
                    raise ValueError("The features {} have missing values "
                                     "in transform but have no missing values "
                                     "in fit.".format(features_diff_fit_trans))

            if self.features_.size < self._n_features:
                imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask

    def fit_transform(self, X, y=None) -> SparseCumlArray:
        """Generate missing values indicator for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray or sparse matrix}, shape (n_samples, n_features) \
        or (n_samples, n_features_with_missing)
            The missing indicator for input data. The data type of ``Xt``
            will be boolean.

        """
        imputer_mask = self._fit(X, y)

        if self.features_.size < self._n_features:
            imputer_mask = imputer_mask[:, self.features_]

        return imputer_mask


class KNNImputer(_BaseImputer, BaseEstimator):
    """Imputation for completing missing values using k-Nearest Neighbors.

    Each sample's missing values are imputed using the mean value from
    `n_neighbors` nearest neighbors found in the training set. Two samples are
    close if the features that neither is missing are close.

    Read more in the :ref:`User Guide <knnimpute>`.


    Parameters
    ----------
    missing_values : int, float, str, cp.nan or None, default=cp.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        should be set to cp.nan, since `pd.NA` will be converted to cp.nan.

    n_neighbors : int, default=5
        Number of neighboring samples to use for imputation.

    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights. All points in each neighborhood are
          weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - callable : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    metric : {'nan_euclidean'} or callable, default='nan_euclidean'
        Distance metric for searching neighbors. Possible values:

        - 'nan_euclidean'
        - callable : a user-defined function which conforms to the definition
          of ``_pairwise_callable(X, Y, metric, **kwds)``. The function
          accepts two arrays, X and Y, and a `missing_values` keyword in
          `kwds` and returns a scalar distance value.

    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible.

    add_indicator : bool, default=False
        If True, a :class:`MissingIndicator` transform will stack onto the
        output of the imputer's transform. This allows a predictive estimator
        to account for missingness despite imputation. If a feature has no
        missing values at fit/train time, the feature won't appear on the
        missing indicator even if there are missing values at transform/test
        time.

    Attributes
    ----------
    indicator_ : :class:`~sklearn.impute.MissingIndicator`
        Indicator used to add binary indicators for missing values.
        ``None`` if add_indicator is False.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    SimpleImputer : Imputation transformer for completing missing values
        with simple strategies.
    IterativeImputer : Multivariate imputer that estimates each feature
        from all the others.

    References
    ----------
    * Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor
      Hastie, Robert Tibshirani, David Botstein and Russ B. Altman, Missing
      value estimation methods for DNA microarrays, BIOINFORMATICS Vol. 17
      no. 6, 2001 Pages 520-525.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml._thirdparty.sklearn.preprocessing._imputation
    import KNNImputer
    >>> X = [[1, 2, cp.nan], [3, 4, 3],
            [cp.nan, 6, 5], [8, 8, 7]]
    >>> imputer = KNNImputer(n_neighbors=2)
    >>> imputer.fit_transform(X)
    array([[1. , 2. , 4. ],
           [3. , 4. , 3. ],
           [5.5, 6. , 5. ],
           [8. , 8. , 7. ]])
    """

    def __init__(
        self,
        *,
        missing_values=cp.nan,
        n_neighbors=5,
        weights="uniform",
        metric="nan_euclidean",
        copy=True,
        add_indicator=False,
    ):
        super().__init__(
            missing_values=missing_values, add_indicator=add_indicator)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.copy = copy

    def _calc_impute(
            self,
            dist_pot_donors,
            n_neighbors,
            fit_X_col,
            mask_fit_X_col):
        """Helper function to impute a single column.
        Parameters
        ----------
        dist_pot_donors : ndarray of shape (n_receivers, n_potential_donors)
            Distance matrix between the receivers and potential donors from
            training set. There must be at least one non-nan distance between
            a receiver and a potential donor.
        n_neighbors : int
            Number of neighbors to consider.
        fit_X_col : ndarray of shape (n_potential_donors,)
            Column of potential donors from training set.
        mask_fit_X_col : ndarray of shape (n_potential_donors,)
            Missing mask for fit_X_col.
        Returns
        -------
        imputed_values: ndarray of shape (n_receivers,)
            Imputed values for receiver.
        """
        # Get donors
        donors_idx = cp.argpartition(dist_pot_donors, n_neighbors - 1, axis=1)[
            :, :n_neighbors
        ]

        # Get weight matrix from distance matrix
        donors_dist = dist_pot_donors[
            cp.arange(donors_idx.shape[0])[:, None], donors_idx
        ]

        weight_matrix = _get_weights(donors_dist, self.weights)

        # Retrieve donor values and calculate kNN average
        donors = fit_X_col.take(donors_idx)
        donors_mask = mask_fit_X_col.take(donors_idx)
        # fill nans with zeros
        if weight_matrix is None:
            weight_matrix = cp.ones(donors_mask.shape)

        # Ignore values missing in donors
        weighted_donors = donors * weight_matrix
        donors_sum = cp.nansum(weighted_donors, axis=1)
        weights_sum = cp.nansum(weight_matrix, axis=1)

        return donors_sum / weights_sum

    def fit(self, X, y=None):
        """Fit the imputer on X.
        Parameters
        ----------
        X : array-like shape of (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        self : object
            The fitted `KNNImputer` class instance.
        """
        # Check data integrity and calling arguments
        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"
            if self.metric not in _NAN_METRICS and not callable(self.metric):
                raise ValueError(
                    "The selected metric does not support NaN values")
        if self.n_neighbors <= 0:
            raise ValueError(
                "Expected n_neighbors > 0. Got {}".format(self.n_neighbors)
            )

        X = self._validate_data(
            X,
            accept_sparse=False,
            dtype=FLOAT_DTYPES,
            force_all_finite=force_all_finite,
            copy=self.copy,
        )

        _check_weights(self.weights)
        self._fit_X = X
        self._mask_fit_X = _get_mask(self._fit_X, self.missing_values)
        self._valid_mask = ~cp.all(self._mask_fit_X, axis=0)

        super()._fit_indicator(self._mask_fit_X)

        return self

    def transform(self, X):
        """Impute all missing values in X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to complete.
        Returns
        -------
        X : array-like of shape (n_samples, n_output_features)
            The imputed dataset. `n_output_features` is the number of features
            that is not always missing during `fit`.
        """

        check_is_fitted(self)
        if not is_scalar_nan(self.missing_values):
            force_all_finite = True
        else:
            force_all_finite = "allow-nan"
        X = self._validate_data(
            X,
            accept_sparse=False,
            dtype=FLOAT_DTYPES,
            force_all_finite=force_all_finite,
            copy=self.copy,
            reset=False,
        )

        mask = _get_mask(X, self.missing_values)
        mask_fit_X = self._mask_fit_X
        valid_mask = self._valid_mask

        X_indicator = super()._transform_indicator(mask)

        # Removes columns where the training data is all nan
        if not cp.any(mask):
            # No missing values in X
            # Remove columns where the training data is all nan
            return X[:, valid_mask]

        row_missing_idx = cp.flatnonzero(mask.any(axis=1))

        non_missing_fix_X = cp.logical_not(mask_fit_X)

        # Maps from indices from X to indices in dist matrix
        dist_idx_map = cp.zeros(X.shape[0], dtype=int)
        dist_idx_map[row_missing_idx] = cp.arange(row_missing_idx.shape[0])

        if cp.isnan(self.missing_values
                    ) and self.metric not in _NAN_METRICS and not callable(
                self.metric):
            raise ValueError("The selected metric does not support NaN values")

        if not callable(self.metric):
            dist = cp.array(
                pairwise_distances(
                    X,
                    metric=self.metric,
                    missing_values=self.missing_values))
        else:
            raise ValueError(
                " The callable metric: '{}', is not supported at this time."
                .format(
                    self.metric))
        dist = dist[row_missing_idx][:, ]

        for col in range(X.shape[1]):
            if not valid_mask[col]:
                # column was all missing during training
                continue

            col_mask = mask[row_missing_idx, col]
            if not cp.any(col_mask):
                # column has no missing values
                continue

            (potential_donors_idx,) = cp.nonzero(non_missing_fix_X[:, col])

            # receivers_idx are indices in X
            receivers_idx = row_missing_idx[cp.flatnonzero(col_mask)]

            # distances for samples that needed imputation for column
            dist_subset = dist[dist_idx_map[receivers_idx]][
                :, potential_donors_idx
            ]

            # receivers with all nan distances impute with mean
            all_nan_dist_mask = cp.isnan(dist_subset).all(axis=1)
            all_nan_receivers_idx = receivers_idx[all_nan_dist_mask]
            if all_nan_receivers_idx.size:
                col_mean = numpy.ma.array(
                    self._fit_X[:, col].get(), mask=mask_fit_X[:, col].get()
                ).mean()

                X[all_nan_receivers_idx, col] = col_mean

                if len(all_nan_receivers_idx) == len(receivers_idx):
                    # all receivers imputed with mean
                    continue

                # receivers with at least one defined distance
                receivers_idx = receivers_idx[~all_nan_dist_mask]
                dist_subset = dist[dist_idx_map[receivers_idx]][
                    :, potential_donors_idx
                ]

            n_neighbors = min(self.n_neighbors, len(potential_donors_idx))
            value = self._calc_impute(
                dist_subset,
                n_neighbors,
                self._fit_X[potential_donors_idx, col],
                mask_fit_X[potential_donors_idx, col],
            )

            X[receivers_idx, col] = value

        return super()._concatenate_indicator(X[:, valid_mask], X_indicator)
