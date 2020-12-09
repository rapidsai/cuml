# Original authors from Sckit-Learn:
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Andreas Mueller <amueller@ais.uni-bonn.de>
#          Eric Martin <eric@ericmart.in>
#          Giorgio Patrini <giorgio.patrini@anu.edu.au>
#          Eric Chang <ericchang2017@u.northwestern.edu>
# License: BSD 3 clause


# This code originates from the Scikit-Learn library,
# it was since modified to allow GPU acceleration.
# This code is under BSD 3 clause license.
# Authors mentioned above do not endorse or promote this production.


from itertools import chain, combinations
import numbers
import warnings
from itertools import combinations_with_replacement as combinations_w_r

import cupy as np
from cupy import sparse

from ..utils.skl_dependencies import BaseEstimator, TransformerMixin
from ....thirdparty_adapters import check_array
from ..utils.extmath import row_norms
from ..utils.extmath import _incremental_mean_and_var
from ..utils.validation import (check_is_fitted, FLOAT_DTYPES,
                                _deprecate_positional_args)

from ..utils.sparsefuncs import (inplace_column_scale,
                                 min_max_axis,
                                 mean_variance_axis)

from ....thirdparty_adapters.sparsefuncs_fast import \
    (inplace_csr_row_normalize_l1, inplace_csr_row_normalize_l2,
     csr_polynomial_expansion)
from ....common.import_utils import check_cupy8

from ....common.array import CumlArray
from ....common.array_sparse import SparseCumlArray
from ....common.array_descriptor import CumlArrayDescriptor
from ....internals import api_return_generic
from ....common.memory_utils import using_output_type


__all__ = [
    'Binarizer',
    'MinMaxScaler',
    'MaxAbsScaler',
    'Normalizer',
    'RobustScaler',
    'StandardScaler',
    'add_dummy_feature',
    'binarize',
    'normalize',
    'scale',
    'robust_scale',
    'maxabs_scale',
    'minmax_scale'
]


def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.

    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


@_deprecate_positional_args
@api_return_generic(get_output_type=True)
def scale(X, *, axis=0, with_mean=True, with_std=True, copy=True):
    """Standardize a dataset along any axis

    Center to the mean and component wise scale to unit variance.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        The data to center and scale.

    axis : int (0 by default)
        axis used to compute the means and standard deviations along. If 0,
        independently standardize each feature, otherwise (if 1) standardize
        each sample.

    with_mean : boolean, True by default
        If True, center the data before scaling.

    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : boolean, optional, default True
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    Notes
    -----
    This implementation will refuse to center sparse matrices
    since it would make them non-sparse and would potentially crash the
    program with memory exhaustion problems.

    Instead the caller is expected to either set explicitly
    `with_mean=False` (in that case, only variance scaling will be
    performed on the features of the sparse matrix) or to densify the matrix
    if he/she expects the materialized dense array to fit in memory.

    For optimal processing the caller should pass a CSC matrix.

    NaNs are treated as missing values: disregarded to compute the statistics,
    and maintained during the data transformation.

    We use a biased estimator for the standard deviation, equivalent to
    `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
    affect model performance.

    See also
    --------
    StandardScaler: Performs scaling to unit variance using the``Transformer`` API

    """  # noqa
    X = check_array(X, accept_sparse=['csr', 'csc'], copy=copy,
                    ensure_2d=False, estimator='the scale function',
                    dtype=FLOAT_DTYPES, force_all_finite='allow-nan')

    if sparse.issparse(X):
        if with_mean:
            raise ValueError(
                "Cannot center sparse matrices: pass `with_mean=False` instead"
                " See docstring for motivation and alternatives.")
        if axis != 0:
            raise ValueError("Can only scale sparse matrix on axis=0, "
                             " got axis=%d" % axis)
        if with_std:
            _, var = mean_variance_axis(X, axis=0)
            var = _handle_zeros_in_scale(var, copy=False)
            inplace_column_scale(X, 1 / np.sqrt(var))
    else:
        X = np.asarray(X)
        if with_mean:
            mean_ = np.nanmean(X, axis)
        if with_std:
            scale_ = np.nanstd(X, axis)
        # Xr is a view on the original array that enables easy use of
        # broadcasting on the axis in which we are interested in
        Xr = np.rollaxis(X, axis)
        if with_mean:
            Xr -= mean_
            mean_1 = np.nanmean(Xr, axis=0)
            # Verify that mean_1 is 'close to zero'. If X contains very
            # large values, mean_1 can also be very large, due to a lack of
            # precision of mean_. In this case, a pre-scaling of the
            # concerned feature is efficient, for instance by its mean or
            # maximum.
            if not np.allclose(mean_1, 0):
                warnings.warn("Numerical issues were encountered "
                              "when centering the data "
                              "and might not be solved. Dataset may "
                              "contain too large values. You may need "
                              "to prescale your features.")
                Xr -= mean_1
        if with_std:
            scale_ = _handle_zeros_in_scale(scale_, copy=False)
            Xr /= scale_
            if with_mean:
                mean_2 = np.nanmean(Xr, axis=0)
                # If mean_2 is not 'close to zero', it comes from the fact that
                # scale_ is very small so that mean_2 = mean_1/scale_ > 0, even
                # if mean_1 was close to zero. The problem is thus essentially
                # due to the lack of precision of mean_. A solution is then to
                # subtract the mean again:
                if not np.allclose(mean_2, 0):
                    warnings.warn("Numerical issues were encountered "
                                  "when scaling the data "
                                  "and might not be solved. The standard "
                                  "deviation of the data is probably "
                                  "very close to 0. ")
                    Xr -= mean_2

    return X


class MinMaxScaler(TransformerMixin, BaseEstimator):
    """Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.

    The transformation is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    copy : bool, default=True
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Per feature adjustment for minimum. Equivalent to
        ``min - X.min(axis=0) * self.scale_``

    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data. Equivalent to
        ``(max - min) / (X.max(axis=0) - X.min(axis=0))``

    data_min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the data

    data_max_ : ndarray of shape (n_features,)
        Per feature maximum seen in the data

    data_range_ : ndarray of shape (n_features,)
        Per feature range ``(data_max_ - data_min_)`` seen in the data

    n_samples_seen_ : int
        The number of samples processed by the estimator.
        It will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.

    Examples
    --------
    >>> from cuml.preprocessing import MinMaxScaler
    >>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    >>> scaler = MinMaxScaler()
    >>> print(scaler.fit(data))
    MinMaxScaler()
    >>> print(scaler.data_max_)
    [ 1. 18.]
    >>> print(scaler.transform(data))
    [[0.   0.  ]
     [0.25 0.25]
     [0.5  0.5 ]
     [1.   1.  ]]
    >>> print(scaler.transform([[2, 2]]))
    [[1.5 0. ]]

    See also
    --------
    minmax_scale: Equivalent function without the estimator API.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.
    """

    scale_ = CumlArrayDescriptor()
    min_ = CumlArrayDescriptor()
    n_samples_seen_ = CumlArrayDescriptor()
    data_min_ = CumlArrayDescriptor()
    data_max_ = CumlArrayDescriptor()
    data_range_ = CumlArrayDescriptor()

    @_deprecate_positional_args
    def __init__(self, feature_range=(0, 1), *, copy=True):
        self.feature_range = feature_range
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            self.scale_ = None
            self.min_ = None
            self.n_samples_seen_ = None
            self.data_min_ = None
            self.data_max_ = None
            self.data_range_ = None

    def get_param_names(self):
        return super().get_param_names() + [
            "feature_range",
            "copy"
        ]

    def fit(self, X, y=None) -> "MinMaxScaler":
        """Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted scaler.
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None) -> "MinMaxScaler":
        """Online computation of min and max on X for later scaling.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Transformer instance.
        """
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))

        first_pass = not hasattr(self, 'n_samples_seen_')
        X = self._validate_data(X, reset=first_pass,
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite="allow-nan")

        data_min = np.nanmin(X, axis=0)
        data_max = np.nanmax(X, axis=0)

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            data_min = np.minimum(self.data_min_, data_min)
            data_max = np.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = ((feature_range[1] - feature_range[0]) /
                       _handle_zeros_in_scale(data_range))
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X) -> CumlArray:
        """Scale features of X according to feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.

        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
                        force_all_finite="allow-nan")

        X *= self.scale_
        X += self.min_

        return X

    def inverse_transform(self, X) -> CumlArray:
        """Undo the scaling of X according to feature_range.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.

        Returns
        -------
        Xt : array-like of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
                        force_all_finite="allow-nan")

        X -= self.min_
        X /= self.scale_
        return X

    def _more_tags(self):
        return {'allow_nan': True}


@_deprecate_positional_args
@api_return_generic(get_output_type=True)
def minmax_scale(X, feature_range=(0, 1), *, axis=0, copy=True):
    """Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, i.e. between
    zero and one.

    The transformation is given by (when ``axis=0``)::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    The transformation is calculated as (when ``axis=0``)::

       X_scaled = scale * X + min - X.min(axis=0) * scale
       where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data.

    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    axis : int, default=0
        Axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.

    copy : bool, default=True
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    See also
    --------
    MinMaxScaler: Performs scaling to a given range using the``Transformer`` API
    """  # noqa
    # Unlike the scaler object, this function allows 1d input.
    # If copy is required, it will be done inside the scaler object.

    X = check_array(X, copy=False, ensure_2d=False,
                    dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
    original_ndim = X.ndim

    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)

    with using_output_type('cupy'):
        s = MinMaxScaler(feature_range=feature_range, copy=copy)
        if axis == 0:
            X = s.fit_transform(X)
        else:
            X = s.fit_transform(X.T).T

        if original_ndim == 1:
            X = X.ravel()

        return X


class StandardScaler(TransformerMixin, BaseEstimator):
    """Standardize features by removing the mean and scaling to unit variance

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the training samples or zero if `with_mean=False`,
    and `s` is the standard deviation of the training samples or one if
    `with_std=False`.

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using
    :meth:`transform`.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual features do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).

    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    that others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.

    This scaler can also be applied to sparse CSR or CSC matrices by passing
    `with_mean=False` to avoid breaking the sparsity structure of the data.

    Parameters
    ----------
    copy : boolean, optional, default True
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    with_mean : boolean, True by default
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    Attributes
    ----------
    scale_ : ndarray or None, shape (n_features,)
        Per feature relative scaling of the data. This is calculated using
        `sqrt(var_)`. Equal to ``None`` when ``with_std=False``.

    mean_ : ndarray or None, shape (n_features,)
        The mean value for each feature in the training set.
        Equal to ``None`` when ``with_mean=False``.

    var_ : ndarray or None, shape (n_features,)
        The variance for each feature in the training set. Used to compute
        `scale_`. Equal to ``None`` when ``with_std=False``.

    n_samples_seen_ : int or array, shape (n_features,)
        The number of samples processed by the estimator for each feature.
        If there are not missing samples, the ``n_samples_seen`` will be an
        integer, otherwise it will be an array.
        Will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.

    Examples
    --------
    >>> from cuml.preprocessing import StandardScaler
    >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    >>> scaler = StandardScaler()
    >>> print(scaler.fit(data))
    StandardScaler()
    >>> print(scaler.mean_)
    [0.5 0.5]
    >>> print(scaler.transform(data))
    [[-1. -1.]
     [-1. -1.]
     [ 1.  1.]
     [ 1.  1.]]
    >>> print(scaler.transform([[2, 2]]))
    [[3. 3.]]

    See also
    --------
    scale: Equivalent function without the estimator API.

    :class:`cuml.decomposition.PCA`
        Further removes the linear correlation across features with 'whiten=True'.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.

    We use a biased estimator for the standard deviation, equivalent to
    `numpy.std(x, ddof=0)`. Note that the choice of `ddof` is unlikely to
    affect model performance.
    """  # noqa

    scale_ = CumlArrayDescriptor()
    n_samples_seen_ = CumlArrayDescriptor()
    mean_ = CumlArrayDescriptor()
    var_ = CumlArrayDescriptor()

    @_deprecate_positional_args
    def __init__(self, *, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            self.scale_ = None
            self.n_samples_seen_ = None
            self.mean_ = None
            self.var_ = None

    def get_param_names(self):
        return super().get_param_names() + [
            "with_mean",
            "with_std",
            "copy"
        ]

    def fit(self, X, y=None) -> "StandardScaler":
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y
            Ignored
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None) -> "StandardScaler":
        """
        Online computation of mean and std on X for later scaling.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Transformer instance.
        """
        X = self._validate_data(X, accept_sparse=('csr', 'csc'),
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan')

        # Even in the case of `with_mean=False`, we update the mean anyway
        # This is needed for the incremental computation of the var
        # See incr_mean_variance_axis and _incremental_mean_variance_axis

        # if n_samples_seen_ is an integer (i.e. no missing values), we need to
        # transform it to a NumPy array of shape (n_features,) required by
        # incr_mean_variance_axis and _incremental_variance_axis
        if (hasattr(self, 'n_samples_seen_') and
                isinstance(self.n_samples_seen_, numbers.Integral)):
            self.n_samples_seen_ = np.repeat(
                self.n_samples_seen_, X.shape[1]).astype(np.int64, copy=False)

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")

            if X.format == 'csr':
                X = X.tocsc()

            counts_nan = np.empty(X.shape[1])
            _isnan = np.isnan(X.data)

            start = X.indptr[0]
            for i, end in enumerate(X.indptr[1:]):
                counts_nan[i] = _isnan[start:end].sum()
                start = end

            if not hasattr(self, 'n_samples_seen_'):
                self.n_samples_seen_ = (
                        X.shape[0] - counts_nan).astype(np.int64, copy=False)

            if self.with_std:
                # First pass
                if not hasattr(self, 'scale_'):
                    self.mean_, self.var_ = mean_variance_axis(X, axis=0)

                # TODO
                """
                # Next passes
                else:
                    self.mean_, self.var_, self.n_samples_seen_ = \
                        incr_mean_variance_axis(X, axis=0,
                                                last_mean=self.mean_,
                                                last_var=self.var_,
                                                last_n=self.n_samples_seen_)
                """
            else:
                self.mean_ = None
                self.var_ = None
                if hasattr(self, 'scale_'):
                    self.n_samples_seen_ += X.shape[0] - counts_nan
        else:
            if not hasattr(self, 'n_samples_seen_'):
                self.n_samples_seen_ = np.zeros(X.shape[1], dtype=np.int64)

            # First pass
            if not hasattr(self, 'scale_'):
                self.mean_ = .0
                if self.with_std:
                    self.var_ = .0
                else:
                    self.var_ = None

            if not self.with_mean and not self.with_std:
                self.mean_ = None
                self.var_ = None
                self.n_samples_seen_ += X.shape[0] - np.isnan(X).sum(axis=0)
            else:
                self.mean_, self.var_, self.n_samples_seen_ = \
                    _incremental_mean_and_var(X, self.mean_, self.var_,
                                              self.n_samples_seen_)

        # for backward-compatibility, reduce n_samples_seen_ to an integer
        # if the number of samples is the same for each feature (i.e. no
        # missing values)
        ptp = np.amax(self.n_samples_seen_) - np.amin(self.n_samples_seen_)
        if ptp == 0:
            self.n_samples_seen_ = self.n_samples_seen_[0]
        del ptp

        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))
        else:
            self.scale_ = None

        return self

    def transform(self, X, copy=None) -> SparseCumlArray:
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Whether a forced copy will be triggered. If copy=False,
            a copy might be triggered by a conversion.
        """
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy

        X = self._validate_data(X, reset=False,
                                accept_sparse=['csr', 'csc'], copy=copy,
                                estimator=self, dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan')

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives.")
            if self.scale_ is not None:
                inplace_column_scale(X, 1 / self.scale_)
        else:
            if self.with_mean:
                X -= self.mean_
            if self.with_std:
                X /= self.scale_

        return X

    def inverse_transform(self, X, copy=None) -> SparseCumlArray:
        """Scale back the data to the original representation

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Whether a forced copy will be triggered. If copy=False,
            a copy might be triggered by a conversion.

        Returns
        -------
        X_tr : {array-like, sparse matrix}, shape [n_samples, n_features]
            Transformed array.
        """
        check_is_fitted(self)

        copy = copy if copy is not None else self.copy

        X = check_array(X, accept_sparse=['csr', 'csc'], copy=copy,
                        estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')

        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot uncenter sparse matrices: pass `with_mean=False` "
                    "instead See docstring for motivation and alternatives.")
            if not sparse.isspmatrix_csr(X):
                X = X.tocsr()
                copy = False
            if copy:
                X = X.copy()
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            X = np.asarray(X)
            if copy:
                X = X.copy()
            if self.with_std:
                X *= self.scale_
            if self.with_mean:
                X += self.mean_
        return X

    def _more_tags(self):
        return {'X_types_gpu': ['2darray', 'sparse'],
                'X_types': ['2darray', 'sparse'],
                'allow_nan': True}


class MaxAbsScaler(TransformerMixin, BaseEstimator):
    """Scale each feature by its maximum absolute value.

    This estimator scales and translates each feature individually such
    that the maximal absolute value of each feature in the
    training set will be 1.0. It does not shift/center the data, and
    thus does not destroy any sparsity.

    This scaler can also be applied to sparse CSR or CSC matrices.

    Parameters
    ----------
    copy : boolean, optional, default is True
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    Attributes
    ----------
    scale_ : ndarray, shape (n_features,)
        Per feature relative scaling of the data.

    max_abs_ : ndarray, shape (n_features,)
        Per feature maximum absolute value.

    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.

    Examples
    --------
    >>> from cuml.preprocessing import MaxAbsScaler
    >>> X = [[ 1., -1.,  2.],
    ...      [ 2.,  0.,  0.],
    ...      [ 0.,  1., -1.]]
    >>> transformer = MaxAbsScaler().fit(X)
    >>> transformer
    MaxAbsScaler()
    >>> transformer.transform(X)
    array([[ 0.5, -1. ,  1. ],
           [ 1. ,  0. ,  0. ],
           [ 0. ,  1. , -0.5]])

    See also
    --------
    maxabs_scale: Equivalent function without the estimator API.

    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.
    """

    scale_ = CumlArrayDescriptor()
    n_samples_seen_ = CumlArrayDescriptor()
    max_abs_ = CumlArrayDescriptor()

    @check_cupy8()
    @_deprecate_positional_args
    def __init__(self, *, copy=True):
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            self.scale_ = None
            self.n_samples_seen_ = None
            self.max_abs_ = None

    def get_param_names(self):
        return super().get_param_names() + [
            "copy"
        ]

    def fit(self, X, y=None) -> "MaxAbsScaler":
        """Compute the maximum absolute value to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None) -> "MaxAbsScaler":
        """
        Online computation of max absolute value of X for later scaling.

        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Transformer instance.
        """
        first_pass = not hasattr(self, 'n_samples_seen_')
        X = self._validate_data(X, reset=first_pass,
                                accept_sparse=('csr', 'csc'), estimator=self,
                                dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan')

        if sparse.issparse(X):
            mins, maxs = min_max_axis(X, axis=0, ignore_nan=True)
            max_abs = np.maximum(np.abs(mins), np.abs(maxs))
        else:
            max_abs = np.nanmax(np.abs(X), axis=0)

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            max_abs = np.maximum(self.max_abs_, max_abs)
            self.n_samples_seen_ += X.shape[0]

        self.max_abs_ = max_abs
        self.scale_ = _handle_zeros_in_scale(max_abs)
        return self

    def transform(self, X) -> SparseCumlArray:
        """Scale the data

        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data that should be scaled.
        """
        check_is_fitted(self)

        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')

        if sparse.issparse(X):
            inplace_column_scale(X, 1.0 / self.scale_)
        else:
            X /= self.scale_

        return X

    def inverse_transform(self, X) -> SparseCumlArray:
        """Scale back the data to the original representation

        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data that should be transformed back.
        """
        check_is_fitted(self)

        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')

        if sparse.issparse(X):
            inplace_column_scale(X, self.scale_)
        else:
            X *= self.scale_
        return X

    def _more_tags(self):
        return {'X_types_gpu': ['2darray', 'sparse'],
                'X_types': ['2darray', 'sparse'],
                'allow_nan': True}


@check_cupy8()
@_deprecate_positional_args
@api_return_generic(get_output_type=True)
def maxabs_scale(X, *, axis=0, copy=True):
    """Scale each feature to the [-1, 1] range without breaking the sparsity.

    This estimator scales each feature individually such
    that the maximal absolute value of each feature in the
    training set will be 1.0.

    This scaler can also be applied to sparse CSR or CSC matrices.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data.

    axis : int (0 by default)
        axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.

    copy : boolean, optional, default is True
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    See also
    --------
    MaxAbsScaler: Performs scaling to the [-1, 1] range using the``Transformer`` API

    Notes
    -----
    NaNs are treated as missing values: disregarded to compute the statistics,
    and maintained during the data transformation.
    """  # noqa
    # Unlike the scaler object, this function allows 1d input.

    # If copy is required, it will be done inside the scaler object.
    X = check_array(X, accept_sparse=('csr', 'csc'), copy=False,
                    ensure_2d=False, dtype=FLOAT_DTYPES,
                    force_all_finite='allow-nan')
    original_ndim = X.ndim

    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)

    s = MaxAbsScaler(copy=copy)
    if axis == 0:
        X = s.fit_transform(X)
    else:
        X = s.fit_transform(X.T).T

    if original_ndim == 1:
        X = X.ravel()

    return X


class RobustScaler(TransformerMixin, BaseEstimator):
    """Scale features using statistics that are robust to outliers.

    This Scaler removes the median and scales the data according to the
    quantile range (defaults to IQR: Interquartile Range). The IQR is the range
    between the 1st quartile (25th quantile) and the 3rd quartile (75th
    quantile).

    Centering and scaling happen independently on each feature by computing the
    relevant statistics on the samples in the training set. Median and
    interquartile range are then stored to be used on later data using the
    ``transform`` method.

    Standardization of a dataset is a common requirement for many machine
    learning estimators. Typically this is done by removing the mean and
    scaling to unit variance. However, outliers can often influence the sample
    mean / variance in a negative way. In such cases, the median and the
    interquartile range often give better results.

    Parameters
    ----------

    with_centering : boolean, default=True
        If True, center the data before scaling.
        This will cause ``transform`` to raise an exception when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_scaling : boolean, default=True
        If True, scale the data to interquartile range.

    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
        Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR
        Quantile range used to calculate ``scale_``.

    copy : boolean, optional, default=True
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    Attributes
    ----------
    center_ : array of floats
        The median value for each feature in the training set.

    scale_ : array of floats
        The (scaled) interquartile range for each feature in the training set.

    Examples
    --------
    >>> from cuml.preprocessing import RobustScaler
    >>> X = [[ 1., -2.,  2.],
    ...      [ -2.,  1.,  3.],
    ...      [ 4.,  1., -2.]]
    >>> transformer = RobustScaler().fit(X)
    >>> transformer
    RobustScaler()
    >>> transformer.transform(X)
    array([[ 0. , -2. ,  0. ],
           [-1. ,  0. ,  0.4],
           [ 1. ,  0. , -1.6]])

    See also
    --------

    robust_scale: Equivalent function without the estimator API.

    cuml.decomposition.PCA: Further removes the linear correlation across
        features with ``whiten=True``.

    """

    center_ = CumlArrayDescriptor()
    scale_ = CumlArrayDescriptor()

    @_deprecate_positional_args
    def __init__(self, *, with_centering=True, with_scaling=True,
                 quantile_range=(25.0, 75.0), copy=True):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy

    def get_param_names(self):
        return super().get_param_names() + [
            "with_centering",
            "with_scaling",
            "quantile_range",
            "copy"
        ]

    def fit(self, X, y=None) -> "RobustScaler":
        """Compute the median and quantiles to be used for scaling.

        Parameters
        ----------
        X : {array-like, CSC matrix}, shape [n_samples, n_features]
            The data used to compute the median and quantiles
            used for later scaling along the features axis.
        """
        # at fit, convert sparse matrices to csc for optimized computation of
        # the quantiles
        X = self._validate_data(X, accept_sparse='csc', estimator=self,
                                dtype=FLOAT_DTYPES,
                                force_all_finite='allow-nan')

        q_min, q_max = self.quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError("Invalid quantile range: %s" %
                             str(self.quantile_range))

        if self.with_centering:
            if sparse.issparse(X):
                raise ValueError(
                    "Cannot center sparse matrices: use `with_centering=False`"
                    " instead. See docstring for motivation and alternatives.")
            middle, is_odd = divmod(X.shape[0], 2)
            X_sorted = np.sort(X, axis=0)
            if is_odd:
                self.center_ = X_sorted[middle]
            else:
                elm1 = X_sorted[middle-1]
                elm2 = X_sorted[middle]
                self.center_ = (elm1 + elm2) / 2.
        else:
            self.center_ = None

        if self.with_scaling:
            quantiles = []
            for feature_idx in range(X.shape[1]):
                if sparse.issparse(X):
                    column_nnz_data = X.data[X.indptr[feature_idx]:
                                             X.indptr[feature_idx + 1]]
                    column_data = np.zeros(shape=X.shape[0], dtype=X.dtype)
                    column_data[:len(column_nnz_data)] = column_nnz_data
                else:
                    column_data = X[:, feature_idx]

                is_not_nan = ~np.isnan(column_data).astype(np.bool)
                column_data = column_data[is_not_nan]
                quantiles.append(np.percentile(column_data,
                                               self.quantile_range))

            quantiles = np.array(quantiles).T

            self.scale_ = quantiles[1] - quantiles[0]
            self.scale_ = _handle_zeros_in_scale(self.scale_, copy=False)
        else:
            self.scale_ = None

        return self

    def transform(self, X) -> SparseCumlArray:
        """Center and scale the data.

        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data used to scale along the specified axis.
        """
        check_is_fitted(self)

        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')

        if sparse.issparse(X):
            if self.with_scaling:
                inplace_column_scale(X, 1.0 / self.scale_)
        else:
            if self.with_centering:
                X -= self.center_
            if self.with_scaling:
                X /= self.scale_
        return X

    def inverse_transform(self, X) -> SparseCumlArray:
        """Scale back the data to the original representation

        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data used to scale along the specified axis.
        """
        check_is_fitted(self)

        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        estimator=self, dtype=FLOAT_DTYPES,
                        force_all_finite='allow-nan')

        if sparse.issparse(X):
            if self.with_scaling:
                inplace_column_scale(X, self.scale_)
        else:
            if self.with_scaling:
                X *= self.scale_
            if self.with_centering:
                X += self.center_
        return X

    def _more_tags(self):
        return {'X_types_gpu': ['2darray', 'sparse'],
                'X_types': ['2darray', 'sparse'],
                'allow_nan': True}


@_deprecate_positional_args
@api_return_generic(get_output_type=True)
def robust_scale(X, *, axis=0, with_centering=True, with_scaling=True,
                 quantile_range=(25.0, 75.0), copy=True):
    """
    Standardize a dataset along any axis

    Center to the median and component wise scale
    according to the interquartile range.

    Parameters
    ----------
    X : {array-like, sparse matrix}
        The data to center and scale.

    axis : int (0 by default)
        axis used to compute the medians and IQR along. If 0,
        independently scale each feature, otherwise (if 1) scale
        each sample.

    with_centering : boolean, True by default
        If True, center the data before scaling.

    with_scaling : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
        Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR
        Quantile range used to calculate ``scale_``.

    copy : boolean, optional, default is True
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    Notes
    -----
    This implementation will refuse to center sparse matrices
    since it would make them non-sparse and would potentially crash the
    program with memory exhaustion problems.

    Instead the caller is expected to either set explicitly
    `with_centering=False` (in that case, only variance scaling will be
    performed on the features of the CSR matrix) or to densify the matrix
    if he/she expects the materialized dense array to fit in memory.

    To avoid memory copy the caller should pass a CSR matrix.

    See also
    --------
    RobustScaler: Performs centering and scaling using the ``Transformer`` API

    """
    X = check_array(X, accept_sparse=('csr', 'csc'), copy=False,
                    ensure_2d=False, dtype=FLOAT_DTYPES,
                    force_all_finite='allow-nan')
    original_ndim = X.ndim

    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)

    with using_output_type("cupy"):
        s = RobustScaler(with_centering=with_centering,
                         with_scaling=with_scaling,
                         quantile_range=quantile_range,
                         copy=copy)
        if axis == 0:
            X = s.fit_transform(X)
        else:
            X = s.fit_transform(X.T).T

        if original_ndim == 1:
            X = X.ravel()

        return X


class PolynomialFeatures(TransformerMixin, BaseEstimator):
    """Generate polynomial and interaction features.

    Generate a new feature matrix consisting of all polynomial combinations
    of the features with degree less than or equal to the specified degree.
    For example, if an input sample is two dimensional and of the form
    [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    Parameters
    ----------
    degree : integer
        The degree of the polynomial features. Default = 2.

    interaction_only : boolean, default = False
        If true, only interaction features are produced: features that are
        products of at most ``degree`` *distinct* input features (so not
        ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).

    include_bias : boolean
        If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).

    order : str in {'C', 'F'}, default 'C'
        Order of output array in the dense case. 'F' order is faster to
        compute, but may slow down subsequent estimators.

    Examples
    --------
    >>> import numpy as np
    >>> from cuml.preprocessing import PolynomialFeatures
    >>> X = np.arange(6).reshape(3, 2)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> poly = PolynomialFeatures(2)
    >>> poly.fit_transform(X)
    array([[ 1.,  0.,  1.,  0.,  0.,  1.],
           [ 1.,  2.,  3.,  4.,  6.,  9.],
           [ 1.,  4.,  5., 16., 20., 25.]])
    >>> poly = PolynomialFeatures(interaction_only=True)
    >>> poly.fit_transform(X)
    array([[ 1.,  0.,  1.,  0.],
           [ 1.,  2.,  3.,  6.],
           [ 1.,  4.,  5., 20.]])

    Attributes
    ----------
    powers_ : array, shape (n_output_features, n_input_features)
        powers_[i, j] is the exponent of the jth input in the ith output.

    n_input_features_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of polynomial output features. The number of output
        features is computed by iterating over all suitably sized combinations
        of input features.

    Notes
    -----
    Be aware that the number of features in the output array scales
    polynomially in the number of features of the input array, and
    exponentially in the degree. High degrees can cause overfitting.
    """

    @check_cupy8()
    @_deprecate_positional_args
    def __init__(self, degree=2, *, interaction_only=False, include_bias=True,
                 order='C'):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.order = order

    def get_param_names(self):
        return super().get_param_names() + [
            "degree",
            "interaction_only",
            "include_bias",
            "order"
        ]

    @staticmethod
    @check_cupy8()
    def _combinations(n_features, degree, interaction_only, include_bias):
        comb = (combinations if interaction_only else combinations_w_r)
        start = int(not include_bias)
        return chain.from_iterable(comb(range(n_features), i)
                                   for i in range(start, degree + 1))

    @property
    def powers_(self):
        check_is_fitted(self)

        combinations = self._combinations(self.n_input_features_, self.degree,
                                          self.interaction_only,
                                          self.include_bias)
        return np.vstack([np.bincount(c, minlength=self.n_input_features_)
                          for c in combinations])

    def get_feature_names(self, input_features=None):
        """
        Return feature names for output features

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features

        """
        powers = self.powers_
        if input_features is None:
            input_features = ['x%d' % i for i in range(powers.shape[1])]
        feature_names = []
        for row in powers:
            inds = np.where(row)[0]
            if len(inds):
                name = " ".join("%s^%d" % (input_features[ind], exp)
                                if exp != 1 else input_features[ind]
                                for ind, exp in zip(inds, row[inds]))
            else:
                name = "1"
            feature_names.append(name)
        return feature_names

    def fit(self, X, y=None) -> "PolynomialFeatures":
        """
        Compute number of output features.


        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        n_samples, n_features = self._validate_data(
            X, accept_sparse=True).shape
        combinations = self._combinations(n_features, self.degree,
                                          self.interaction_only,
                                          self.include_bias)
        self.n_input_features_ = n_features
        self.n_output_features_ = sum(1 for _ in combinations)
        return self

    def transform(self, X) -> SparseCumlArray:
        """Transform data to polynomial features

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data to transform, row by row.

            Prefer CSR over CSC for sparse input (for speed), but CSC is
            required if the degree is 4 or higher. If the degree is less than
            4 and the input format is CSC, it will be converted to CSR, have
            its polynomial features generated, then converted back to CSC.

            If the degree is 2 or 3, the method described in "Leveraging
            Sparsity to Speed Up Polynomial Feature Expansions of CSR Matrices
            Using K-Simplex Numbers" by Andrew Nystrom and John Hughes is
            used, which is much faster than the method used on CSC input. For
            this reason, a CSC input will be converted to CSR, and the output
            will be converted back to CSC prior to being returned, hence the
            preference of CSR.

        Returns
        -------
        XP : {array-like, sparse matrix}, shape [n_samples, NP]
            The matrix of features, where NP is the number of polynomial
            features generated from the combination of inputs.
        """
        check_is_fitted(self)

        X = check_array(X, order='F', dtype=FLOAT_DTYPES,
                        accept_sparse=('csr', 'csc'))

        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        if sparse.isspmatrix_csr(X):
            if self.degree > 3:
                return self.transform(X.tocsc())  # TODO keep order
            to_stack = []
            if self.include_bias:
                bias = np.ones(shape=(n_samples, 1), dtype=X.dtype)
                to_stack.append(sparse.csr_matrix(bias))
            to_stack.append(X)
            for deg in range(2, self.degree+1):
                Xp_next = csr_polynomial_expansion(X, self.interaction_only,
                                                   deg)
                if Xp_next is None:
                    break
                to_stack.append(Xp_next)
            XP = sparse.hstack(to_stack, format='csr')
        elif sparse.isspmatrix_csc(X) and self.degree < 4:
            return self.transform(X.tocsr())  # TODO convert to csc, keep order
        else:
            if sparse.isspmatrix(X):
                combinations = self._combinations(n_features, self.degree,
                                                  self.interaction_only,
                                                  self.include_bias)
                columns = []
                for comb in combinations:
                    if comb:
                        out_col = 1
                        for col_idx in comb:
                            out_col = X[:, col_idx].multiply(out_col)
                        columns.append(out_col)
                    else:
                        bias = sparse.csc_matrix(np.ones((X.shape[0], 1)))
                        columns.append(bias)
                XP = sparse.hstack(columns, dtype=X.dtype).tocsc()
            else:
                XP = np.empty((n_samples, self.n_output_features_),
                              dtype=X.dtype, order=self.order)

                # What follows is a faster implementation of:
                # for i, comb in enumerate(combinations):
                #     XP[:, i] = X[:, comb].prod(1)
                # This implementation uses two optimisations.
                # First one is broadcasting,
                # multiply ([X1, ..., Xn], X1) -> [X1 X1, ..., Xn X1]
                # multiply ([X2, ..., Xn], X2) -> [X2 X2, ..., Xn X2]
                # ...
                # multiply ([X[:, start:end], X[:, start]) -> ...
                # Second optimisation happens for degrees >= 3.
                # Xi^3 is computed reusing previous computation:
                # Xi^3 = Xi^2 * Xi.

                if self.include_bias:
                    XP[:, 0] = 1
                    current_col = 1
                else:
                    current_col = 0

                # d = 0
                XP[:, current_col:current_col + n_features] = X
                index = list(range(current_col,
                                   current_col + n_features))
                current_col += n_features
                index.append(current_col)

                # d >= 1
                for _ in range(1, self.degree):
                    new_index = []
                    end = index[-1]
                    for feature_idx in range(n_features):
                        start = index[feature_idx]
                        new_index.append(current_col)
                        if self.interaction_only:
                            start += (index[feature_idx + 1] -
                                      index[feature_idx])
                        next_col = current_col + end - start
                        if next_col <= current_col:
                            break
                        # XP[:, start:end] are terms of degree d - 1
                        # that exclude feature #feature_idx.
                        np.multiply(XP[:, start:end],
                                    X[:, feature_idx:feature_idx + 1],
                                    out=XP[:, current_col:next_col],
                                    casting='no')
                        current_col = next_col

                    new_index.append(current_col)
                    index = new_index

        return XP  # TODO keep order

    def _more_tags(self):
        return {'X_types_gpu': ['2darray', 'sparse'],
                'X_types': ['2darray', 'sparse'],
                'allow_nan': True}


@check_cupy8()
@_deprecate_positional_args
@api_return_generic(get_output_type=True)
def normalize(X, norm='l2', *, axis=1, copy=True, return_norm=False):
    """Scale input vectors individually to unit norm (vector length).

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        The data to normalize, element by element.
        Please provide CSC matrix to normalize on axis 0,
        conversely provide CSR matrix to normalize on axis 1

    norm : 'l1', 'l2', or 'max', optional ('l2' by default)
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).

    axis : 0 or 1, optional (1 by default)
        axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.

    copy : boolean, optional, default True
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    return_norm : boolean, default False
        whether to return the computed norms

    Returns
    -------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        Normalized input X.

    norms : array, shape [n_samples] if axis=1 else [n_features]
        An array of norms along given axis for X.
        When X is sparse, a NotImplementedError will be raised
        for norm 'l1' or 'l2'.

    See also
    --------
    Normalizer: Performs normalization using the ``Transformer`` API
    """
    if norm not in ('l1', 'l2', 'max'):
        raise ValueError("'%s' is not a supported norm" % norm)

    if axis == 0:
        sparse_format = 'csc'
    elif axis == 1:
        sparse_format = 'csr'
    else:
        raise ValueError("'%d' is not a supported axis" % axis)

    X = check_array(X, accept_sparse=sparse_format, copy=copy,
                    estimator='the normalize function', dtype=FLOAT_DTYPES)

    if axis == 0:
        X = X.T

    if sparse.issparse(X):
        if return_norm and norm in ('l1', 'l2'):
            raise NotImplementedError("return_norm=True is not implemented "
                                      "for sparse matrices with norm 'l1' "
                                      "or norm 'l2'")
        if norm == 'l1':
            inplace_csr_row_normalize_l1(X)
        elif norm == 'l2':
            inplace_csr_row_normalize_l2(X)
        elif norm == 'max':
            mins, maxes = min_max_axis(X, 1)
            norms = np.maximum(abs(mins), maxes)
            norms_elementwise = norms.repeat(np.diff(X.indptr).tolist())
            mask = norms_elementwise != 0
            X.data[mask] /= norms_elementwise[mask]
    else:
        if norm == 'l1':
            norms = np.abs(X).sum(axis=1)
        elif norm == 'l2':
            norms = row_norms(X)
        elif norm == 'max':
            norms = np.max(abs(X), axis=1)
        norms = _handle_zeros_in_scale(norms, copy=False)
        X /= norms[:, np.newaxis]

    if axis == 0:
        X = X.T

    if return_norm:
        return X, norms
    else:
        return X


class Normalizer(TransformerMixin, BaseEstimator):
    """Normalize samples individually to unit norm.

    Each sample (i.e. each row of the data matrix) with at least one
    non zero component is rescaled independently of other samples so
    that its norm (l1, l2 or inf) equals one.

    This transformer is able to work both with dense numpy arrays and
    sparse matrix

    Scaling inputs to unit norms is a common operation for text
    classification or clustering for instance. For instance the dot
    product of two l2-normalized TF-IDF vectors is the cosine similarity
    of the vectors and is the base similarity metric for the Vector
    Space Model commonly used by the Information Retrieval community.

    Parameters
    ----------
    norm : 'l1', 'l2', or 'max', optional ('l2' by default)
        The norm to use to normalize each non zero sample. If norm='max'
        is used, values will be rescaled by the maximum of the absolute
        values.

    copy : boolean, optional, default True
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    Examples
    --------
    >>> from cuml.preprocessing import Normalizer
    >>> X = [[4, 1, 2, 2],
    ...      [1, 3, 9, 3],
    ...      [5, 7, 5, 1]]
    >>> transformer = Normalizer().fit(X)  # fit does nothing.
    >>> transformer
    Normalizer()
    >>> transformer.transform(X)
    array([[0.8, 0.2, 0.4, 0.4],
           [0.1, 0.3, 0.9, 0.3],
           [0.5, 0.7, 0.5, 0.1]])

    Notes
    -----
    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.


    See also
    --------
    normalize: Equivalent function without the estimator API.
    """

    @check_cupy8()
    @_deprecate_positional_args
    def __init__(self, norm='l2', *, copy=True):
        self.norm = norm
        self.copy = copy

    def fit(self, X, y=None) -> "Normalizer":
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : {array-like, CSR matrix}
        """
        self._validate_data(X, accept_sparse='csr')
        return self

    def transform(self, X, copy=None) -> SparseCumlArray:
        """Scale each non zero row of X to unit norm

        Parameters
        ----------
        X : {array-like, CSR matrix}, shape [n_samples, n_features]
            The data to normalize, row by row.
        copy : bool, optional (default: None)
            Whether a forced copy will be triggered. If copy=False,
            a copy might be triggered by a conversion.
        """
        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr')
        return normalize(X, norm=self.norm, axis=1, copy=copy)

    def _more_tags(self):
        return {'X_types_gpu': ['2darray', 'sparse'],
                'X_types': ['2darray', 'sparse'],
                'stateless': True}


@_deprecate_positional_args
@api_return_generic(get_output_type=True)
def binarize(X, *, threshold=0.0, copy=True):
    """Boolean thresholding of array-like or sparse matrix

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        The data to binarize, element by element.

    threshold : float, optional (0.0 by default)
        Feature values below or equal to this are replaced by 0, above it by 1.
        Threshold may not be less than 0 for operations on sparse matrices.

    copy : boolean, optional, default True
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    See also
    --------
    Binarizer: Performs binarization using the ``Transformer`` API
    """
    X = check_array(X, accept_sparse=['csr', 'csc'], copy=copy)
    if sparse.issparse(X):
        if threshold < 0:
            raise ValueError('Cannot binarize a sparse matrix with threshold '
                             '< 0')
        cond = X.data > threshold
        not_cond = np.logical_not(cond)
        X.data[cond] = 1
        X.data[not_cond] = 0
        X.eliminate_zeros()
    else:
        cond = X > threshold
        not_cond = np.logical_not(cond)
        X[cond] = 1
        X[not_cond] = 0
    return X


class Binarizer(TransformerMixin, BaseEstimator):
    """Binarize data (set feature values to 0 or 1) according to a threshold

    Values greater than the threshold map to 1, while values less than
    or equal to the threshold map to 0. With the default threshold of 0,
    only positive values map to 1.

    Binarization is a common operation on text count data where the
    analyst can decide to only consider the presence or absence of a
    feature rather than a quantified number of occurrences for instance.

    It can also be used as a pre-processing step for estimators that
    consider boolean random variables (e.g. modelled using the Bernoulli
    distribution in a Bayesian setting).

    Parameters
    ----------
    threshold : float, optional (0.0 by default)
        Feature values below or equal to this are replaced by 0, above it by 1.
        Threshold may not be less than 0 for operations on sparse matrices.

    copy : boolean, optional, default True
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    Examples
    --------
    >>> from cuml.preprocessing import Binarizer
    >>> X = [[ 1., -1.,  2.],
    ...      [ 2.,  0.,  0.],
    ...      [ 0.,  1., -1.]]
    >>> transformer = Binarizer().fit(X)  # fit does nothing.
    >>> transformer
    Binarizer()
    >>> transformer.transform(X)
    array([[1., 0., 1.],
           [1., 0., 0.],
           [0., 1., 0.]])

    Notes
    -----
    If the input is a sparse matrix, only the non-zero values are subject
    to update by the Binarizer class.

    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.

    See also
    --------
    binarize: Equivalent function without the estimator API.
    """

    @_deprecate_positional_args
    def __init__(self, *, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy

    def fit(self, X, y=None) -> "Binarizer":
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : {array-like, sparse matrix}
        """
        self._validate_data(X, accept_sparse=['csr', 'csc'])
        return self

    def transform(self, X, copy=None) -> SparseCumlArray:
        """Binarize each element of X

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data to binarize, element by element.

        copy : bool
            Whether a forced copy will be triggered. If copy=False,
            a copy might be triggered by a conversion.
        """
        copy = copy if copy is not None else self.copy
        return binarize(X, threshold=self.threshold, copy=copy)

    def _more_tags(self):
        return {'X_types_gpu': ['2darray', 'sparse'],
                'X_types': ['2darray', 'sparse'],
                'stateless': True}


@api_return_generic(get_output_type=True)
def add_dummy_feature(X, value=1.0):
    """Augment dataset with an additional dummy feature.

    This is useful for fitting an intercept term with implementations which
    cannot otherwise fit it directly.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        Data.

    value : float
        Value to use for the dummy feature.

    Returns
    -------

    X : {array, sparse matrix}, shape [n_samples, n_features + 1]
        Same data with dummy feature added as first column.

    Examples
    --------

    >>> from cuml.preprocessing import add_dummy_feature
    >>> add_dummy_feature([[0, 1], [1, 0]])
    array([[1., 0., 1.],
           [1., 1., 0.]])
    """
    X = check_array(X, accept_sparse=['csc', 'csr', 'coo'], dtype=FLOAT_DTYPES)
    n_samples, n_features = X.shape
    shape = (n_samples, n_features + 1)
    if sparse.issparse(X):
        if sparse.isspmatrix_coo(X):
            # Shift columns to the right.
            col = X.col + 1
            # Column indices of dummy feature are 0 everywhere.
            col = np.concatenate((np.zeros(n_samples), col))
            # Row indices of dummy feature are 0, ..., n_samples-1.
            row = np.concatenate((np.arange(n_samples), X.row))
            # Prepend the dummy feature n_samples times.
            data = np.concatenate((np.full(n_samples, value), X.data))
            X = sparse.coo_matrix((data, (row, col)), shape)
            return X
        elif sparse.isspmatrix_csc(X):
            # Shift index pointers since we need to add n_samples elements.
            indptr = X.indptr + n_samples
            # indptr[0] must be 0.
            indptr = np.concatenate((np.array([0]), indptr))
            # Row indices of dummy feature are 0, ..., n_samples-1.
            indices = np.concatenate((np.arange(n_samples), X.indices))
            # Prepend the dummy feature n_samples times.
            data = np.concatenate((np.full(n_samples, value), X.data))
            X = sparse.csc_matrix((data, indices, indptr), shape)
            return X
        else:
            klass = X.__class__
            with using_output_type('cupy'):
                res = add_dummy_feature(X.tocoo(), value)
            X = klass(res)
            return X
    else:
        X = np.hstack((np.full((n_samples, 1), value), X))
        return X
