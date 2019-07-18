#
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

import cupy as cp
import cudf
import numpy as np
import warnings

from cuml.thirdparty._sklearn._utils import (
    check_array, _safe_accumulator_op)


FLOAT_DTYPES = (cp.float64, cp.float32, cp.float16)


__all__ = [
    'Binarizer',
    'KernelCenterer',
    'MinMaxScaler',
    'MaxAbsScaler',
    'Normalizer',
    # 'OneHotEncoder',
    'RobustScaler',
    'StandardScaler',
    # 'QuantileTransformer',
    # 'PowerTransformer',
    'add_dummy_feature',
    'binarize',
    'normalize',
    'scale',
    'robust_scale',
    'maxabs_scale',
    'minmax_scale',
    # 'quantile_transform',
    # 'power_transform',
]


def check_fitted(self, attributes):
    ''' Check if the scaler already fitted
    '''
    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]
    if not all([hasattr(self, attr) for attr in attributes]):
        raise RuntimeError("Scaler must first be fitted")


def to_cupy(X):
    ''' convert input to cupy, and return info about input at the same time

    Parameters
    ----------
    X : cudf.DataFrame, cudf.Series or cupy array
        The data to be converted

    Return
    ------
    X: converted input
    input_info: dictionary
        It contains key: 'type', 'dim' and 'name_or_columns'(optional)
    '''
    # TODO: accept numba cuda array input
    input_info = {}
    if isinstance(X, cp.ndarray):
        input_dim = len(X.shape)
        input_type = 'cupy'
    elif isinstance(X, cudf.Series):
        input_dim = 1
        input_type = 'cudf'
        if hasattr(X, 'name'):
            series_name = X.name
            input_info['name_or_columns'] = series_name
        X = cp.array(X)
    elif isinstance(X, cudf.DataFrame):
        input_dim = 2
        input_type = 'cudf'
        if hasattr(X, 'columns'):
            columns = X.columns
            input_info['name_or_columns'] = columns
        X = cp.array(X.as_gpu_matrix())
    else:
        raise TypeError('Input should be cupy array or cudf.DataFrame '
                        + 'or cudf.Series')
    input_info['type'] = input_type
    input_info['dim'] = input_dim
    return X, input_info


def to_orig_type(X, input_info, dim=None, add_dummy_feature=False):
    ''' convert X to original datatype
    '''
    if input_info['type'] == 'cupy':
        return X
    elif input_info['type'] == 'cudf':
        # check if output dim is explicitly set(to be different with input dim)
        if dim is None:
            dim = input_info['dim']
        if dim == 1:
            X = cudf.from_dlpack(X.toDlpack())
            if 'name_or_columns' in input_info:     # restore Series name
                X.name = input_info['name_or_columns']
        elif dim == 2:
            X = cudf.DataFrame.from_gpu_matrix(X)
            if 'name_or_columns' in input_info:
                # new column will be added in add_dummy_feature()
                if add_dummy_feature:
                    X.columns = (input_info['name_or_columns']
                                 .insert(0, 'dummy_feature'))
                else:
                    X.columns = input_info['name_or_columns']
        else:
            raise ValueError('dim is {}, while it should be 1 or 2'
                             .format(dim))

    else:
        raise TypeError('Input should be either cupy array or cudf')
    return X


def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if cp.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, cp.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


def row_norms(X, squared=False):
    ''' Row-wise (squared) Euclidean norm of X.
    Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
    matrices and does not create an X.shape-sized temporary.
    Performs no input validation.
    Parameters
    ----------
    X : cupy array
        The input array
    squared : bool, optional (default = False)
        If True, return squared norms.
    Returns
    -------
    array_like
        The row-wise (squared) Euclidean norm of X.
    '''
    norms = cp.einsum('ij,ij->i', X, X)
    if not squared:
        cp.sqrt(norms, norms)
    return norms


def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
    ''' Calculate mean update and a Youngs and Cramer variance update.
    last_mean and last_variance are statistics computed at the last step by the
    function. Both must be initialized to 0.0. In case no scaling is required
    last_variance can be None. The mean is always required and returned because
    necessary for the calculation of the variance. last_n_samples_seen is the
    number of samples encountered until now.
    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to use for variance update
    last_mean : cupy array, shape: (n_features,)
    last_variance : array-like, shape: (n_features,)
    last_sample_count : array-like, shape (n_features,)
    Returns
    -------
    updated_mean : array, shape (n_features,)
    updated_variance : array, shape (n_features,)
        If None, only mean is computed
    updated_sample_count : array, shape (n_features,)
    Notes
    -----
    Input cannot contain NaN
    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247s
    '''
    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count
    new_sum = _safe_accumulator_op(cp.sum, X, axis=0)

    new_sample_count = cp.sum(~cp.isnan(X), axis=0)
    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        new_unnormalized_variance = (
            _safe_accumulator_op(cp.var, X, axis=0) * new_sample_count)
        last_unnormalized_variance = last_variance * last_sample_count

        last_over_new_count = last_sample_count / new_sample_count
        updated_unnormalized_variance = (
            last_unnormalized_variance + new_unnormalized_variance +
            last_over_new_count / updated_sample_count *
            (last_sum / last_over_new_count - new_sum) ** 2)

        zeros = last_sample_count == 0
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count


def scale(X, axis=0, with_mean=True, with_std=True, copy=True):
    ''' Standardize a dataset along any axis
    Center to the mean and component wise scale to unit variance.

    Parameters
    ----------
    X : array-like. The data to center and scale.
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
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a cupy).
    '''
    X, input_info = to_cupy(X)
    X = check_array(X, copy=copy, ensure_2d=False,
                    estimator='the scale function', dtype=FLOAT_DTYPES,
                    force_all_finite=True)

    if with_mean:
        mean_ = cp.mean(X, axis)
    if with_std:
        scale_ = cp.std(X, axis)
    # Xr is a view on the original array that enables easy use of
    # broadcasting on the axis in which we are interested in
    Xr = cp.rollaxis(X, axis)
    if with_mean:
        Xr -= mean_
        mean_1 = cp.mean(Xr, axis=0)
        # Verify that mean_1 is 'close to zero'. If X contains very
        # large values, mean_1 can also be very large, due to a lack of
        # precision of mean_. In this case, a pre-scaling of the
        # concerned feature is efficient, for instance by its mean or
        # maximum.
        if not cp.allclose(mean_1, 0):
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
            mean_2 = cp.mean(Xr, axis=0)
            # If mean_2 is not 'close to zero', it comes from the fact that
            # scale_ is very small so that mean_2 = mean_1/scale_ > 0, even
            # if mean_1 was close to zero. The problem is thus essentially
            # due to the lack of precision of mean_. A solution is then to
            # subtract the mean again:
            if not cp.allclose(mean_2, 0):
                warnings.warn("Numerical issues were encountered "
                              "when scaling the data "
                              "and might not be solved. The standard "
                              "deviation of the data is probably "
                              "very close to 0. ")
                Xr -= mean_2
    X = to_orig_type(X, input_info)
    return X


class TransformerMixin:
    '''Mixin class for transformers'''

    def fit_transform(self, X, y=None, **fit_params):
        '''Fit to data, then transform it.
        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.
        Parameters
        ----------
        X : cupy array of shape [n_samples, n_features]
            Training set.
        y : cupy array of shape [n_samples]
            Target values.
        Returns
        -------
        X_new : cupy array of shape [n_samples, n_features_new]
            Transformed array.
        '''
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)


class MinMaxScaler(TransformerMixin):
    ''' Transforms features by scaling each feature to a given range.
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.
    The transformation is given by::
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min
    where min, max = feature_range.
    The transformation is calculated as::
        X_scaled = scale * X + min - X.min(axis=0) * scale
        where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))
    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    copy : boolean, optional, default True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a cupy array).
    Attributes
    ----------
    min_ : ndarray, shape (n_features,)
        Per feature adjustment for minimum. Equivalent to
        ``min - X.min(axis=0) * self.scale_``
    scale_ : ndarray, shape (n_features,)
        Per feature relative scaling of the data. Equivalent to
        ``(max - min) / (X.max(axis=0) - X.min(axis=0))``
           *scale_* attribute.
    data_min_ : ndarray, shape (n_features,)
        Per feature minimum seen in the data
           *data_min_*
    data_max_ : ndarray, shape (n_features,)
        Per feature maximum seen in the data
           *data_max_*
    data_range_ : ndarray, shape (n_features,)
        Per feature range ``(data_max_ - data_min_)`` seen in the data
           *data_range_*
    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.preprocessing import MinMaxScaler
    >>> data = cp.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])
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

    Notes
    -----
    Currently does not support sparse matrix or input containing NaNs
    '''
    def __init__(self, feature_range=(0, 1), copy=True):
        self.feature_range = feature_range
        self.copy = copy

    def _reset(self):
        '''Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        '''

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X, y=None):
        '''Compute the minimum and maximum to be used for later scaling.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        '''

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        '''Online computation of min and max on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when `fit` is not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.
        Parameters
        ----------
        X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        '''
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))
        X, _ = to_cupy(X)
        X = check_array(X, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES)

        data_min = cp.min(X, axis=0)
        data_max = cp.max(X, axis=0)

        # First pass
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = X.shape[0]
        # Next steps
        else:
            data_min = cp.minimum(self.data_min_, data_min)
            data_max = cp.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = ((feature_range[1] - feature_range[0]) /
                       _handle_zeros_in_scale(data_range))
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        '''Scaling features of X according to feature_range.
        Parameters
        ----------
        X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
            Input data that will be transformed.
        '''
        check_fitted(self, 'scale_')
        X, input_info = to_cupy(X)
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)

        X *= self.scale_
        X += self.min_
        X = to_orig_type(X, input_info)
        return X

    def inverse_transform(self, X):
        '''Undo the scaling of X according to feature_range.
        Parameters
        ----------
        X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
            Input data that will be transformed. It cannot be sparse.
        '''
        check_fitted(self, 'scale_')
        X, input_info = to_cupy(X)
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)

        X -= self.min_
        X /= self.scale_
        X = to_orig_type(X, input_info)
        return X


def minmax_scale(X, feature_range=(0, 1), axis=0, copy=True):
    ''' Transforms features by scaling each feature to a given range.
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
    X : cupy.ndarray or cudf.DataFrame, shape (n_samples, n_features)
        The data.
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    axis : int (0 by default)
        axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.
    copy : boolean, optional, default is True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a cupy array).

    Notes
    -----
    Currently does not support sparse matrix or input containing NaNs
    '''
    # Unlike the scaler object, this function allows 1d input.
    # If copy is required, it will be done inside the scaler object.
    X, input_info = to_cupy(X)
    X = check_array(X, copy=False, ensure_2d=False, dtype=FLOAT_DTYPES)
    original_ndim = X.ndim

    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)

    s = MinMaxScaler(feature_range=feature_range, copy=copy)
    if axis == 0:
        X = s.fit_transform(X)
    else:
        X = s.fit_transform(X.T).T

    if original_ndim == 1:
        X = X.ravel()

    X = to_orig_type(X, input_info)
    return X


class StandardScaler(TransformerMixin):
    ''' Standardize features by removing the mean and scaling to unit variance
    The standard score of a sample `x` is calculated as:
        z = (x - u) / s
    where `u` is the mean of the training samples or zero if `with_mean=False`,
    and `s` is the standard deviation of the training samples or one if
    `with_std=False`.
    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using the
    `transform` method.
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
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a CuPy array, a copy may still be
        returned.
    with_mean : boolean, True by default
        If True, center the data before scaling.
    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).
    Attributes
    ----------
    scale_ : ndarray or None, shape (n_features,)
        Per feature relative scaling of the data. This is calculated using
        `cp.sqrt(var_)`. Equal to ``None`` when ``with_std=False``.
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
    >>> import cupy as cp
    >>> from cuml.preprocessing import StandardScaler
    >>> data = cp.array([[0, 0], [0, 0], [1, 1], [1, 1]])
    >>> scaler = StandardScaler()
    >>> print(scaler.fit(data))
    StandardScaler()
    >>> print(scaler.mean_)
    array([0.5 0.5])
    >>> print(scaler.transform(data))
    array([[-1. -1.]
           [-1. -1.]
           [ 1.  1.]
           [ 1.  1.]])
    >>> print(scaler.transform([[2, 2]]))
    array([[3. 3.]])
    '''
    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        '''Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        '''
        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, y=None):
        '''Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        '''
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        '''Online computation of mean and std on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when `fit` is not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.
        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:
        Parameters
        ----------
        X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        '''
        X, _ = to_cupy(X)
        X = check_array(X, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES)

        # Even in the case of `with_mean=False`, we update the mean anyway
        # This is needed for the incremental computation of the var
        # See incr_mean_variance_axis and _incremental_mean_variance_axis

        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = cp.zeros(X.shape[1], dtype=cp.int64)

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
            self.n_samples_seen_ += X.shape[0] - cp.isnan(X).sum(axis=0)
        else:
            self.mean_, self.var_, self.n_samples_seen_ = \
                _incremental_mean_and_var(X, self.mean_, self.var_,
                                          self.n_samples_seen_)

        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(cp.sqrt(self.var_))
        else:
            self.scale_ = None

        return self

    def transform(self, X, copy=None):
        '''Perform standardization by centering and scaling
        Parameters
        ----------
        X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        '''
        check_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        X, input_info = to_cupy(X)
        X = check_array(X, copy=copy, estimator=self, dtype=FLOAT_DTYPES)

        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.scale_
        X = to_orig_type(X, input_info)
        return X

    def inverse_transform(self, X, copy=None):
        '''Scale back the data to the original representation
        Parameters
        ----------
        X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        Returns
        -------
        X_tr : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
            Transformed array.
        '''
        check_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy

        X, input_info = to_cupy(X)
        if copy:
            X = X.copy()
        if self.with_std:
            X *= self.scale_
        if self.with_mean:
            X += self.mean_
        X = to_orig_type(X, input_info)
        return X


class MaxAbsScaler(TransformerMixin):
    ''' Scale each feature by its maximum absolute value.
    This estimator scales and translates each feature individually such
    that the maximal absolute value of each feature in the
    training set will be 1.0.

    Parameters
    ----------
    copy : boolean, optional, default is True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a cupy array).
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
    >>> import cupy as cp
    >>> from cuml.preprocessing import MaxAbsScaler
    >>> X = cp.array([[ 1., -1.,  2.],
    ...               [ 2.,  0.,  0.],
    ...               [ 0.,  1., -1.]])
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
    Currently does not support sparse matrix or input containing NaNs
    '''

    def __init__(self, copy=True):
        self.copy = copy

    def _reset(self):
        '''Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        '''

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.max_abs_

    def fit(self, X, y=None):
        '''Compute the maximum absolute value to be used for later scaling.
        Parameters
        ----------
        X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        '''

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        '''Online computation of max absolute value of X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when `fit` is not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.
        Parameters
        ----------
        X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        '''
        X, _ = to_cupy(X)
        X = check_array(X, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES)

        max_abs = cp.max(cp.abs(X), axis=0)

        # First pass
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = X.shape[0]
        # Next passes
        else:
            max_abs = cp.maximum(self.max_abs_, max_abs)
            self.n_samples_seen_ += X.shape[0]

        self.max_abs_ = max_abs
        self.scale_ = _handle_zeros_in_scale(max_abs)
        return self

    def transform(self, X):
        '''Scale the data
        Parameters
        ----------
        X : cupy.ndarray or cudf.DataFrame
            The data that should be scaled.
        '''
        check_fitted(self, 'scale_')
        X, input_info = to_cupy(X)
        X = check_array(X, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES)

        X /= self.scale_
        X = to_orig_type(X, input_info)
        return X

    def inverse_transform(self, X):
        '''Scale back the data to the original representation
        Parameters
        ----------
        X : cupy.ndarray or cudf.DataFrame
            The data that should be transformed back.
        '''
        check_fitted(self, 'scale_')
        X, input_info = to_cupy(X)
        X = check_array(X, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES)

        X *= self.scale_
        X = to_orig_type(X, input_info)
        return X


def maxabs_scale(X, axis=0, copy=True):
    ''' Scale each feature to the [-1, 1] range without breaking the sparsity.
    This estimator scales each feature individually such
    that the maximal absolute value of each feature in the
    training set will be 1.0.

    Parameters
    ----------
    X : cupy.ndarray or cudf.DataFrame, shape (n_samples, n_features)
        The data.
    axis : int (0 by default)
        axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.
    copy : boolean, optional, default is True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a cupy array).

    Notes
    -----
    Currently does not support sparse matrix or input containing NaNs
    '''
    X, input_info = to_cupy(X)
    # Unlike the scaler object, this function allows 1d input.
    # If copy is required, it will be done inside the scaler object.
    X = check_array(X, copy=False, ensure_2d=False, dtype=FLOAT_DTYPES)
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

    X = to_orig_type(X, input_info)
    return X


class RobustScaler(TransformerMixin):
    ''' Scale features using statistics that are robust to outliers.
    This Scaler removes the median and scales the data according to
    the quantile range (defaults to IQR: Interquartile Range).
    The IQR is the range between the 1st quartile (25th quantile)
    and the 3rd quartile (75th quantile).
    Centering and scaling happen independently on each feature by
    computing the relevant statistics on the samples in the training
    set. Median and interquartile range are then stored to be used on
    later data using the ``transform`` method.
    Standardization of a dataset is a common requirement for many
    machine learning estimators. Typically this is done by removing the mean
    and scaling to unit variance. However, outliers can often influence the
    sample mean / variance in a negative way. In such cases, the median and
    the interquartile range often give better results.

    Parameters
    ----------
    with_centering : boolean, True by default
        If True, center the data before scaling.
        This will cause ``transform`` to raise an exception when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.
    with_scaling : boolean, True by default
        If True, scale the data to interquartile range.
    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
        Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR
        Quantile range used to calculate ``scale_``.
    copy : boolean, optional, default is True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a CuPy array, a copy may still be
        returned.
    Attributes
    ----------
    center_ : array of floats
        The median value for each feature in the training set.
    scale_ : array of floats
        The (scaled) interquartile range for each feature in the training set.
    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.preprocessing import RobustScaler
    >>> X = cp.array([[ 1., -2.,  2.],
    ...               [ -2.,  1.,  3.],
    ...               [ 4.,  1., -2.]])
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

    Notes
    -----
    Currently does not support sparse matrix or input containing NaNs
    '''
    def __init__(self, with_centering=True, with_scaling=True,
                 quantile_range=(25.0, 75.0), copy=True):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy

    def fit(self, X, y=None):
        '''Compute the median and quantiles to be used for scaling.
        Parameters
        ----------
        X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
            The data used to compute the median and quantiles
            used for later scaling along the features axis.
        '''
        X, _ = to_cupy(X)
        X = check_array(X, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES)

        q_min, q_max = self.quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError("Invalid quantile range: %s" %
                             str(self.quantile_range))

        if self.with_centering:
            # As CuPy and cuDF doesn't support median() yet, we have to convert
            # to numpy 
            self.center_ = cp.array(np.median(cp.asnumpy(X), axis=0))
        else:
            self.center_ = None

        if self.with_scaling:
            quantiles = None
            for feature_idx in range(X.shape[1]):
                column_data = X[:, feature_idx]

                if quantiles is None:
                    quantiles = cp.percentile(
                        column_data, self.quantile_range).reshape(-1, 1)
                else:
                    quantiles = cp.hstack([quantiles,
                                           cp
                                           .percentile(column_data,
                                                       self.quantile_range)
                                           .reshape(-1, 1)])

            self.scale_ = quantiles[1] - quantiles[0]
            self.scale_ = _handle_zeros_in_scale(self.scale_, copy=False)
        else:
            self.scale_ = None

        return self

    def transform(self, X):
        '''Center and scale the data.
        Parameters
        ----------
        X : cupy.ndarray or cudf.DataFrame
            The data used to scale along the specified axis.
        '''
        check_fitted(self, ['center_', 'scale_'])
        X, input_info = to_cupy(X)
        X = check_array(X, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES)

        if self.with_centering:
            X -= self.center_
        if self.with_scaling:
            X /= self.scale_
        X = to_orig_type(X, input_info)
        return X

    def inverse_transform(self, X):
        '''Scale back the data to the original representation
        Parameters
        ----------
        X : array-like
            The data used to scale along the specified axis.
        '''
        check_fitted(self, ['center_', 'scale_'])
        X, input_info = to_cupy(X)
        X = check_array(X, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES)

        if self.with_scaling:
            X *= self.scale_
        if self.with_centering:
            X += self.center_
        X = to_orig_type(X, input_info)
        return X


def robust_scale(X, axis=0, with_centering=True, with_scaling=True,
                 quantile_range=(25.0, 75.0), copy=True):
    ''' Standardize a dataset along any axis
    Center to the median and component wise scale
    according to the interquartile range.

    Parameters
    ----------
    X : cupy.ndarray or cudf.DataFrame
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
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a CuPy array).

    Notes
    -----
    Currently does not support sparse matrix or input containing NaNs
    '''
    X, input_info = to_cupy(X)
    X = check_array(X, copy=False, ensure_2d=False, dtype=FLOAT_DTYPES)
    original_ndim = X.ndim

    if original_ndim == 1:
        X = X.reshape(X.shape[0], 1)

    s = RobustScaler(with_centering=with_centering, with_scaling=with_scaling,
                     quantile_range=quantile_range, copy=copy)
    if axis == 0:
        X = s.fit_transform(X)
    else:
        X = s.fit_transform(X.T).T

    if original_ndim == 1:
        X = X.ravel()

    X = to_orig_type(X, input_info)
    return X


def normalize(X, norm='l2', axis=1, copy=True, return_norm=False):
    ''' Scale input vectors individually to unit norm (vector length).

    Parameters
    ----------
    X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
        The data to normalize, element by element.
    norm : 'l1', 'l2', or 'max', optional ('l2' by default)
        The norm to use to normalize each non zero sample (or each non-zero
        feature if axis is 0).
    axis : 0 or 1, optional (1 by default)
        axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.
    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a CuPy array).
    return_norm : boolean, default False
        whether to return the computed norms
    Returns
    -------
    X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
        Normalized input X.
    norms : array, shape [n_samples] if axis=1 else [n_features]
        An array of norms along given axis for X.
        When X is sparse, a NotImplementedError will be raised
        for norm 'l1' or 'l2'.

    Notes
    -----
    Currently does not support sparse matrix or input containing NaNs
    '''
    if norm not in ('l1', 'l2', 'max'):
        raise ValueError("'%s' is not a supported norm" % norm)

    if axis != 0 and axis != 1:
        raise ValueError("'%d' is not a supported axis" % axis)

    X, input_info = to_cupy(X)
    X = check_array(X, copy=copy,
                    estimator='the normalize function', dtype=FLOAT_DTYPES)
    if axis == 0:
        X = X.T

    if norm == 'l1':
        norms = cp.abs(X).sum(axis=1)
    elif norm == 'l2':
        norms = row_norms(X)
    elif norm == 'max':
        norms = cp.max(X, axis=1)
    norms = _handle_zeros_in_scale(norms, copy=False)
    X /= norms[:, cp.newaxis]

    if axis == 0:
        X = X.T
    X = to_orig_type(X, input_info)
    norms = to_orig_type(norms, input_info, dim=1)
    if return_norm:
        return X, norms
    else:
        return X


class Normalizer(TransformerMixin):
    ''' Normalize samples individually to unit norm.
    Each sample (i.e. each row of the data matrix) with at least one
    non zero component is rescaled independently of other samples so
    that its norm (l1 or l2) equals one.
    Scaling inputs to unit norms is a common operation for text
    classification or clustering for instance. For instance the dot
    product of two l2-normalized TF-IDF vectors is the cosine similarity
    of the vectors and is the base similarity metric for the Vector
    Space Model commonly used by the Information Retrieval community.

    Parameters
    ----------
    norm : 'l1', 'l2', or 'max', optional ('l2' by default)
        The norm to use to normalize each non zero sample.
    copy : boolean, optional, default True
        set to False to perform inplace row normalization and avoid a
        copy (if the input is already a cupy array).
    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.preprocessing import Normalizer
    >>> X = cp.array([[4, 1, 2, 2],
    ...               [1, 3, 9, 3],
    ...               [5, 7, 5, 1]])
    >>> transformer = Normalizer().fit(X)  # fit does nothing.
    >>> transformer
    Normalizer()
    >>> transformer.transform(X)
    array([[0.8, 0.2, 0.4, 0.4],
           [0.1, 0.3, 0.9, 0.3],
           [0.5, 0.7, 0.5, 0.1]])
    Notes
    -----
    Currently does not support sparse matrix or input containing NaNs

    See also
    --------
    normalize: Equivalent function without the estimator API.
    '''
    def __init__(self, norm='l2', copy=True):
        self.norm = norm
        self.copy = copy

    def fit(self, X, y=None):
        '''Do nothing and return the estimator unchanged
        This method is just there to implement the usual API and hence
        work in pipelines.
        Parameters
        ----------
        X : array-like
        '''
        X, _ = to_cupy(X)
        check_array(X)
        return self

    def transform(self, X, copy=None):
        '''Scale each non zero row of X to unit norm
        Parameters
        ----------
        X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
            The data to normalize, row by row. scipy.sparse matrices should be
            in CSR format to avoid an un-necessary copy.
        copy : bool, optional (default: None)
            Copy the input X or not.
        '''
        copy = copy if copy is not None else self.copy
        X, input_info = to_cupy(X)
        X = check_array(X)
        return normalize(X, norm=self.norm, axis=1, copy=copy)


def binarize(X, threshold=0.0, copy=True):
    ''' Boolean thresholding of cupy.ndarray or cudf.DataFrame

    Parameters
    ----------
    X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
        The data to binarize, element by element.
    threshold : float, optional (0.0 by default)
        Feature values below or equal to this are replaced by 0, above it by 1.
    copy : boolean, optional, default True
        set to False to perform inplace binarization and avoid a copy
        (if the input is already a CuPy array).
    '''
    X, input_info = to_cupy(X)
    X = check_array(X, copy=copy)
    cond = X > threshold
    not_cond = cp.logical_not(cond)
    X[cond] = 1
    X[not_cond] = 0
    X = to_orig_type(X, input_info)
    return X


class Binarizer(TransformerMixin):
    ''' Binarize data (set feature values to 0 or 1) according to a threshold
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
    copy : boolean, optional, default True
        set to False to perform inplace binarization and avoid a copy (if
        the input is already a cupy array).
    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.preprocessing import Binarizer
    >>> X = cp.array([[ 1., -1.,  2.],
    ...               [ 2.,  0.,  0.],
    ...               [ 0.,  1., -1.]])
    >>> transformer = Binarizer().fit(X)  # fit does nothing.
    >>> transformer
    Binarizer()
    >>> transformer.transform(X)
    array([[1., 0., 1.],
           [1., 0., 0.],
           [0., 1., 0.]])
    Notes
    -----
    Currently does not support sparse matrix or input containing NaNs

    See also
    --------
    binarize: Equivalent function without the estimator API.
    '''
    def __init__(self, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy

    def fit(self, X, y=None):
        '''Do nothing and return the estimator unchanged
        This method is just there to implement the usual API and hence
        work in pipelines.
        Parameters
        ----------
        X : array-like
        '''
        X, _ = to_cupy(X)
        check_array(X)
        return self

    def transform(self, X, copy=None):
        '''Binarize each element of X
        Parameters
        ----------
        X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
            The data to binarize, element by element.
            scipy.sparse matrices should be in CSR format to avoid an
            un-necessary copy.
        copy : bool
            Copy the input X or not.
        '''
        copy = copy if copy is not None else self.copy
        return binarize(X, threshold=self.threshold, copy=copy)


class KernelCenterer(TransformerMixin):
    ''' Center a kernel matrix
    Let K(x, z) be a kernel defined by phi(x)^T phi(z), where phi is a
    function mapping x to a Hilbert space. KernelCenterer centers (i.e.,
    normalize to have zero mean) the data without explicitly computing phi(x).
    It is equivalent to centering phi(x) with
    cuml.preprocessing.StandardScaler(with_std=False).

    Notes
    -----
    Currently does not support sparse matrix or input containing NaNs
    '''
    def __init__(self):
        # Needed for backported inspect.signature compatibility with PyPy
        pass

    def fit(self, K, y=None):
        '''Fit KernelCenterer
        Parameters
        ----------
        K : cupy.ndarray or cudf.DataFrame of shape [n_samples, n_samples]
            Kernel matrix.
        Returns
        -------
        self : returns an instance of self.
        '''
        K, _ = to_cupy(K)
        K = check_array(K, dtype=FLOAT_DTYPES)
        n_samples = K.shape[0]
        self.K_fit_rows_ = cp.sum(K, axis=0) / n_samples
        self.K_fit_all_ = self.K_fit_rows_.sum() / n_samples
        return self

    def transform(self, K, copy=True):
        '''Center kernel matrix.
        Parameters
        ----------
        K : cupy.ndarray or cudf.DataFrame of shape [n_samples1, n_samples2]
            Kernel matrix.
        copy : boolean, optional, default True
            Set to False to perform inplace computation.
        Returns
        -------
        K_new: cupy.ndarray or cudf.DataFrame of shape [n_samples1, n_samples2]
        '''
        check_fitted(self, 'K_fit_all_')
        K, input_info = to_cupy(K)
        K = check_array(K, copy=copy, dtype=FLOAT_DTYPES)

        K_pred_cols = (cp.sum(K, axis=1) /
                       self.K_fit_rows_.shape[0])[:, cp.newaxis]

        K -= self.K_fit_rows_
        K -= K_pred_cols
        K += self.K_fit_all_
        K = to_orig_type(K, input_info)

        return K

    @property
    def _pairwise(self):
        return True


def add_dummy_feature(X, value=1.0):
    '''Augment dataset with an additional dummy feature.
    This is useful for fitting an intercept term with implementations which
    cannot otherwise fit it directly.
    Parameters
    ----------
    X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features]
        Data.
    value : float
        Value to use for the dummy feature.
    Returns
    -------
    X : cupy.ndarray or cudf.DataFrame, shape [n_samples, n_features + 1]
        Same data with dummy feature added as first column.
    Examples
    --------
    >>> import cupy as cp
    >>> from cumlpreprocessing. import add_dummy_feature
    >>> add_dummy_feature(cp.array([[0, 1], [1, 0]]))
    array([[1., 0., 1.],
           [1., 1., 0.]])
    '''

    # need to mod to_orig_type, since new column will be added
    X, input_info = to_cupy(X)
    X = check_array(X, dtype=FLOAT_DTYPES)
    n_samples, n_features = X.shape
    new_X = cp.hstack((cp.full((n_samples, 1), value), X))
    new_X = to_orig_type(new_X, input_info, add_dummy_feature=True)
    return new_X
