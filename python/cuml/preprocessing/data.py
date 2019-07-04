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

import numpy as np
import cudf
# from librmm_cffi import librmm


__all__ = [
    # 'Binarizer',
    # 'KernelCenterer',
    'MinMaxScaler',
    'MaxAbsScaler',
    # 'Normalizer',
    # 'OneHotEncoder',
    # 'RobustScaler',
    'StandardScaler',
    # 'QuantileTransformer',
    # 'PowerTransformer',
    # 'add_dummy_feature',
    # 'binarize',
    # 'normalize',
    # 'scale',
    # 'robust_scale',
    'maxabs_scale',
    'minmax_scale',
    # 'quantile_transform',
    # 'power_transform'
]


def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if isinstance(scale, int):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, cudf.Series):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale.masked_assign(1.0, scale == 0.0)
        return scale


def _check_input_type(inp):
    if not isinstance(inp, (cudf.Series, cudf.DataFrame)):
        raise TypeError('Input must be either cudf.Series or cudf.DataFrame')


def sub(df, ser):
    for i, col in enumerate(df.columns):
        df[col] = df[col] - ser[i]


def add(df, ser):
    for i, col in enumerate(df.columns):
        df[col] = df[col] + ser[i]


def mult(df, ser):
    for i, col in enumerate(df.columns):
        df[col] = df[col] * ser[i]


def div(df, ser):
    for i, col in enumerate(df.columns):
        df[col] = df[col] / ser[i]


class TransformerMixin:
    """Mixin class for all transformers in scikit-learn."""

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.
        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.
        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.
        y : numpy array of shape [n_samples]
            Target values.
        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)


class MinMaxScaler(TransformerMixin):
    def __init__(self, feature_range=(0, 1), copy=True):
        self.feature_range = feature_range
        self.copy = copy

    def _reset(self):
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X, y=None):
         """Compute the minimum and maximum to be used for later scaling.
         Parameters
         ----------
         X : array-like, shape [n_samples, n_features]
             The data used to compute the per-feature minimum and maximum
             used for later scaling along the features axis.
         """

         # Reset internal state before fitting
         self._reset()
         return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Online computation of min and max on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when `fit` is not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        """
        _check_input_type(X)

        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))

        # X = check_array(X, copy=self.copy,
        #                 estimator=self, dtype=FLOAT_DTYPES,
        #                 force_all_finite="allow-nan")

        data_min = X.min()
        data_max = X.max()

        # First pass
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = X.shape[0]
        # Next steps
        else:
            data_min = cudf.concat(
                [self.data_min_.to_frame(), data_min.to_frame()]).min()
            data_max = cudf.concat(
                [self.data_max_.to_frame(), data_max.to_frame()]).max()
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min    # cudf.Series
        self.scale_ = ((feature_range[1] - feature_range[0])
                       / _handle_zeros_in_scale(data_range))  # cudf.Series
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min           # cudf.Series
        self.data_max_ = data_max           # cudf.Series
        self.data_range_ = data_range
        return self

    def transform(self, X):
        """Scaling features of X according to feature_range.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """
        ############################
        # check_is_fitted(self, 'scale_')

        # X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
        #                 force_all_finite="allow-nan")

        mult(X, self.scale_)
        add(X, self.min_)
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed. It cannot be sparse.
        """
        ################################
        # check_is_fitted(self, 'scale_')

        # X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
        #                 force_all_finite="allow-nan")

        sub(X, self.min_)
        div(X, self.scale_)
        return X


def minmax_scale(X, feature_range=(0, 1), axis=0, copy=True):
    # Unlike the scaler object, this function allows 1d input.
    # If copy is required, it will be done inside the scaler object.

    # X = check_array(X, copy=False, ensure_2d=False,
    #                 dtype=FLOAT_DTYPES, force_all_finite='allow-nan')

    s = MinMaxScaler(feature_range=feature_range, copy=copy)

    if isinstance(X, cudf.Series):  # if is Series, then axis does not matter
        X = X.to_frame()
        X = s.fit_transform(X)
        return X.T.iloc[0]
    else:
        if axis == 0:
            X = s.fit_transform(X)
        else:
            X = s.fit_transform(X.T).T
        return X



def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
    """Calculate mean update and a Youngs and Cramer variance update.
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
    last_mean : array-like, shape: (n_features,)
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
    NaNs are ignored during the algorithm.
    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247
    Also, see the sparse implementation of this in
    `utils.sparsefuncs.incr_mean_variance_axis` and
    `utils.sparsefuncs_fast.incr_mean_variance_axis0`
    """
    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count
    new_sum = X.sum()                   # cudf.Series

    new_sample_count = X.count()        # cudf.Series
    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        new_unnormalized_variance = X.var() * new_sample_count
        last_unnormalized_variance = last_variance * last_sample_count

        # with np.errstate(divide='ignore', invalid='ignore'):
        last_over_new_count = last_sample_count / new_sample_count
        updated_unnormalized_variance = (
            last_unnormalized_variance + new_unnormalized_variance +
            last_over_new_count / updated_sample_count *
            (last_sum / last_over_new_count - new_sum) ** 2)

        zeros = last_sample_count == 0
        ###################   Series does not support item assignment
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        updated_variance = updated_unnormalized_variance / updated_sample_count

    return updated_mean, updated_variance, updated_sample_count



class StandardScaler(TransformerMixin):
    def __init__(self, copy=True, with_mean=True, with_std=True):
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
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, y=None):
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

    def partial_fit(self, X, y=None):
        """Online computation of mean and std on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when `fit` is not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.
        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        """
        # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
        #                 estimator=self, dtype=FLOAT_DTYPES,
        #                 force_all_finite='allow-nan')

        # Even in the case of `with_mean=False`, we update the mean anyway
        # This is needed for the incremental computation of the var
        # See incr_mean_variance_axis and _incremental_mean_variance_axis

        # if n_samples_seen_ is an integer (i.e. no missing values), we need to
        # transform it to a NumPy array of shape (n_features,) required by
        # incr_mean_variance_axis and _incremental_variance_axis
        #########################################################3
        if (hasattr(self, 'n_samples_seen_') and
                isinstance(self.n_samples_seen_, (int, np.integer))):
            self.n_samples_seen_ = np.repeat(
                self.n_samples_seen_, X.shape[1]).astype(np.int64, copy=False)
        ######################################################
        

        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = cudf.Series(np.zeros(X.shape[1],
                                                        dtype=np.int64))

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
            self.n_samples_seen_ += X.count()
        else:
            self.mean_, self.var_, self.n_samples_seen_ = \
                _incremental_mean_and_var(X, self.mean_, self.var_,
                                          self.n_samples_seen_)

        # for backward-compatibility, reduce n_samples_seen_ to an integer
        # if the number of samples is the same for each feature (i.e. no
        # missing values)
        if np.ptp(self.n_samples_seen_) == 0:
            self.n_samples_seen_ = self.n_samples_seen_[0]

        ###############################################
        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))
        else:
            self.scale_ = None

        return self

    def transform(self, X, copy=None):
        """Perform standardization by centering and scaling
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        # check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        # X = check_array(X, accept_sparse='csr', copy=copy,
        #                 estimator=self, dtype=FLOAT_DTYPES,
        #                 force_all_finite='allow-nan')

        if self.with_mean:
            sub(X, self.mean_)
        if self.with_std:
            div(X, self.scale_)
        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Transformed array.
        """
        # check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy

        if copy:
            X = X.copy()
        if self.with_std:
            mult(X, self.scale_)
        if self.with_mean:
            add(X, self.mean_)
        return X


class MaxAbsScaler(TransformerMixin):
    """Scale each feature by its maximum absolute value.
    This estimator scales and translates each feature individually such
    that the maximal absolute value of each feature in the
    training set will be 1.0. It does not shift/center the data, and
    thus does not destroy any sparsity.
    Parameters
    ----------
    copy : boolean, optional, default is True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a numpy array).
    Attributes
    ----------
    scale_ : ndarray, shape (n_features,)
        Per feature relative scaling of the data.
           *scale_* attribute.
    max_abs_ : ndarray, shape (n_features,)
        Per feature maximum absolute value.
    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.
    Examples
    --------
    >>> from sklearn.preprocessing import MaxAbsScaler
    >>> X = [[ 1., -1.,  2.],
    ...      [ 2.,  0.,  0.],
    ...      [ 0.,  1., -1.]]
    >>> transformer = MaxAbsScaler().fit(X)
    >>> transformer
    MaxAbsScaler(copy=True)
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
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """

    def __init__(self, copy=True):
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.max_abs_

    def fit(self, X, y=None):
        """Compute the maximum absolute value to be used for later scaling.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Online computation of max absolute value of X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when `fit` is not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        """
        # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
        #                 estimator=self, dtype=FLOAT_DTYPES,
        #                 force_all_finite='allow-nan')

        max_abs = X.abs().max()

        # First pass
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = X.shape[0]
        # Next passes
        else:
            max_abs = cudf.concat(
                [self.max_abs_.to_frame(), max_abs.to_frame()]).max()
            self.n_samples_seen_ += X.shape[0]

        self.max_abs_ = max_abs
        self.scale_ = _handle_zeros_in_scale(max_abs)
        return self

    def transform(self, X):
        """Scale the data
        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data that should be scaled.
        """
        # check_is_fitted(self, 'scale_')
        # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
        #                 estimator=self, dtype=FLOAT_DTYPES,
        #                 force_all_finite='allow-nan')

        div(X, self.scale_)
        return X

    def inverse_transform(self, X):
        """Scale back the data to the original representation
        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data that should be transformed back.
        """
        # check_is_fitted(self, 'scale_')
        # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
        #                 estimator=self, dtype=FLOAT_DTYPES,
        #                 force_all_finite='allow-nan')

        mult(X, self.scale_)
        return X



def maxabs_scale(X, axis=0, copy=True):
    """Scale each feature to the [-1, 1] range without breaking the sparsity.
    This estimator scales each feature individually such
    that the maximal absolute value of each feature in the
    training set will be 1.0.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data.
    axis : int (0 by default)
        axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.
    copy : boolean, optional, default is True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a numpy array).
    See also
    --------
    MaxAbsScaler: Performs scaling to the [-1, 1] range using the``Transformer`` API
        (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).
    Notes
    -----
    NaNs are treated as missing values: disregarded to compute the statistics,
    and maintained during the data transformation.
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """  # noqa
    # Unlike the scaler object, this function allows 1d input.

    # If copy is required, it will be done inside the scaler object.
    # X = check_array(X, accept_sparse=('csr', 'csc'), copy=False,
    #                 ensure_2d=False, dtype=FLOAT_DTYPES,
    #                 force_all_finite='allow-nan')

    s = MaxAbsScaler(copy=copy)

    if isinstance(X, cudf.Series):  # if is Series, then axis does not matter
        X = X.to_frame()
        X = s.fit_transform(X)
        return X.T.iloc[0]
    else:
        if axis == 0:
            X = s.fit_transform(X)
        else:
            X = s.fit_transform(X.T).T
        return X
