import cupy as cp
import cudf
import numpy as np
from cupyx.scipy import sparse


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
    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]
    if not all([hasattr(self, attr) for attr in attributes]):
        raise RuntimeError("Scaler must first be fitted")


def to_cupy(X):
    # TODO: accept numba cuda array input 
    if isinstance(X, cudf.DataFrame):
        X = cp.array(X.as_gpu_matrix())
    if not isinstance(X, cp.ndarray):
        raise TypeError('Input should be either cupy array or cudf.DataFrame')
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


def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides cupy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.
    Parameters
    ----------
    op : function
        A cupy accumulator function such as cp.mean or cp.sum
    x : cupy array
        A cupy array to apply the accumulator function
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function
    Returns
    -------
    result : The output of the accumulator function passed to this function
    """
    if cp.issubdtype(x.dtype, cp.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=cp.float64)
    else:
        result = op(x, *args, **kwargs)
    return result


def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count
    new_sum = _safe_accumulator_op(cp.nansum, X, axis=0)

    new_sample_count = cp.sum(~cp.isnan(X), axis=0)
    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        new_unnormalized_variance = (
            _safe_accumulator_op(cp.nanvar, X, axis=0) * new_sample_count)
        last_unnormalized_variance = last_variance * last_sample_count

        # with np.errstate(divide='ignore', invalid='ignore'):
        # Unfortunately, cupy does not support errstate now, so input shouldn't
        # contain nan
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

    # X = check_array(X, accept_sparse='csc', copy=copy, ensure_2d=False,
    #                 estimator='the scale function', dtype=FLOAT_DTYPES,
    #                 force_all_finite='allow-nan')    
    X = to_cupy(X)

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
    return X

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
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

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
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))
        ###########################################################
        # X = check_array(X, copy=self.copy,
        #                 estimator=self, dtype=FLOAT_DTYPES,
        #                 force_all_finite="allow-nan")

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
        """Scaling features of X according to feature_range.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """
        check_fitted(self, 'scale_')

        # X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
        #                 force_all_finite="allow-nan")

        X *= self.scale_
        X += self.min_
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed. It cannot be sparse.
        """
        check_fitted(self, 'scale_')

        # X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES,
        #                 force_all_finite="allow-nan")

        X -= self.min_
        X /= self.scale_
        return X


def minmax_scale(X, feature_range=(0, 1), axis=0, copy=True):
    # Unlike the scaler object, this function allows 1d input.
    # If copy is required, it will be done inside the scaler object.
    # X = check_array(X, copy=False, ensure_2d=False,
    #                 dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
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

    return X


class StandardScaler(BaseEstimator, TransformerMixin):
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
        """Perform standardization by centering and scaling
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        check_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        # X = check_array(X, accept_sparse='csr', copy=copy,
        #                 estimator=self, dtype=FLOAT_DTYPES,
        #                 force_all_finite='allow-nan')

        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.scale_
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
        check_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy

        X = to_cupy(X)
        if copy:
            X = X.copy()
        if self.with_std:
            X *= self.scale_
        if self.with_mean:
            X += self.mean_
        return X


class MaxAbsScaler(BaseEstimator, TransformerMixin):

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
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
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
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y
            Ignored
        """
        # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
        #                 estimator=self, dtype=FLOAT_DTYPES,
        #                 force_all_finite='allow-nan')

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
        """Scale the data
        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data that should be scaled.
        """
        check_fitted(self, 'scale_')
        # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
        #                 estimator=self, dtype=FLOAT_DTYPES,
        #                 force_all_finite='allow-nan')

        X /= self.scale_
        return X

    def inverse_transform(self, X):
        """Scale back the data to the original representation
        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data that should be transformed back.
        """
        check_fitted(self, 'scale_')
        # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
        #                 estimator=self, dtype=FLOAT_DTYPES,
        #                 force_all_finite='allow-nan')

        X *= self.scale_
        return X


def maxabs_scale(X, axis=0, copy=True):
    # Unlike the scaler object, this function allows 1d input.

    # If copy is required, it will be done inside the scaler object.
    # X = check_array(X, accept_sparse=('csr', 'csc'), copy=False,
    #                 ensure_2d=False, dtype=FLOAT_DTYPES,
    #                 force_all_finite='allow-nan')
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



class RobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self, with_centering=True, with_scaling=True,
                 quantile_range=(25.0, 75.0), copy=True):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the median and quantiles to be used for scaling.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the median and quantiles
            used for later scaling along the features axis.
        """
        # at fit, convert sparse matrices to csc for optimized computation of
        # the quantiles
        # X = check_array(X, accept_sparse='csc', copy=self.copy, estimator=self,
        #                 dtype=FLOAT_DTYPES, force_all_finite='allow-nan')

        q_min, q_max = self.quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError("Invalid quantile range: %s" %
                             str(self.quantile_range))

        if self.with_centering:
            self.center_ = cp.median(X, axis=0)
        else:
            self.center_ = None

        if self.with_scaling:
            quantiles = []
            for feature_idx in range(X.shape[1]):
                column_data = X[:, feature_idx]

                quantiles.append(cp.percentile(column_data,
                                               self.quantile_range))

            quantiles = cp.transpose(quantiles)

            self.scale_ = quantiles[1] - quantiles[0]
            self.scale_ = _handle_zeros_in_scale(self.scale_, copy=False)
        else:
            self.scale_ = None

        return self

    def transform(self, X):
        """Center and scale the data.
        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data used to scale along the specified axis.
        """
        check_fitted(self, ['center_', 'scale_'])
        # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
        #                 estimator=self, dtype=FLOAT_DTYPES,
        #                 force_all_finite='allow-nan')

        if self.with_centering:
            X -= self.center_
        if self.with_scaling:
            X /= self.scale_
        return X

    def inverse_transform(self, X):
        """Scale back the data to the original representation
        Parameters
        ----------
        X : array-like
            The data used to scale along the specified axis.
        """
        check_fitted(self, ['center_', 'scale_'])
        # X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
        #                 estimator=self, dtype=FLOAT_DTYPES,
        #                 force_all_finite='allow-nan')

        if self.with_scaling:
            X *= self.scale_
        if self.with_centering:
            X += self.center_
        return X



def robust_scale(X, axis=0, with_centering=True, with_scaling=True,
                 quantile_range=(25.0, 75.0), copy=True):

    # X = check_array(X, accept_sparse=('csr', 'csc'), copy=False,
    #                 ensure_2d=False, dtype=FLOAT_DTYPES,
    #                 force_all_finite='allow-nan')
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

    return X


def row_norms(X, squared=False):
    norms = cp.einsum('ij,ij->i', X, X)

    if not squared:
        cp.sqrt(norms, norms)
    return norms


def normalize(X, norm='l2', axis=1, copy=True, return_norm=False):

    if norm not in ('l1', 'l2', 'max'):
        raise ValueError("'%s' is not a supported norm" % norm)

    if axis != 0 and axis != 1:
        raise ValueError("'%d' is not a supported axis" % axis)

    # X = check_array(X, sparse_format, copy=copy,
    #                 estimator='the normalize function', dtype=FLOAT_DTYPES)
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

    if return_norm:
        return X, norms
    else:
        return X


class Normalizer(TransformerMixin):

    def __init__(self, norm='l2', copy=True):
        self.norm = norm
        self.copy = copy

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged
        This method is just there to implement the usual API and hence
        work in pipelines.
        Parameters
        ----------
        X : array-like
        """
        # check_array(X, accept_sparse='csr')
        return self

    def transform(self, X, copy=None):
        """Scale each non zero row of X to unit norm
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data to normalize, row by row. scipy.sparse matrices should be
            in CSR format to avoid an un-necessary copy.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        copy = copy if copy is not None else self.copy
        # X = check_array(X, accept_sparse='csr')
        return normalize(X, norm=self.norm, axis=1, copy=copy)


def binarize(X, threshold=0.0, copy=True):
    # X = check_array(X, accept_sparse=['csr', 'csc'], copy=copy)
    cond = X > threshold
    not_cond = cp.logical_not(cond)
    X[cond] = 1
    X[not_cond] = 0
    return X


class Binarizer(TransformerMixin):
    def __init__(self, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged
        This method is just there to implement the usual API and hence
        work in pipelines.
        Parameters
        ----------
        X : array-like
        """
        # check_array(X, accept_sparse='csr')
        return self

    def transform(self, X, copy=None):
        """Binarize each element of X
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data to binarize, element by element.
            scipy.sparse matrices should be in CSR format to avoid an
            un-necessary copy.
        copy : bool
            Copy the input X or not.
        """
        copy = copy if copy is not None else self.copy
        return binarize(X, threshold=self.threshold, copy=copy)


class KernelCenterer(BaseEstimator, TransformerMixin):

    def __init__(self):
        # Needed for backported inspect.signature compatibility with PyPy
        pass

    def fit(self, K, y=None):
        """Fit KernelCenterer
        Parameters
        ----------
        K : numpy array of shape [n_samples, n_samples]
            Kernel matrix.
        Returns
        -------
        self : returns an instance of self.
        """
        # K = check_array(K, dtype=FLOAT_DTYPES)
        n_samples = K.shape[0]
        self.K_fit_rows_ = cp.sum(K, axis=0) / n_samples
        self.K_fit_all_ = self.K_fit_rows_.sum() / n_samples
        return self

    def transform(self, K, copy=True):
        """Center kernel matrix.
        Parameters
        ----------
        K : numpy array of shape [n_samples1, n_samples2]
            Kernel matrix.
        copy : boolean, optional, default True
            Set to False to perform inplace computation.
        Returns
        -------
        K_new : numpy array of shape [n_samples1, n_samples2]
        """
        check_fitted(self, 'K_fit_all_')

        # K = check_array(K, copy=copy, dtype=FLOAT_DTYPES)

        K_pred_cols = (cp.sum(K, axis=1) /
                       self.K_fit_rows_.shape[0])[:, cp.newaxis]

        K -= self.K_fit_rows_
        K -= K_pred_cols
        K += self.K_fit_all_

        return K

    @property
    def _pairwise(self):
        return True


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
    >>> from sklearn.preprocessing import add_dummy_feature
    >>> add_dummy_feature([[0, 1], [1, 0]])
    array([[1., 0., 1.],
           [1., 1., 0.]])
    """
    # X = check_array(X, accept_sparse=['csc', 'csr', 'coo'], dtype=FLOAT_DTYPES)
    n_samples, n_features = X.shape
    shape = (n_samples, n_features + 1)
    return cp.hstack((cp.full((n_samples, 1), value), X))
