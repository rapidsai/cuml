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

__all__ = [
    # 'Binarizer',
    # 'KernelCenterer',
    'MinMaxScaler',
    # 'MaxAbsScaler',
    # 'Normalizer',
    # 'OneHotEncoder',
    # 'RobustScaler',
    # 'StandardScaler',
    # 'QuantileTransformer',
    # 'PowerTransformer',
    # 'add_dummy_feature',
    # 'binarize',
    # 'normalize',
    # 'scale',
    # 'robust_scale',
    # 'maxabs_scale',
    # 'minmax_scale',
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
    """Transforms features by scaling each feature to a given range.
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
    Read more in the :ref:`User Guide <preprocessing_scaler>`.
    .. versionadded:: 0.17
       *minmax_scale* function interface
       to :class:`sklearn.preprocessing.MinMaxScaler`.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data.
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    axis : int (0 by default)
        axis used to scale along. If 0, independently scale each feature,
        otherwise (if 1) scale each sample.
    copy : boolean, optional, default is True
        Set to False to perform inplace scaling and avoid a copy (if the input
        is already a numpy array).
    See also
    --------
    MinMaxScaler: Performs scaling to a given range using the``Transformer`` API
        (e.g. as part of a preprocessing :class:`sklearn.pipeline.Pipeline`).
    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """  # noqa
    # Unlike the scaler object, this function allows 1d input.
    # If copy is required, it will be done inside the scaler object.

    # X = check_array(X, copy=False, ensure_2d=False,
    #                 dtype=FLOAT_DTYPES, force_all_finite='allow-nan')

    if isinstance(X, cudf.Series):
        X = X.reshape(X.shape[0], 1)

    s = MinMaxScaler(feature_range=feature_range, copy=copy)
    if axis == 0:
        X = s.fit_transform(X)
    else:
        X = s.fit_transform(X.T).T

    if original_ndim == 1:
        X = X.ravel()

    return X
