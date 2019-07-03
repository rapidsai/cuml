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


class MinMaxScaler():
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