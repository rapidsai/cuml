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

import cudf
import cupy as cp
from cuml.common.exceptions import NotFittedError


class TargetEncoder:
    """
    A cudf based implementation of target encoding [1], which converts
    one or mulitple categorical variables, 'Xs', with the average of
    corresponding values of the target variable, 'Y'. The input data is
    grouped by the columns `Xs` and the aggregated mean value of `Y` of
    each group is calculated to replace each value of `Xs`. Several
    optimizations are applied to prevent label leakage and parallelize
    the execution.

    Parameters
    ----------
    n_folds : int (default=4)
        Default number of folds for fitting training data. To prevent
        label leakage in `fit`, we split data into `n_folds` and
        encode one fold using the target variables of the remaining folds.
    smooth : int (default=0)
        Number of samples to smooth the encoding
    seed : int (default=42)
        Random seed
    split_method : {'random', 'continuous', 'interleaved'},
        default='interleaved'
        Method to split train data into `n_folds`.
        'random': random split.
        'continuous': consecutive samples are grouped into one folds.
        'interleaved': samples are assign to each fold in a round robin way.

    References
    ----------
    .. [1] https://maxhalford.github.io/blog/target-encoding/

    Examples
    --------
    Converting a categorical implementation to a numerical one

    .. code-block:: python

        from cudf import DataFrame, Series

        train = DataFrame({'category': ['a', 'b', 'b', 'a'],
                           'label': [1, 0, 1, 1]})
        test = DataFrame({'category': ['a', 'c', 'b', 'a']})

        encoder = TargetEncoder()
        train_encoded = encoder.fit_transform(train.category, train.label)
        test_encoded = encoder.transform(test.category)
        print(train_encoded)
        print(test_encoded)

    Output:

    .. code-block:: python

        [1. 1. 0. 1.]
        [1.   0.75 0.5  1.  ]

    """
    def __init__(self, n_folds=4, smooth=0, seed=42,
                 split_method='interleaved'):
        self.n_folds = n_folds
        self.seed = seed
        self.smooth = smooth
        self.split = split_method
        self.y_col = '__TARGET__'
        self.x_col = '__FEA__'
        self.out_col = '__TARGET_ENCODE__'
        self.fold_col = '__FOLD__'
        self.id_col = '__INDEX__'
        self.train = None

    def fit(self, x, y):
        """
        Fit a TargetEncoder instance to a set of categories

        Parameters
        ----------
        x: cudf.Series or cudf.DataFrame or cupy.ndarray
           categories to be encoded. It's elements may or may
           not be unique
        y : cudf.Series or cupy.ndarray
            Series containing the target variable.

        Returns
        -------
        self : TargetEncoder
            A fitted instance of itself to allow method chaining
        """
        res, train = self._fit_transform(x, y)
        self.train_encode = res
        self.train = train
        self._fitted = True
        return self

    def fit_transform(self, x, y):
        """
        Simultaneously fit and transform an input

        This is functionally equivalent to (but faster than)
        `TargetEncoder().fit(y).transform(y)`
        """
        self.fit(x, y)
        return self.train_encode

    def transform(self, x):
        """
        Transform an input into its categorical keys.

        This is intended for test data. For fitting and transforming
        the training data, prefer `fit_transform`.

        Parameters
        ----------
        x : cudf.Series
            Input keys to be transformed. Its values doesn't have to
            match the categories given to `fit`

        Returns
        -------
        encoded : cupy.ndarray
            The ordinally encoded input series

        """
        self._check_is_fitted()
        test = self._to_cudf_frame(x)
        if self._is_train_df(test):
            return self.train_encode
        x_cols = [i for i in test.columns.tolist() if i != self.id_col]
        test = test.merge(self.agg_all, on=x_cols, how='left')
        return self._get_return_value(test)

    def _fit_transform(self, x, y):
        cp.random.seed(self.seed)
        train = self._to_cudf_frame(x)
        x_cols = [i for i in train.columns.tolist() if i != self.id_col]

        if isinstance(y, cudf.Series):
            train[self.y_col] = y.values
        elif isinstance(y, cp.ndarray):
            if len(y.shape) == 1:
                train[self.y_col] = y
            elif y.shape[1] == 1:
                train[self.y_col] = y[:, 0]
            else:
                raise ValueError(f"Input of shape {y.shape} "
                                 "is not a 1-D array.")
        else:
            raise TypeError(
                "Input of type {type(y)} is not cudf.Series, "
                "or cupy.ndarray")

        self.n_folds = min(self.n_folds, len(train))

        if self.split == 'random':
            train[self.fold_col] = cp.random.randint(0,
                                                     self.n_folds, len(train))
        elif self.split == 'continuous':
            train[self.fold_col] = cp.arange(len(train)) / \
                (len(train)/self.n_folds)
            train[self.fold_col] = train[self.fold_col] % self.n_folds
        elif self.split == 'interleaved':
            train[self.fold_col] = cp.arange(len(train))
            train[self.fold_col] = train[self.fold_col] % self.n_folds
        else:
            msg = ("split should be either 'random'"
                   " or 'continuous' or 'interleaved', "
                   "got {0}.".format(self.split))
            raise ValueError(msg)

        train[self.fold_col] = train[self.fold_col].astype('int32')
        self.mean = train[self.y_col].mean()

        cols = [self.fold_col]+x_cols

        agg_y_count = train.groupby(cols, as_index=False)\
            .agg({self.y_col: 'count'})
       
        agg_y_sum = train.groupby(cols, as_index=False)\
             .agg({self.y_col: 'sum'})

        agg_all_y_count = agg_y_count.groupby(x_cols, as_index=False)\
            .agg({self.y_col: 'sum'})

        agg_all_y_sum = agg_y_sum.groupby(x_cols, as_index=False)\
            .agg({self.y_col: 'sum'})  

        agg_y_count = agg_y_count.merge(agg_all_y_count, on=x_cols, how='left')
        agg_y_sum = agg_y_sum.merge(agg_all_y_sum, on=x_cols, how='left')
        agg_y_count[f'{self.y_col}_x'] = agg_y_count[f'{self.y_col}_y'] -\
            agg_y_count[f'{self.y_col}_x']
        agg_y_sum[f'{self.y_col}_x'] = agg_y_sum[f'{self.y_col}_y'] -\
            agg_y_sum[f'{self.y_col}_x']
 
        agg_y_sum[self.out_col] = (agg_y_sum[f'{self.y_col}_x'] +
                                       self.smooth*self.mean) / \
                                      (agg_y_count[f'{self.y_col}_x'] +
                                       self.smooth)

        agg_all_y_sum[self.out_col] = (agg_all_y_sum[self.y_col] +
                                       self.smooth*self.mean) / \
                                      (agg_all_y_count[self.y_col] +
                                       self.smooth) 
        self.agg_all = agg_all_y_sum

        cols = [self.fold_col]+x_cols
        train = train.merge(agg_y_sum, on=cols, how='left')
        del agg_y_sum
        return self._get_return_value(train), train

    def _check_is_fitted(self):
        if not self._fitted or self.train is None:
            msg = ("This LabelEncoder instance is not fitted yet. Call 'fit' "
                   "with appropriate arguments before using this estimator.")
            raise NotFittedError(msg)

    def _is_train_df(self, df):
        """
        Return True if the dataframe `df` is the training dataframe, which
        is used in `fit_transform`
        """
        if len(df) != len(self.train):
            return False
        for col in df.columns:
            if col not in self.train.columns:
                raise ValueError(f"Input column {col} "
                                 "is not in train data.")
            if not (df[col] == self.train[col]).all():
                return False
        return True

    def _get_return_value(self, df):
        """
        Return the result encoding in the same row order as input
        """
        df[self.out_col] = df[self.out_col].nans_to_nulls()
        df[self.out_col] = df[self.out_col].fillna(self.mean)
        df = df.sort_values(self.id_col)
        res = df[self.out_col].values.copy()
        return res

    def _to_cudf_frame(self, x):
        if isinstance(x, cudf.DataFrame):
            df = x.copy()
        elif isinstance(x, cudf.Series):
            df = x.to_frame().copy()
        elif isinstance(x, cp.ndarray):
            df = cudf.DataFrame()
            if len(x.shape) == 1:
                df[self.x_col] = x
            else:
                df = cudf.DataFrame(x,
                                    columns=[f'{self.x_col}_{i}'
                                             for i in range(x.shape[1])])
        else:
            raise TypeError(
                f"Input of type {x.shape} is not cudf.Series, cudf.DataFrame "
                "or cupy.ndarray")
        df[self.id_col] = cp.arange(len(x))
        return df
