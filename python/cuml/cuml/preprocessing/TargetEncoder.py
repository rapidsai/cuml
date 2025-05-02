#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import warnings

import cudf
import cupy as cp
import numpy as np
import pandas as pd

from cuml.common.exceptions import NotFittedError


def get_stat_func(stat):
    def func(ds):
        if hasattr(ds, stat):
            return getattr(ds, stat)()
        else:
            # implement stat
            raise ValueError(f"{stat} function is not implemented.")

    return func


class TargetEncoder:
    """
    A cudf based implementation of target encoding [1]_, which converts
    one or multiple categorical variables, 'Xs', with the average of
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
    smooth : int or float (default=0)
        Count of samples to smooth the encoding. 0 means no smoothing.
    seed : int (default=42)
        Random seed
    split_method : {'random', 'continuous', 'interleaved'}, \
        (default='interleaved')
        Method to split train data into `n_folds`.
        'random': random split.
        'continuous': consecutive samples are grouped into one folds.
        'interleaved': samples are assign to each fold in a round robin way.
        'customize': customize splitting by providing a `fold_ids` array
        in `fit()` or `fit_transform()` functions.
    output_type : {'cupy', 'numpy', 'auto'}, default = 'auto'
        The data type of output. If 'auto', it matches input data.
    stat : {'mean','var','median'}, default = 'mean'
        The statistic used in encoding, mean, variance or median of the
        target.

    References
    ----------
    .. [1] https://maxhalford.github.io/blog/target-encoding/

    Examples
    --------
    Converting a categorical implementation to a numerical one

    >>> from cudf import DataFrame, Series
    >>> from cuml.preprocessing import TargetEncoder
    >>> train = DataFrame({'category': ['a', 'b', 'b', 'a'],
    ...                    'label': [1, 0, 1, 1]})
    >>> test = DataFrame({'category': ['a', 'c', 'b', 'a']})

    >>> encoder = TargetEncoder()
    >>> train_encoded = encoder.fit_transform(train.category, train.label)
    >>> test_encoded = encoder.transform(test.category)
    >>> print(train_encoded)
    [1. 1. 0. 1.]
    >>> print(test_encoded)
    [1.   0.75 0.5  1.  ]
    """

    def __init__(
        self,
        n_folds=4,
        smooth=0,
        seed=42,
        split_method="interleaved",
        output_type="auto",
        stat="mean",
    ):
        if smooth < 0:
            raise ValueError(f"smooth {smooth} is not zero or positive")
        if n_folds < 0 or not isinstance(n_folds, int):
            raise ValueError(
                "n_folds {} is not a positive integer".format(n_folds)
            )
        if output_type not in {"cupy", "numpy", "auto"}:
            msg = (
                "output_type should be either 'cupy'"
                " or 'numpy' or 'auto', "
                "got {0}.".format(output_type)
            )
            raise ValueError(msg)
        if stat not in {"mean", "var", "median"}:
            msg = "stat should be 'mean', 'var' or 'median'." f"got {stat}."
            raise ValueError(msg)

        if not isinstance(seed, int):
            raise ValueError("seed {} is not an integer".format(seed))

        if split_method not in {
            "random",
            "continuous",
            "interleaved",
            "customize",
        }:
            msg = (
                "split_method should be either 'random'"
                " or 'continuous' or 'interleaved', or 'customize'"
                "got {0}.".format(split_method)
            )
            raise ValueError(msg)

        self.n_folds = n_folds
        self.seed = seed
        self.smooth = smooth
        self.split_method = split_method
        self.y_col = "__TARGET__"
        self.y_col2 = "__TARGET__SQUARE__"
        self.x_col = "__FEA__"
        self.out_col = "__TARGET_ENCODE__"
        self.out_col2 = "__TARGET_ENCODE__SQUARE__"
        self.fold_col = "__FOLD__"
        self.id_col = "__INDEX__"
        self.train = None
        self.output_type = output_type
        self.stat = stat

    def fit(self, x, y, fold_ids=None):
        """
        Fit a TargetEncoder instance to a set of categories

        Parameters
        ----------
        x : cudf.Series or cudf.DataFrame or cupy.ndarray
           categories to be encoded. It's elements may or may
           not be unique
        y : cudf.Series or cupy.ndarray
            Series containing the target variable.
        fold_ids : cudf.Series or cupy.ndarray
            Series containing the indices of the customized
            folds. Its values should be integers in range
            `[0, N-1]` to split data into `N` folds. If None,
            fold_ids is generated based on `split_method`.

        Returns
        -------
        self : TargetEncoder
            A fitted instance of itself to allow method chaining
        """
        if self.split_method == "customize" and fold_ids is None:
            raise ValueError(
                "`fold_ids` is required "
                "since split_method is set to"
                "'customize'."
            )
        if fold_ids is not None and self.split_method != "customize":
            self.split_method == "customize"
            warnings.warn(
                "split_method is set to 'customize'"
                "since `fold_ids` are provided."
            )
        if fold_ids is not None and len(fold_ids) != len(x):
            raise ValueError(
                f"`fold_ids` length {len(fold_ids)}"
                "is different from input data length"
                f"{len(x)}"
            )

        res, train = self._fit_transform(x, y, fold_ids=fold_ids)
        self.train_encode = res
        self.train = train
        self._fitted = True
        return self

    def fit_transform(self, x, y, fold_ids=None):
        """
        Simultaneously fit and transform an input

        This is functionally equivalent to (but faster than)
        `TargetEncoder().fit(y).transform(y)`

        Parameters
        ----------
        x : cudf.Series or cudf.DataFrame or cupy.ndarray
           categories to be encoded. It's elements may or may
           not be unique
        y : cudf.Series or cupy.ndarray
            Series containing the target variable.
        fold_ids : cudf.Series or cupy.ndarray
            Series containing the indices of the customized
            folds. Its values should be integers in range
            `[0, N-1]` to split data into `N` folds. If None,
            fold_ids is generated based on `split_method`.

        Returns
        -------
        encoded : cupy.ndarray
            The ordinally encoded input series

        """
        self.fit(x, y, fold_ids=fold_ids)
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
        test = self._data_with_strings_to_cudf_dataframe(x)
        if self._is_train_df(test):
            return self.train_encode
        x_cols = [i for i in test.columns.tolist() if i != self.id_col]
        test = test.merge(self.encode_all, on=x_cols, how="left")
        return self._impute_and_sort(test)

    def _fit_transform(self, x, y, fold_ids):
        """
        Core function of target encoding
        """
        self.output_type = self._get_output_type(x)
        cp.random.seed(self.seed)
        train = self._data_with_strings_to_cudf_dataframe(x)
        x_cols = [i for i in train.columns.tolist() if i != self.id_col]
        train[self.y_col] = self._make_y_column(y)

        self.n_folds = min(self.n_folds, len(train))
        train[self.fold_col] = self._make_fold_column(len(train), fold_ids)

        self.y_stat_val = get_stat_func(self.stat)(train[self.y_col])
        if self.stat in ["median"]:
            return self._fit_transform_for_loop(train, x_cols)

        self.mean = train[self.y_col].mean()
        if self.stat == "var":
            y_cols = [self.y_col, self.y_col2]
            train[self.y_col2] = self._make_y_column(y * y)
            self.mean2 = train[self.y_col2].mean()
        else:
            y_cols = [self.y_col]

        y_count_each_fold, y_count_all = self._groupby_agg(
            train, x_cols, op="count", y_cols=y_cols
        )

        y_sum_each_fold, y_sum_all = self._groupby_agg(
            train, x_cols, op="sum", y_cols=y_cols
        )

        """
        Note:
            encode_each_fold is used to encode train data.
            encode_all is used to encode test data.
        """
        cols = [self.fold_col] + x_cols
        encode_each_fold = self._compute_output(
            y_sum_each_fold,
            y_count_each_fold,
            cols,
            f"{self.y_col}_x",
            f"{self.y_col2}_x",
        )
        encode_all = self._compute_output(
            y_sum_all, y_count_all, x_cols, self.y_col, self.y_col2
        )

        self.encode_all = encode_all

        train = train.merge(encode_each_fold, on=cols, how="left")
        del encode_each_fold
        return self._impute_and_sort(train), train

    def _fit_transform_for_loop(self, train, x_cols):
        def _rename_col(df, col):
            df.columns = [col]
            return df.reset_index()

        res = []
        unq_vals = train[self.fold_col].unique()
        if not isinstance(unq_vals, (cp.ndarray, np.ndarray)):
            unq_vals = unq_vals.values_host
        for f in unq_vals:
            mask = train[self.fold_col].values == f
            dg = train.loc[~mask].groupby(x_cols).agg({self.y_col: self.stat})
            dg = _rename_col(dg, self.out_col)
            res.append(train.loc[mask].merge(dg, on=x_cols, how="left"))
        res = cudf.concat(res, axis=0)
        self.encode_all = train.groupby(x_cols).agg({self.y_col: self.stat})
        self.encode_all = _rename_col(self.encode_all, self.out_col)
        return self._impute_and_sort(res), train

    def _make_y_column(self, y):
        """
        Create a target column given y
        """
        if isinstance(y, cudf.Series) or isinstance(y, pd.Series):
            return y.values
        elif isinstance(y, cp.ndarray) or isinstance(y, np.ndarray):
            if len(y.shape) == 1:
                return y
            elif y.shape[1] == 1:
                return y[:, 0]
            else:
                raise ValueError(
                    f"Input of shape {y.shape} " "is not a 1-D array."
                )
        else:
            raise TypeError(
                f"Input of type {type(y)} is not cudf.Series, "
                "or pandas.Series"
                "or numpy.ndarray"
                "or cupy.ndarray"
            )

    def _make_fold_column(self, len_train, fold_ids):
        """
        Create a fold id column for each split
        """

        if self.split_method == "random":
            return cp.random.randint(0, self.n_folds, len_train)
        elif self.split_method == "continuous":
            return (
                cp.arange(len_train) / (len_train / self.n_folds)
            ) % self.n_folds
        elif self.split_method == "interleaved":
            return cp.arange(len_train) % self.n_folds
        elif self.split_method == "customize":
            if fold_ids is None:
                raise ValueError(
                    "fold_ids can't be None"
                    "since split_method is set to"
                    "'customize'."
                )
            return fold_ids
        else:
            msg = (
                "split_method should be either 'random'"
                " or 'continuous' or 'interleaved', "
                "got {0}.".format(self.split_method)
            )
            raise ValueError(msg)

    def _compute_output(self, df_sum, df_count, cols, y_col, y_col2=None):
        """
        Compute the output encoding based on aggregated sum and count
        """
        df_sum = df_sum.merge(df_count, on=cols, how="left")
        smooth = self.smooth
        df_sum[self.out_col] = (df_sum[f"{y_col}_x"] + smooth * self.mean) / (
            df_sum[f"{y_col}_y"] + smooth
        )
        if self.stat == "var":
            df_sum[self.out_col2] = (
                df_sum[f"{y_col2}_x"] + smooth * self.mean2
            ) / (df_sum[f"{y_col2}_y"] + smooth)
            df_sum[self.out_col] = (
                df_sum[self.out_col2] - df_sum[self.out_col] ** 2
            )
            df_sum[self.out_col] = (
                df_sum[self.out_col]
                * df_sum[f"{y_col2}_y"]
                / (df_sum[f"{y_col2}_y"] - 1)
            )
        return df_sum

    def _groupby_agg(self, train, x_cols, op, y_cols):
        """
        Compute aggregated value of each fold and overall dataframe
        grouped by `x_cols` and agg by `op`
        """
        cols = [self.fold_col] + x_cols
        df_each_fold = train.groupby(cols, as_index=False).agg(
            {y_col: op for y_col in y_cols}
        )
        df_all = df_each_fold.groupby(x_cols, as_index=False).agg(
            {y_col: "sum" for y_col in y_cols}
        )

        df_each_fold = df_each_fold.merge(df_all, on=x_cols, how="left")
        for y_col in y_cols:
            df_each_fold[f"{y_col}_x"] = (
                df_each_fold[f"{y_col}_y"] - df_each_fold[f"{y_col}_x"]
            )
        return df_each_fold, df_all

    def _check_is_fitted(self):
        if not self._fitted or self.train is None:
            msg = (
                "This LabelEncoder instance is not fitted yet. Call 'fit' "
                "with appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg)

    def _is_train_df(self, df):
        """
        Return True if the dataframe `df` is the training dataframe, which
        is used in `fit_transform`
        """
        if len(df) != len(self.train):
            return False
        self.train = self.train.sort_values(self.id_col).reset_index(drop=True)
        for col in df.columns:
            if col not in self.train.columns:
                raise ValueError(
                    f"Input column {col} " "is not in train data."
                )
            if not (df[col] == self.train[col]).all():
                return False
        return True

    def _impute_and_sort(self, df):
        """
        Impute and sort the result encoding in the same row order as input
        """
        df[self.out_col] = df[self.out_col].nans_to_nulls()
        df[self.out_col] = df[self.out_col].fillna(self.y_stat_val)
        df = df.sort_values(self.id_col)
        res = df[self.out_col].values.copy()
        if self.output_type == "numpy":
            return cp.asnumpy(res)
        return res

    def _data_with_strings_to_cudf_dataframe(self, x):
        """
        Convert input data with strings to cudf dataframe.
        Supported data types are:
            1D or 2D numpy/cupy arrays
            pandas/cudf Series
            pandas/cudf DataFrame
        Input data could have one or more string columns.
        """
        if isinstance(x, cudf.DataFrame):
            df = x.copy()
        elif isinstance(x, cudf.Series):
            df = x.to_frame().copy()
        elif isinstance(x, cp.ndarray) or isinstance(x, np.ndarray):
            df = cudf.DataFrame()
            if len(x.shape) == 1:
                df[self.x_col] = x
            else:
                df = cudf.DataFrame(
                    x, columns=[f"{self.x_col}_{i}" for i in range(x.shape[1])]
                )
        elif isinstance(x, pd.DataFrame):
            df = cudf.from_pandas(x)
        elif isinstance(x, pd.Series):
            df = cudf.from_pandas(x.to_frame())
        else:
            raise TypeError(
                f"Input of type {type(x)} is not cudf.Series, cudf.DataFrame "
                "or pandas.Series or pandas.DataFrame"
                "or cupy.ndarray or numpy.ndarray"
            )
        df[self.id_col] = cp.arange(len(x))
        return df.reset_index(drop=True)

    def _get_output_type(self, x):
        """
        Infer output type if 'auto'
        """
        if self.output_type != "auto":
            return self.output_type
        if (
            isinstance(x, np.ndarray)
            or isinstance(x, pd.DataFrame)
            or isinstance(x, pd.Series)
        ):
            return "numpy"
        return "cupy"

    @classmethod
    def _get_param_names(cls):
        return [
            "n_folds",
            "smooth",
            "seed",
            "split_method",
        ]

    def get_params(self, deep=False):
        """
        Returns a dict of all params owned by this class.
        """
        params = dict()
        variables = self._get_param_names()
        for key in variables:
            var_value = getattr(self, key, None)
            params[key] = var_value
        return params
