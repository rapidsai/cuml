#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import warnings

import cudf
import cupy as cp
import numpy as np
import pandas as pd

from cuml.common.exceptions import NotFittedError
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.interop import InteropMixin, to_cpu, to_gpu
from cuml.internals.outputs import reflect


def get_stat_func(stat):
    def func(ds):
        if hasattr(ds, stat):
            return getattr(ds, stat)()
        else:
            # implement stat
            raise ValueError(f"{stat} function is not implemented.")

    return func


class TargetEncoder(Base, InteropMixin):
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
    handle : cuml.Handle or None, default=None

        .. deprecated:: 26.02
            The `handle` argument was deprecated in 26.02 and will be removed
            in 26.04. There's no need to pass in a handle, cuml now manages
            this resource automatically.

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    stat : {'mean','var','median'}, default = 'mean'
        The statistic used in encoding, mean, variance or median of the
        target.
    multi_feature_mode : {'combination', 'independent'}, default='combination'
        How to handle multiple input features:

        - ``'combination'``: Encode all feature combinations together
          (cuML native behavior). Produces a single output column with
          encodings based on the joint distribution of all features.
        - ``'independent'``: Encode each feature independently (sklearn
          behavior). Produces N output columns, one per input feature,
          each containing encodings based only on that feature's
          relationship with the target.

        For single-feature input, both modes produce identical results.

    Attributes
    ----------
    categories_ : list of cupy.ndarray
        The categories of each input feature determined during fitting.
        Each element is an array of unique category values for that feature,
        sorted in ascending order.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    encode_all : cudf.DataFrame
        DataFrame containing the learned encodings for all category
        combinations. Used internally for transforming new data.
    mean : float
        The overall mean of the target variable, computed during fitting.
        Used for smoothing and imputing unseen categories.
    y_stat_val : float
        The statistic value (mean, variance, or median) of the target
        variable, depending on the ``stat`` parameter. Used to impute
        encodings for unseen categories.
    train : cudf.DataFrame or None
        The training DataFrame used during fitting, containing the original
        features, target values, and fold assignments. Set to ``None`` if
        the encoder was loaded from a sklearn model via :meth:`from_sklearn`.
    train_encode : cuml.internals.array.CumlArray or None
        The encoded values for the training data, computed during
        :meth:`fit` or :meth:`fit_transform`. Set to ``None`` if the
        encoder was loaded from a sklearn model via :meth:`from_sklearn`.

    Notes
    -----
    **sklearn Conversion Limitations**

    When converting between cuML and sklearn via :meth:`as_sklearn` and
    :meth:`from_sklearn`, be aware of the following semantic differences:

    - **Training data behavior**: cuML's :meth:`transform` returns
      cross-validated (regularized) encodings when called on training data
      to prevent data leakage. sklearn's ``transform`` always returns global
      mean encodings regardless of whether the input is training or test data.
      After roundtrip conversion, the cuML model will return global encodings
      for all data since the training data reference is not preserved.

    - **Multi-feature encoding**: cuML's default ``multi_feature_mode='combination'``
      encodes feature combinations jointly, while sklearn always encodes features
      independently. Multi-feature models fitted with ``'combination'`` mode
      cannot be converted to sklearn; use ``multi_feature_mode='independent'``
      for sklearn compatibility.

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

    >>> encoder = TargetEncoder(output_type='numpy')
    >>> train_encoded = encoder.fit_transform(train.category, train.label)
    >>> test_encoded = encoder.transform(test.category)
    >>> print(train_encoded)
    [1. 1. 0. 1.]
    >>> print(test_encoded)
    [1.   0.75 0.5  1.  ]
    """

    # InteropMixin requirements
    _cpu_class_path = "sklearn.preprocessing.TargetEncoder"

    def __init__(
        self,
        *,
        n_folds=4,
        smooth=0,
        seed=42,
        split_method="interleaved",
        handle=None,
        verbose=False,
        output_type=None,
        stat="mean",
        multi_feature_mode="combination",
    ):
        super().__init__(
            handle=handle, verbose=verbose, output_type=output_type
        )
        if smooth < 0:
            raise ValueError(f"smooth {smooth} is not zero or positive")
        if n_folds < 0 or not isinstance(n_folds, int):
            raise ValueError(
                "n_folds {} is not a positive integer".format(n_folds)
            )
        if stat not in {"mean", "var", "median"}:
            msg = f"stat should be 'mean', 'var' or 'median'.got {stat}."
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
        self.stat = stat
        self.multi_feature_mode = multi_feature_mode
        self._fitted = False

    @reflect(reset=True)
    def fit(self, X, y, *, fold_ids=None):
        """
        Fit a TargetEncoder instance to a set of categories

        Parameters
        ----------
        X : cudf.Series or cudf.DataFrame or cupy.ndarray
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
        if y is None:
            raise TypeError(
                f"Input of type {type(y)} is not cudf.Series, "
                "or pandas.Series"
                "or numpy.ndarray"
                "or cupy.ndarray"
            )

        if len(X) == 0:
            raise ValueError(
                "Found array with 0 sample(s) while a minimum of 1 is "
                "required."
            )

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
        if fold_ids is not None and len(fold_ids) != len(X):
            raise ValueError(
                f"`fold_ids` length {len(fold_ids)}"
                "is different from input data length"
                f"{len(X)}"
            )

        res, train = self._fit_transform(X, y, fold_ids=fold_ids)
        self.train_encode = res
        self.train = train
        self._fitted = True

        # Set _n_features_out for sklearn compatibility (get_feature_names_out)
        if getattr(self, "_independent_mode_fitted", False):
            self._n_features_out = self.n_features_in_
        else:
            self._n_features_out = 1

        return self

    @reflect
    def fit_transform(self, X, y, *, fold_ids=None) -> CumlArray:
        """
        Simultaneously fit and transform an input

        This is functionally equivalent to (but faster than)
        `TargetEncoder().fit(y).transform(y)`

        Parameters
        ----------
        X : cudf.Series or cudf.DataFrame or cupy.ndarray
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
        self.fit(X, y, fold_ids=fold_ids)
        return self.train_encode

    @reflect
    def transform(self, X) -> CumlArray:
        """
        Transform an input into its categorical keys.

        This is intended for test data. For fitting and transforming
        the training data, prefer `fit_transform`.

        Parameters
        ----------
        X : cudf.Series
            Input keys to be transformed. Its values doesn't have to
            match the categories given to `fit`

        Returns
        -------
        encoded : cupy.ndarray
            The ordinally encoded input series

        """
        self._check_is_fitted()
        test = self._data_with_strings_to_cudf_dataframe(X)

        # Check feature dimensions match
        x_cols = [i for i in test.columns.tolist() if i != self.id_col]
        if (
            hasattr(self, "n_features_in_")
            and len(x_cols) != self.n_features_in_
        ):
            raise ValueError(
                f"X has {len(x_cols)} features, but TargetEncoder is "
                f"expecting {self.n_features_in_} features as input."
            )

        if self._is_train_df(test):
            return self.train_encode

        # Handle independent mode (per-feature encoding)
        if getattr(self, "_independent_mode_fitted", False):
            return self._transform_independent(test, x_cols)

        test = test.merge(self.encode_all, on=x_cols, how="left")
        return self._impute_and_sort(test)

    def _transform_independent(self, test, x_cols):
        """Transform using independent per-feature encodings."""
        result_cols = []
        for i, col in enumerate(x_cols):
            out_col_i = f"{self.out_col}_{i}"
            encode_all_i = self._encode_all_per_feature[i]

            test = test.merge(
                encode_all_i.rename(columns={self.out_col: out_col_i}),
                on=col,
                how="left",
            )
            test[out_col_i] = test[out_col_i].nans_to_nulls()
            test[out_col_i] = test[out_col_i].fillna(self.y_stat_val)
            result_cols.append(out_col_i)

        test = test.sort_values(self.id_col)
        res = test[result_cols].values.copy()
        return CumlArray(res)

    def _fit_transform(self, x, y, fold_ids):
        cp.random.seed(self.seed)
        train = self._data_with_strings_to_cudf_dataframe(x)
        x_cols = [i for i in train.columns.tolist() if i != self.id_col]

        # Store n_features_in_ and categories_ for sklearn interop
        self.n_features_in_ = len(x_cols)
        self._x_cols = x_cols

        # Extract unique categories for each feature (sorted for consistency)
        self.categories_ = []
        for col in x_cols:
            # Handle string columns specially - cudf.unique() fails on object dtype
            # because it tries to return .values which cupy doesn't support
            try:
                unique_vals = train[col].unique()
            except TypeError:
                # String column in cudf - get unique values via drop_duplicates
                unique_vals = (
                    train[col].drop_duplicates().sort_values().to_numpy()
                )
                self.categories_.append(unique_vals)
                continue

            # Handle both cudf Series and numpy arrays (cudf.pandas compatibility)
            if hasattr(unique_vals, "sort_values"):
                # cudf Series - use sort_values()
                unique_vals = unique_vals.sort_values()
                # Use to_numpy() for string columns since .values fails on strings
                # (cupy doesn't support object dtype)
                try:
                    self.categories_.append(unique_vals.values)
                except TypeError:
                    # String column - use numpy array instead of cupy
                    self.categories_.append(unique_vals.to_numpy())
            else:
                # numpy/cupy array - use np.sort()
                self.categories_.append(np.sort(unique_vals))

        if self.multi_feature_mode not in {"combination", "independent"}:
            raise ValueError(
                f"multi_feature_mode should be 'combination' or 'independent', "
                f"got '{self.multi_feature_mode}'."
            )

        # Delegate to appropriate method based on mode and stat
        if len(x_cols) > 1 and self.multi_feature_mode == "independent":
            return self._fit_transform_independent(train, x_cols, y, fold_ids)
        elif self.stat == "median":
            return self._fit_transform_median(train, x_cols, y, fold_ids)
        else:
            return self._fit_transform_combination(train, x_cols, y, fold_ids)

    def _fit_transform_combination(self, train, x_cols, y, fold_ids):
        """
        Fit-transform with combination encoding (cuML native behavior).

        Encodes all feature combinations together, producing a single output
        column with encodings based on the joint distribution of all features.
        """
        train[self.y_col] = self._make_y_column(y)

        self.n_folds = min(self.n_folds, len(train))
        train[self.fold_col] = self._make_fold_column(len(train), fold_ids)

        self.y_stat_val = get_stat_func(self.stat)(train[self.y_col])
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

        # encode_each_fold is used to encode train data
        # encode_all is used to encode test data
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

    def _fit_transform_median(self, train, x_cols, y, fold_ids):
        """
        Fit-transform with median stat using a for-loop approach.

        Median requires computing the statistic per fold separately,
        which cannot be vectorized like mean/var.
        """
        train[self.y_col] = self._make_y_column(y)

        self.n_folds = min(self.n_folds, len(train))
        train[self.fold_col] = self._make_fold_column(len(train), fold_ids)

        self.y_stat_val = get_stat_func(self.stat)(train[self.y_col])
        self.mean = train[self.y_col].mean()

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

    def _fit_transform_independent(self, train, x_cols, y, fold_ids):
        """
        Fit-transform with independent per-feature encoding (sklearn-like).

        Each feature is encoded independently based on its relationship with
        the target, producing N output columns for N input features.
        """
        y_values = self._make_y_column(y)
        train[self.y_col] = y_values

        self.n_folds = min(self.n_folds, len(train))
        train[self.fold_col] = self._make_fold_column(len(train), fold_ids)

        self.mean = train[self.y_col].mean()
        self.y_stat_val = get_stat_func(self.stat)(train[self.y_col])

        self._encodings_per_feature = []
        self._encode_all_per_feature = []
        result_cols = []

        for i, col in enumerate(x_cols):
            out_col_i = f"{self.out_col}_{i}"

            if self.stat in ["median"]:
                # Use for-loop approach for median
                encode_all_i = self._compute_single_feature_encoding_median(
                    train, col
                )
            else:
                encode_all_i = self._compute_single_feature_encoding(
                    train, col, out_col_i
                )

            self._encode_all_per_feature.append(encode_all_i)

            # Extract encodings in category order for sklearn compatibility
            feature_encodings = []
            for cat_val in self.categories_[i]:
                mask = encode_all_i[col] == cat_val
                if mask.any():
                    enc_val = float(
                        encode_all_i.loc[mask, self.out_col].iloc[0]
                    )
                else:
                    enc_val = float(self.mean)
                feature_encodings.append(enc_val)
            self._encodings_per_feature.append(np.array(feature_encodings))

            # Merge encoding into train for this feature
            train = train.merge(
                encode_all_i.rename(columns={self.out_col: out_col_i}),
                on=col,
                how="left",
            )
            train[out_col_i] = train[out_col_i].nans_to_nulls()
            train[out_col_i] = train[out_col_i].fillna(self.y_stat_val)
            result_cols.append(out_col_i)

        # Create a combined encode_all for transform() compatibility
        # This stores the per-feature encodings in a format transform() can use
        self.encode_all = self._encode_all_per_feature
        self._independent_mode_fitted = True

        # Sort by original index and return results
        train = train.sort_values(self.id_col)
        res = train[result_cols].values.copy()
        return CumlArray(res), train

    def _compute_single_feature_encoding(self, train, col, out_col):
        """Compute target encoding for a single feature column."""
        y_col = self.y_col

        # Group by single feature and compute stats
        df_count = train.groupby(col, as_index=False).agg({y_col: "count"})
        df_count.columns = [col, f"{y_col}_count"]

        df_sum = train.groupby(col, as_index=False).agg({y_col: "sum"})
        df_sum.columns = [col, f"{y_col}_sum"]

        df = df_sum.merge(df_count, on=col, how="left")

        # Apply smoothing
        smooth = self.smooth
        df[self.out_col] = (df[f"{y_col}_sum"] + smooth * self.mean) / (
            df[f"{y_col}_count"] + smooth
        )

        return df[[col, self.out_col]]

    def _compute_single_feature_encoding_median(self, train, col):
        """Compute median target encoding for a single feature column."""
        encode_all = train.groupby(col, as_index=False).agg(
            {self.y_col: self.stat}
        )
        encode_all.columns = [col, self.out_col]
        return encode_all

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
                    f"Input of shape {y.shape} is not a 1-D array."
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
        # Check if fitted - either via fit() or from_sklearn()
        # When loaded from sklearn, train may be None but encode_all exists
        if not self._fitted and not hasattr(self, "encode_all"):
            msg = (
                "This TargetEncoder instance is not fitted yet. Call 'fit' "
                "with appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg)

    def _is_train_df(self, df):
        """
        Return True if the dataframe `df` is the training dataframe, which
        is used in `fit_transform`
        """
        # If train is None (e.g., loaded from sklearn), we can't compare
        if self.train is None:
            return False
        if len(df) != len(self.train):
            return False
        self.train = self.train.sort_values(self.id_col).reset_index(drop=True)
        for col in df.columns:
            if col not in self.train.columns:
                raise ValueError(f"Input column {col} is not in train data.")
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
        return CumlArray(res)

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

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "n_folds",
            "smooth",
            "seed",
            "split_method",
            "stat",
            "multi_feature_mode",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        # Use independent mode when converting from sklearn to match sklearn semantics
        params = {
            "n_folds": model.cv,
            "seed": 42 if model.random_state is None else model.random_state,
            "smooth": 1.0 if model.smooth == "auto" else float(model.smooth),
            "split_method": "random" if model.shuffle else "continuous",
            "stat": "mean",
            # Don't force multi_feature_mode here - for single-feature both
            # modes are equivalent, and for multi-feature the fitted attribute
            # _independent_mode_fitted controls transform behavior
        }
        return params

    def _params_to_cpu(self):
        params = {
            "cv": self.n_folds,
            "random_state": self.seed,
            "smooth": self.smooth,
            "shuffle": self.split_method == "random",
            "categories": "auto",
            "target_type": "continuous",
        }
        return params

    def _attrs_from_cpu(self, model):
        """Convert sklearn TargetEncoder attributes to cuML format.

        sklearn always uses independent per-feature encoding, so we set up
        cuML to use independent mode as well for exact compatibility.
        """
        # Handle string categories (object dtype) - keep as numpy arrays
        # since cupy doesn't support object dtype
        categories_gpu = []
        for cat in model.categories_:
            if cat.dtype == np.object_:
                categories_gpu.append(cat)  # Keep as numpy array
            else:
                categories_gpu.append(to_gpu(cat))
        n_features = len(model.categories_)

        # Generate column names matching cuML's internal format
        # Always use indexed format to match _data_with_strings_to_cudf_dataframe
        x_cols = [f"{self.x_col}_{i}" for i in range(n_features)]

        # sklearn uses independent encoding, so we always use independent mode
        # This gives exact compatibility with no approximation
        encode_all_per_feature = []
        for i, col in enumerate(x_cols):
            encode_all_i = cudf.DataFrame(
                {
                    col: model.categories_[i],
                    self.out_col: model.encodings_[i],
                }
            )
            encode_all_per_feature.append(encode_all_i)

        # For single feature, also set encode_all as DataFrame for compatibility
        if n_features == 1:
            encode_all = encode_all_per_feature[0]
            independent_mode = False
        else:
            encode_all = encode_all_per_feature
            independent_mode = True

        return {
            "encode_all": encode_all,
            "_encode_all_per_feature": encode_all_per_feature,
            "_encodings_per_feature": [
                to_gpu(enc) for enc in model.encodings_
            ],
            "_independent_mode_fitted": independent_mode,
            "categories_": categories_gpu,
            "_x_cols": x_cols,
            "n_features_in_": n_features,
            "_n_features_out": n_features,  # sklearn always uses independent mode
            "mean": float(model.target_mean_),
            "y_stat_val": float(model.target_mean_),
            "_fitted": True,
            "train": None,
            "train_encode": None,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        """Convert cuML TargetEncoder attributes to sklearn format.

        sklearn expects independent per-feature encodings. If cuML was fitted
        in independent mode, we have exact encodings. Multi-feature combination
        mode cannot be converted to sklearn.
        """
        # Handle categories that may already be numpy arrays (from string columns)
        categories_cpu = [
            cat if isinstance(cat, np.ndarray) else to_cpu(cat)
            for cat in self.categories_
        ]
        n_features = len(self.categories_)

        # Use per-feature encodings if available (from independent mode or sklearn)
        if hasattr(self, "_encodings_per_feature"):
            encodings_list = [
                to_cpu(enc) for enc in self._encodings_per_feature
            ]
        elif n_features == 1:
            # Single feature: extract encodings directly (exact conversion)
            col = self._x_cols[0]
            cats = self.categories_[0]
            feature_encodings = []
            for cat_val in cats:
                mask = self.encode_all[col] == cat_val
                if mask.any():
                    enc_val = float(
                        self.encode_all.loc[mask, self.out_col].iloc[0]
                    )
                else:
                    enc_val = float(self.mean)
                feature_encodings.append(enc_val)
            encodings_list = [np.array(feature_encodings)]
        else:
            # Multi-feature combination mode cannot be converted to sklearn
            raise ValueError(
                "Cannot convert multi-feature cuML TargetEncoder fitted with "
                "multi_feature_mode='combination' to sklearn. Use "
                "multi_feature_mode='independent' for sklearn compatibility."
            )

        return {
            "encodings_": encodings_list,
            "categories_": categories_cpu,
            "target_mean_": float(self.mean),
            # sklearn internal attributes needed for transform
            "_infrequent_enabled": False,
            "target_type_": "continuous",  # cuML only supports continuous targets
            **super()._attrs_to_cpu(model),
        }
