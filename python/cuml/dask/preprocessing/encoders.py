# Copyright (c) 2020, NVIDIA CORPORATION.
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
import dask
import cupy as cp
import numpy as np
from cudf import Series, DataFrame
from cudf.core import GenericIndex
from cuml.dask.common.part_utils import _extract_partitions
from cuml.utils import with_cupy_rmm
from distributed import default_client
from toolz import first
from dask.distributed.client import Future


class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array.
    The input to this transformer should be a dask_cuDF.DataFrame of ints or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array (depending on the ``sparse``
    parameter).
    By default, the encoder derives the categories based on the unique values
    in each feature. Alternatively, you can also specify the `categories`
    manually.

    Parameters
    ----------
    categories : 'auto' or a dask_cudf.DataFrame, default='auto'
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - DataFrame : ``categories[col]`` holds the categories expected in the
          feature col.
    drop : 'first', None or a dict, default=None
        Specifies a methodology to use to drop one of the categories per
        feature. This is useful in situations where perfectly collinear
        features cause problems, such as when feeding the resulting data
        into a neural network or an unregularized regression.
        - None : retain all features (the default).
        - 'first' : drop the first category in each feature. If only one
          category is present, the feature will be dropped entirely.
        - Dict : ``drop[col]`` is the category in feature col that
          should be dropped.
    sparse : bool, default=False
        This feature was deactivated and will give an exception when True.
        The reason is because sparse matrix are not fully supported by cupy
        yet, causing incorrect values when computing one hot encodings.
        See https://github.com/cupy/cupy/issues/3223
    dtype : number type, default=np.float
        Desired datatype of transform's output.
    handle_unknown : {'error', 'ignore'}, default='error'
        Whether to raise an error or ignore if an unknown categorical feature
        is present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros. In the inverse transform, an unknown category
        will be denoted as None.

    Attributes
    ----------
    drop_idx_ : array of shape (n_features,)
        ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category to
        be dropped for each feature. None if all the transformed features will
        be retained.
    """

    def __init__(self, categories='auto', drop=None, sparse=False,
                 dtype=np.float, handle_unknown='error', client=None):
        self.categories = categories
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.drop = drop
        self._fitted = False
        self.drop_idx_ = None
        self._encoders = None
        self.client_ = client if client is not None else default_client()
        if sparse:
            raise ValueError('Sparse matrix are not fully supported by cupy '
                             'yet, causing incorrect values')
        if sparse and np.dtype(dtype) not in ['f', 'd', 'F', 'D']:
            raise ValueError('Only float32, float64, complex64 and complex128 '
                             'are supported when using sparse')

    def _validate_keywords(self):
        if self.handle_unknown not in ('error', 'ignore'):
            msg = ("handle_unknown should be either 'error' or 'ignore', "
                   "got {0}.".format(self.handle_unknown))
            raise ValueError(msg)
        # If we have both dropped columns and ignored unknown
        # values, there will be ambiguous cells. This creates difficulties
        # in interpreting the model.
        if self.drop is not None and self.handle_unknown != 'error':
            raise ValueError(
                "`handle_unknown` must be 'error' when the drop parameter is "
                "specified, as both would create categories that are all "
                "zero.")

    def _check_is_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model must first be .fit()")

    def _compute_drop_idx(self):
        if self.drop is None:
            return None
        elif isinstance(self.drop, str) and self.drop == 'first':
            return {feature: 0 for feature in self._encoders.keys()}
        elif isinstance(self.drop, dict):
            if len(self.drop.keys()) != len(self._encoders):
                msg = ("`drop` should have as many columns as the number "
                       "of features ({}), got {}")
                raise ValueError(msg.format(len(self._encoders),
                                            len(self.drop.keys())))
            drop_idx = dict()
            for feature in self.drop.keys():
                cats = self._encoders[feature]
                self.drop[feature] = Series(self.drop[feature])
                if len(self.drop[feature]) != 1:
                    msg = ("Trying to drop multiple values for feature {}, "
                           "this is not supported.").format(feature)
                    raise ValueError(msg)
                if not self.drop[feature].isin(cats).all():
                    msg = ("Some categories for feature {} were supposed "
                           "to be dropped, but were not found in the encoder "
                           "categories.".format(feature))
                    raise ValueError(msg)
                idx = cats.isin(self.drop[feature])
                drop_idx[feature] = cp.asarray(cats[idx].index)
            return drop_idx
        else:
            msg = ("Wrong input for parameter `drop`. Expected "
                   "'first', None or a dict, got {}")
            raise ValueError(msg.format(type(self.drop)))

    def get_categories_(self):
        """
        Returns categories used for the one hot encoding in the correct order.

        This copies the categories to the CPU and should only be used to check
        the order of the categories.
        """
        return [self._encoders[f].to_array() for f in self._features]

    def fit(self, X):
        """
        Fit OneHotEncoder to X.
        Parameters
        ----------
        X : dask_cudf.DataFrame
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """
        self._validate_keywords()
        if type(self.categories) is str and self.categories == 'auto':
            self._features = X.columns
            self._encoders = {
                feature: X[feature].unique().compute()
                for feature in X.columns
            }
        else:
            self._features = self.categories.columns
            self._encoders = dict()
            for feature in self.categories.columns:
                self._encoders[feature] = self.categories[
                    feature].unique().compute()
                if self.handle_unknown == 'error':
                    if not X[feature].isin(
                            self._encoders[feature]).all().compute():
                        msg = ("Found unknown categories in column {0}"
                               " during fit".format(feature))
                        raise KeyError(msg)

        self.drop_idx_ = self._compute_drop_idx()
        self._fitted = True
        return self

    def fit_transform(self, X, as_futures=False):
        """
        Fit OneHotEncoder to X, then transform X.
        Equivalent to fit(X).transform(X).

        Parameters
        ----------
        X : dask_cudf.DataFrame
            The data to encode.
        as_futures : bool
            Whether or not to return the one hot encoded arrays
            as futures or to aggregate them in one cupy array.
        Returns
        -------
        out : list of futures if as_futures is True else a cupy array
            Transformed input.
        """
        return self.fit(X).transform(X, as_futures=as_futures)

    @staticmethod
    @with_cupy_rmm
    def _func_xform(classes, df, dtype, drop_idx_, handle_unknown):
        nb_categories = sum(len(s) for s in classes.values())
        ohe = cp.zeros((len(df), nb_categories), dtype=dtype)

        j = 0
        for i, feature in enumerate(df.columns):
            # LabelEncoder part
            y = df[feature]
            y = y.astype('category')
            encoded = y.cat.set_categories(classes[feature])._column.codes
            encoded = Series(encoded)
            if encoded.has_nulls and handle_unknown == 'error':
                raise KeyError("Attempted to encode unseen key")

            # OneHotEncoder part
            col_idx = cp.asarray(encoded.to_gpu_array(fillna='pandas'))
            col_idx += j
            idx_to_keep = col_idx > -1
            row_idx = cp.arange(len(ohe))[idx_to_keep]
            col_idx = col_idx[idx_to_keep]
            ohe[row_idx, col_idx] = 1

            if drop_idx_ is not None:
                drop_idx = drop_idx_[feature]
                drop_idx += j
                mask = cp.ones((ohe.shape[1]), dtype=cp.bool)
                mask[drop_idx] = False
                ohe = ohe[:, mask]
                j -= 1  # account for dropped category in current cats number

            j += len(classes[feature])
        return ohe

    @with_cupy_rmm
    def transform(self, X, as_futures=False):
        """
        Transform X using one-hot encoding.
        Parameters
        ----------
        X : dask_cudf.DataFrame
            The data to encode.
        as_futures : bool
            Whether or not to return the one hot encoded arrays
            as futures or to aggregate them in one cupy array.
        Returns
        -------
        out : list of futures if as_futures is True else a cupy array
            Transformed input.
        """
        self._check_is_fitted()

        parts = self.client_.sync(_extract_partitions, X)

        futures = [
            self.client_.submit(self._func_xform, self._encoders, p,
                                self.dtype, self.drop_idx_,
                                self.handle_unknown)
            for w, p in parts
        ]

        if as_futures:
            return futures
        else:
            onehots = [f.result() for f in self.client_.compute(futures)]
            onehots = cp.concatenate(onehots, axis=0)
            return onehots

    @staticmethod
    def _func_inv_xform(classes, X, drop_idx_, handle_unknown):
        result = DataFrame(columns=classes.keys())
        j = 0
        for feature in classes.keys():
            cats = classes[feature]

            if drop_idx_ is not None:
                # Remove dropped categories
                dropped_class_idx = Series(drop_idx_[feature])
                dropped_class_mask = Series(cats).isin(cats[dropped_class_idx])
                if len(cats) == 1:
                    inv = Series(GenericIndex(cats[0]).repeat(X.shape[0]))
                    result[feature] = inv
                    continue
                cats = cats[~dropped_class_mask]

            enc_size = len(cats)
            x_feature = X[:, j:j + enc_size]
            idx = cp.argmax(x_feature, axis=1)
            inv = Series(cats[idx]).reset_index(drop=True)
            if handle_unknown == 'ignore':
                not_null_idx = x_feature.any(axis=1)
                inv.iloc[~not_null_idx] = None
            elif drop_idx_ is not None:
                # drop will either be None or handle_unknown will be error. If
                # self.drop is not None, then we can safely assume that all of
                # the nulls in each column are the dropped value
                dropped_mask = cp.asarray(x_feature.sum(axis=1) == 0).flatten()
                if dropped_mask.any():
                    categories = classes[feature]
                    inv[dropped_mask] = categories[drop_idx_[feature]][0]
            result[feature] = inv
            j += enc_size
        return result

    @with_cupy_rmm
    def inverse_transform(self, X, as_futures=False):
        """
        Convert the data back to the original representation.
        In case unknown categories are encountered (all zeros in the
        one-hot encoding), ``None`` is used to represent this category.
        Parameters
        ----------
        X : dask.array or list[futures], shape [n_samples, n_encoded_features]
            The transformed data.
        as_futures : bool
            Whether or not to return the resulting dataframes
            as futures or to aggregate them in one dataframe.
        Returns
        -------
        X_tr : cudf.DataFrame or list of futures depending on as_futures
            Inverse transformed array.
        """
        self._check_is_fitted()

        def submit_call(x):
            return self.client_.submit(self._func_inv_xform, self._encoders,
                                       x, self.drop_idx_, self.handle_unknown)

        if isinstance(X, list) and isinstance(first(X), Future):
            futures = [submit_call(p) for p in X]
        elif isinstance(X, dask.array.Array):
            parts = self.client_.sync(_extract_partitions, X)
            futures = [submit_call(p) for w, p in parts]
        else:
            raise TypeError('Incorrect input type for inverse_transform,'
                            'expected list[Future] or dask.array.Array, '
                            'got {}'.format(type(X)))

        if as_futures:
            return futures
        else:
            dataframes = [f.result() for f in self.client_.compute(futures)]
            df = cudf.core.reshape.concat(dataframes).reset_index(drop=True)
            return df
