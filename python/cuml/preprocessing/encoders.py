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
import numpy as np
import cupy as cp
from sklearn.exceptions import NotFittedError

from cuml.preprocessing import LabelEncoder
from cudf import DataFrame, Series
from cudf.core import GenericIndex

from cuml.utils import with_cupy_rmm


class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array.
    The input to this transformer should be a cuDF.DataFrame of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array (depending on the ``sparse``
    parameter).
    By default, the encoder derives the categories based on the unique values
    in each feature. Alternatively, you can also specify the `categories`
    manually.
    Note: a one-hot encoding of y labels should use a LabelBinarizer
    instead.

    Parameters
    ----------
    categories : 'auto' an cupy.ndarray or a cudf.DataFrame, default='auto'
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - DataFrame/Array : ``categories[col]`` holds the categories expected in the
          feature col.
    drop : 'first', None, a dict or a list, default=None
        Specifies a methodology to use to drop one of the categories per
        feature. This is useful in situations where perfectly collinear
        features cause problems, such as when feeding the resulting data
        into a neural network or an unregularized regression.
        - None : retain all features (the default).
        - 'first' : drop the first category in each feature. If only one
          category is present, the feature will be dropped entirely.
        - dict/list : ``drop[col]`` is the category in feature col that
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
    def __init__(self, categories='auto', drop=None, sparse=True,
                 dtype=np.float, handle_unknown='error'):
        self.categories = categories
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.drop = drop
        self._fitted = False
        self.drop_idx_ = None
        self._features = None
        self._encoders = None
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
            msg = ("This OneHotEncoder instance is not fitted yet. Call 'fit' "
                   "with appropriate arguments before using this estimator.")
            raise NotFittedError(msg)

    def _take_feature(self, collection, key):
        """Helper to handle both df and array as input"""
        if self.input_type == 'df':
            return collection[key]
        else:
            return collection[:, key]

    def _compute_drop_idx(self):
        """Helper to compute indices to drop from category to drop"""
        if self.drop is None:
            return None
        elif isinstance(self.drop, str) and self.drop == 'first':
            if self.input_type == 'df':
                return {feature: 0 for feature in self._encoders.keys()}
            else:
                return cp.zeros(shape=(len(self._encoders),), dtype=cp.int32)
        elif isinstance(self.drop, (dict, list)):
            if self.input_type == 'df':
                drop_columns = self.drop.keys()
                drop_idx = dict()
                make_collection, get_size = Series, len
            else:
                drop_columns = range(len(self.drop))
                drop_idx = cp.empty(shape=(len(drop_columns),), dtype=cp.int32)
                make_collection, get_size = cp.array, cp.size

            if len(drop_columns) != len(self._encoders):
                msg = ("`drop` should have as many columns as the number "
                       "of features ({}), got {}")
                raise ValueError(msg.format(len(self._encoders),
                                            len(drop_columns)))
            for feature in drop_columns:
                drop_feature = make_collection(self.drop[feature])
                if get_size(drop_feature) != 1:
                    msg = ("Trying to drop multiple values for feature {}, "
                           "this is not supported.").format(feature)
                    raise ValueError(msg)
                cats = self._encoders[feature].classes_
                if not self.isin(drop_feature, cats).all():
                    msg = ("Some categories for feature {} were supposed "
                           "to be dropped, but were not found in the encoder "
                           "categories.".format(feature))
                    raise ValueError(msg)
                idx = self.isin(cats, drop_feature)
                cats = Series(cats)
                idx_val = cats[idx].index.values
                if self.input_type == 'array':
                    idx_val = idx_val[0]
                drop_idx[feature] = idx_val
            return drop_idx
        else:
            msg = ("Wrong input for parameter `drop`. Expected "
                   "'first', None, a dict or a list, got {}")
            raise ValueError(msg.format(type(self.drop)))

    @property
    def categories_(self):
        """
        Returns categories used for the one hot encoding in the order used by
        transform.
        """
        return [self._encoders[f].classes_ for f in self._features]

    def _set_input_type(self, X):
        if isinstance(X, cp.ndarray):
            self.input_type = 'array'
            self.isin = cp.isin
        elif isinstance(X, DataFrame):
            self.input_type = 'df'
            self.isin = lambda a, b: Series(a).isin(b)
        else:
            raise TypeError(
                'Expected input to be cupy.ndarray or cudf.DataFrame, '
                'got {}'.format(type(X)))

    class _ArrayEncoder:
        """Helper for OneHotEncoder.

        This simplified LabelEncoder reflect the same interface
        but using cp.arrays instead of cudf.Series internally.
        """

        def __init__(self, handle_unknown='error'):
            self.classes_ = None
            self.handle_unknown = handle_unknown

        def fit(self, X):
            self.classes_ = cp.unique(X)
            return self

        def transform(self, X):
            sorted_index = cp.searchsorted(self.classes_, X)

            xindex = cp.take(cp.arange(len(self.classes_)), sorted_index)
            mask = self.classes_[xindex] != X

            if mask.any():
                if self.handle_unknown == 'error':
                    raise KeyError("Attempted to encode unseen key")
                else:
                    xindex[mask] = -1

            return xindex

    def _fit_encoders(self, X, categories=None):
        """
        Helper to reduce code duplication in fit method
        """
        fit_from_categories = categories is not None
        _X = categories if fit_from_categories else X

        if self.input_type == 'df':
            _encoders = dict()
            def append(d, k, v): d[k] = v
            Encoder = LabelEncoder
            self._features = X.columns
        else:
            _encoders = list()
            def append(l, _, v): l.append(v)
            Encoder = self._ArrayEncoder
            # used as indices for a list, no need to use a gpu array here
            self._features = np.arange(0, _X.shape[1], dtype=cp.int32)

        for feature in self._features:
            le = Encoder(handle_unknown=self.handle_unknown)
            x_feature = self._take_feature(_X, feature)
            append(_encoders, feature, le.fit(x_feature))

            if fit_from_categories and self.handle_unknown == 'error':
                x_categories = x_feature
                if not self.isin(self._take_feature(X, feature),
                                 x_categories).all():
                    msg = ("Found unknown categories in column {0}"
                           " during fit".format(feature))
                    raise KeyError(msg)
        return _encoders

    def fit(self, X):
        """
        Fit OneHotEncoder to X.
        Parameters
        ----------
        X : cuDF.DataFrame or cupy.ndarray
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """
        self._validate_keywords()

        self._set_input_type(X)

        if type(self.categories) is str and self.categories == 'auto':
            self._encoders = self._fit_encoders(X)
        else:
            _categories = self.categories
            if self.input_type == 'array':
                _categories = _categories.transpose()  # same format as X
            self._encoders = self._fit_encoders(X, categories=_categories)

        self.drop_idx_ = self._compute_drop_idx()
        self._fitted = True
        return self

    def fit_transform(self, X):
        """
        Fit OneHotEncoder to X, then transform X.
        Equivalent to fit(X).transform(X).

        Parameters
        ----------
        X : cudf.DataFrame or cupy.ndarray
            The data to encode.
        Returns
        -------
        X_out : sparse matrix if sparse=True else a 2-d array
            Transformed input.
        """
        return self.fit(X).transform(X)

    @with_cupy_rmm
    def transform(self, X):
        """
        Transform X using one-hot encoding.
        Parameters
        ----------
        X : cudf.DataFrame or cupy.ndarray
            The data to encode.
        Returns
        -------
        X_out : sparse matrix if sparse=True else a 2-d array
            Transformed input.
        """
        self._check_is_fitted()

        cols, rows = list(), list()
        j = 0
        for feature in self._features:
            encoder = self._encoders[feature]

            col_idx = encoder.transform(self._take_feature(X, feature))
            if self.input_type == 'df':
                col_idx = cp.asarray(col_idx.to_gpu_array(fillna="pandas"))

            idx_to_keep = col_idx > -1

            # increase indices to take previous features into account
            col_idx += j

            # Filter out rows with null values
            row_idx = cp.arange(len(X))[idx_to_keep]
            col_idx = col_idx[idx_to_keep]

            if self.drop_idx_ is not None:
                drop_idx = self.drop_idx_[feature] + j
                mask = cp.ones(col_idx.shape, dtype=cp.bool)
                mask[col_idx == drop_idx] = False
                col_idx = col_idx[mask]
                row_idx = row_idx[mask]
                # account for dropped category in indices
                col_idx[col_idx > drop_idx] -= 1
                # account for dropped category in current cats number
                j -= 1
            j += len(encoder.classes_)
            cols.append(col_idx)
            rows.append(row_idx)

        cols = cp.concatenate(cols)
        rows = cp.concatenate(rows)
        val = cp.ones(rows.shape[0], dtype=self.dtype)
        ohe = cp.sparse.coo_matrix((val, (rows, cols)),
                                   shape=(len(X), j),
                                   dtype=self.dtype)

        if not self.sparse:
            ohe = ohe.toarray()

        return ohe

    @with_cupy_rmm
    def inverse_transform(self, X):
        """
        Convert the data back to the original representation.
        In case unknown categories are encountered (all zeros in the
        one-hot encoding), ``None`` is used to represent this category.
        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
            The transformed data.
        Returns
        -------
        X_tr : cudf.DataFrame
            Inverse transformed array.
        """
        self._check_is_fitted()
        if cp.sparse.issparse(X):
            X = X.toarray()

        if self.input_type == 'df':
            result = DataFrame(columns=self._features)
            def add_result_column(res, key, col): res[key] = col

            def dropped_1cat_inverse(value):
                return Series(GenericIndex(value).repeat(X.shape[0]))

            def drop_inverse(enc, drop_index):
                return enc.inverse_transform(Series(drop_index))[0]
        else:
            result = cp.empty(shape=(len(X), len(self._features)))
            def add_result_column(res, key, col): res[:, key] = col

            def dropped_1cat_inverse(value):
                return cp.full(len(X), value.item(), dtype=self.dtype)

            def drop_inverse(enc, drop_index):
                return enc.classes_[drop_index]

        j = 0
        for feature in self._features:
            feature_enc = self._encoders[feature]
            cats = feature_enc.classes_

            if self.drop is not None:
                # Remove dropped categories
                drop_idx = self.drop_idx_[feature]
                dropped_class_mask = self.isin(cats, cats[drop_idx])
                if len(cats) == 1:
                    # if there is only one category and we drop it, then we
                    # know that the full inverse column is this category
                    inv = dropped_1cat_inverse(cats[0])
                    add_result_column(result, feature, inv)
                    continue
                cats = cats[~dropped_class_mask]

            enc_size = len(cats)
            x_feature = X[:, j:j + enc_size]
            idx = cp.argmax(x_feature, axis=1)
            inv = cats[idx]
            if self.input_type == 'df':
                inv = Series(cats[idx]).reset_index(drop=True)

            if self.handle_unknown == 'ignore':
                not_null_idx = x_feature.any(axis=1)
                if not_null_idx.any():
                    if self.input_type == 'array':
                        raise ValueError('Found an unknown category during '
                                         'inverse_transform, which is not '
                                         'supported with cupy arrays')
                    inv[~not_null_idx] = None
            elif self.drop is not None:
                # drop will either be None or handle_unknown will be error. If
                # self.drop is not None, then we can safely assume that all of
                # the nulls in each column are the dropped value
                dropped_mask = cp.asarray(x_feature.sum(axis=1) == 0).flatten()
                if dropped_mask.any():
                    drop_idx = self.drop_idx_[feature]
                    inv[dropped_mask] = drop_inverse(feature_enc, drop_idx)

            add_result_column(result, feature, inv)
            j += enc_size
        return result
