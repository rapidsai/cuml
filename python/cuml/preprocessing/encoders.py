# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
import cupyx
from cuml.common.exceptions import NotFittedError

from cuml import Base
from cuml.preprocessing import LabelEncoder
from cudf import DataFrame, Series
from cudf import GenericIndex
import cuml.common.logger as logger

import warnings


class OneHotEncoder(Base):
    """
    Encode categorical features as a one-hot numeric array.
    The input to this estimator should be a cuDF.DataFrame or a cupy.ndarray,
    denoting the unique values taken on by categorical (discrete) features.
    The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
    encoding scheme. This creates a binary column for each category and
    returns a sparse matrix or dense array (depending on the ``sparse``
    parameter).
    By default, the encoder derives the categories based on the unique values
    in each feature. Alternatively, you can also specify the `categories`
    manually.

    .. note:: a one-hot encoding of y labels should use a LabelBinarizer
        instead.

    Parameters
    ----------
    categories : 'auto' an cupy.ndarray or a cudf.DataFrame, default='auto'
                 Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.

        - DataFrame/ndarray : ``categories[col]`` holds the categories expected
          in the feature col.

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

    sparse : bool, default=True
        This feature is not fully supported by cupy
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
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    drop_idx_ : array of shape (n_features,)
        ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category to
        be dropped for each feature. None if all the transformed features will
        be retained.

    """
    def __init__(self, *,
                 categories='auto',
                 drop=None,
                 sparse=True,
                 dtype=np.float32,
                 handle_unknown='error',
                 handle=None,
                 verbose=False,
                 output_type=None):
        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)
        self.categories = categories
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.drop = drop
        self._fitted = False
        self.drop_idx_ = None
        self._features = None
        self._encoders = None
        self.input_type = None
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

    def _compute_drop_idx(self):
        """Helper to compute indices to drop from category to drop"""
        if self.drop is None:
            return None
        elif isinstance(self.drop, str) and self.drop == 'first':
            return {feature: 0 for feature in self._encoders.keys()}
        elif isinstance(self.drop, (dict, list)):
            if isinstance(self.drop, list):
                self.drop = dict(zip(range(len(self.drop)), self.drop))
            if len(self.drop.keys()) != len(self._encoders):
                msg = ("`drop` should have as many columns as the number "
                       "of features ({}), got {}")
                raise ValueError(msg.format(len(self._encoders),
                                            len(self.drop.keys())))
            drop_idx = dict()
            for feature in self.drop.keys():
                self.drop[feature] = Series(self.drop[feature])
                if len(self.drop[feature]) != 1:
                    msg = ("Trying to drop multiple values for feature {}, "
                           "this is not supported.").format(feature)
                    raise ValueError(msg)
                cats = self._encoders[feature].classes_
                if not self.drop[feature].isin(cats).all():
                    msg = ("Some categories for feature {} were supposed "
                           "to be dropped, but were not found in the encoder "
                           "categories.".format(feature))
                    raise ValueError(msg)
                cats = Series(cats)
                idx = cats.isin(self.drop[feature])
                drop_idx[feature] = cp.asarray(cats[idx].index)
            return drop_idx
        else:
            msg = ("Wrong input for parameter `drop`. Expected "
                   "'first', None or a dict, got {}")
            raise ValueError(msg.format(type(self.drop)))

    @property
    def categories_(self):
        """
        Returns categories used for the one hot encoding in the correct order.
        """
        return [self._encoders[f].classes_ for f in self._features]

    def _set_input_type(self, value):
        if self.input_type is None:
            self.input_type = value

    def _check_input(self, X, is_categories=False):
        """
        If input is cupy, convert it to a DataFrame with 0 copies
        """
        if isinstance(X, cp.ndarray):
            self._set_input_type('array')
            if is_categories:
                X = X.transpose()
            return DataFrame(X)
        else:
            self._set_input_type('df')
            return X

    def _check_input_fit(self, X, is_categories=False):
        """Helper function used in fit. Can be overridden in subclasses. """
        return self._check_input(X, is_categories=is_categories)

    def _unique(self, inp):
        """Helper function used in fit. Can be overridden in subclasses. """

        # Default implementation passes input through directly since this is
        # performed in `LabelEncoder.fit()`
        return inp

    def _has_unknown(self, X_cat, encoder_cat):
        """Check if X_cat has categories that are not present in encoder_cat"""
        return not X_cat.isin(encoder_cat).all()

    def fit(self, X, y=None):
        """
        Fit OneHotEncoder to X.

        Parameters
        ----------
        X : cuDF.DataFrame or cupy.ndarray, shape = (n_samples, n_features)
            The data to determine the categories of each feature.
        y : None
            Ignored. This parameter exists for compatibility only.

        Returns
        -------
        self

        """
        self._validate_keywords()
        X = self._check_input_fit(X)
        if type(self.categories) is str and self.categories == 'auto':
            self._features = X.columns
            self._encoders = {
                feature: LabelEncoder(handle=self.handle,
                                      verbose=self.verbose,
                                      output_type=self.output_type,
                                      handle_unknown=self.handle_unknown).fit(
                                          self._unique(X[feature]))
                for feature in self._features
            }
        else:
            self.categories = self._check_input_fit(self.categories, True)
            self._features = self.categories.columns
            if len(self._features) != X.shape[1]:
                raise ValueError("Shape mismatch: if categories is not 'auto',"
                                 " it has to be of shape (n_features, _).")
            self._encoders = dict()
            for feature in self._features:

                le = LabelEncoder(handle=self.handle,
                                  verbose=self.verbose,
                                  output_type=self.output_type,
                                  handle_unknown=self.handle_unknown)

                self._encoders[feature] = le.fit(self.categories[feature])

                if self.handle_unknown == 'error':
                    if self._has_unknown(X[feature],
                                         self._encoders[feature].classes_):
                        msg = ("Found unknown categories in column {0}"
                               " during fit".format(feature))
                        raise KeyError(msg)

        self.drop_idx_ = self._compute_drop_idx()
        self._fitted = True
        return self

    def fit_transform(self, X, y=None):
        """
        Fit OneHotEncoder to X, then transform X.
        Equivalent to fit(X).transform(X).

        Parameters
        ----------
        X : cudf.DataFrame or cupy.ndarray, shape = (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X_out : sparse matrix if sparse=True else a 2-d array
            Transformed input.

        """
        X = self._check_input(X)
        return self.fit(X).transform(X)

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
        X = self._check_input(X)

        cols, rows = list(), list()
        col_idx = None
        j = 0

        try:
            for feature in X.columns:
                encoder = self._encoders[feature]
                col_idx = encoder.transform(X[feature])
                idx_to_keep = col_idx.notnull().to_cupy()
                col_idx = col_idx.dropna().to_cupy()

                # Simple test to auto upscale col_idx type as needed
                # First, determine the maximum value we will add assuming
                # monotonically increasing up to len(encoder.classes_)
                # Ensure we dont go negative by clamping to 0
                max_value = int(max(len(encoder.classes_) - 1, 0) + j)

                # If we exceed the max value, upconvert
                if (max_value > np.iinfo(col_idx.dtype).max):
                    col_idx = col_idx.astype(np.min_scalar_type(max_value))
                    logger.debug("Upconverting column: '{}', to dtype: '{}', "
                                 "to support up to {} classes".format(
                                     feature,
                                     np.min_scalar_type(max_value),
                                     max_value))

                # increase indices to take previous features into account
                col_idx += j

                # Filter out rows with null values
                row_idx = cp.arange(len(X))[idx_to_keep]

                if self.drop_idx_ is not None:
                    drop_idx = self.drop_idx_[feature] + j
                    mask = cp.ones(col_idx.shape, dtype=bool)
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
            ohe = cupyx.scipy.sparse.coo_matrix((val, (rows, cols)),
                                                shape=(len(X), j),
                                                dtype=self.dtype)

            if not self.sparse:
                ohe = ohe.toarray()

            return ohe

        except TypeError as e:
            # Append to cols to include the column that threw the error
            cols.append(col_idx)

            # Build a string showing what the types are
            input_types_str = ", ".join([str(x.dtype) for x in cols])

            raise TypeError(
                "A TypeError occurred while calculating column "
                "category indices, most likely due to integer overflow. This "
                "can occur when columns have a large difference in the number "
                "of categories, resulting in different category code dtypes "
                "for different columns."
                "Calculated column code dtypes: {}.\n"
                "Internal Error: {}".format(input_types_str, repr(e)))

    def inverse_transform(self, X):
        """
        Convert the data back to the original representation.
        In case unknown categories are encountered (all zeros in the
        one-hot encoding), ``None`` is used to represent this category.

        The return type is the same as the type of the input used by the first
        call to fit on this estimator instance.

        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
            The transformed data.

        Returns
        -------
        X_tr : cudf.DataFrame or cupy.ndarray
            Inverse transformed array.
        """
        self._check_is_fitted()
        if cupyx.scipy.sparse.issparse(X):
            # cupyx.scipy.sparse 7.x does not support argmax,
            # when we upgrade cupy to 8.x, we should add a condition in the
            # if close: `and not cupyx.scipy.sparse.issparsecsc(X)`
            # and change the following line by `X = X.tocsc()`
            X = X.toarray()
        result = DataFrame(columns=self._encoders.keys())
        j = 0
        for feature in self._encoders.keys():
            feature_enc = self._encoders[feature]
            cats = feature_enc.classes_

            if self.drop is not None:
                # Remove dropped categories
                dropped_class_idx = Series(self.drop_idx_[feature])
                dropped_class_mask = Series(cats).isin(cats[dropped_class_idx])
                if len(cats) == 1:
                    inv = Series(GenericIndex(cats[0]).repeat(X.shape[0]))
                    result[feature] = inv
                    continue
                cats = cats[~dropped_class_mask]

            enc_size = len(cats)
            x_feature = X[:, j:j + enc_size]
            idx = cp.argmax(x_feature, axis=1)
            inv = Series(cats.iloc[idx]).reset_index(drop=True)

            if self.handle_unknown == 'ignore':
                not_null_idx = x_feature.any(axis=1)
                inv.iloc[~not_null_idx] = None
            elif self.drop is not None:
                # drop will either be None or handle_unknown will be error. If
                # self.drop is not None, then we can safely assume that all of
                # the nulls in each column are the dropped value
                dropped_mask = cp.asarray(x_feature.sum(axis=1) == 0).flatten()
                if dropped_mask.any():
                    inv[dropped_mask] = feature_enc.inverse_transform(
                        Series(self.drop_idx_[feature]))[0]

            result[feature] = inv
            j += enc_size
        if self.input_type == 'array':
            try:
                result = result.to_cupy()
            except ValueError:
                warnings.warn("The input one hot encoding contains rows with "
                              "unknown categories. Since device arrays do not "
                              "support null values, the output will be "
                              "returned as a DataFrame "
                              "instead.")
        return result

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of str of shape (n_features,)
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : ndarray of shape (n_output_features,)
            Array of feature names.
        """
        self._check_is_fitted()
        cats = self.categories_
        if input_features is None:
            input_features = ["x%d" % i for i in range(len(cats))]
        elif len(input_features) != len(self.categories_):
            raise ValueError(
                "input_features should have length equal to number of "
                "features ({}), got {}".format(
                    len(self.categories_), len(input_features)
                )
            )

        feature_names = []
        for i in range(len(cats)):
            names = [input_features[i] + "_" + str(t)
                     for t in cats[i].values_host]
            if self.drop_idx_ is not None and self.drop_idx_[i] is not None:
                names.pop(self.drop_idx_[i])
            feature_names.extend(names)

        return np.array(feature_names, dtype=object)

    def get_param_names(self):
        return super().get_param_names() + [
            "categories",
            "drop",
            "sparse",
            "dtype",
            "handle_unknown",
        ]
