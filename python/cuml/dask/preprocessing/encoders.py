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
import dask

from cuml.dask.common.dask_arr_utils import to_dask_cudf
from cuml.utils import with_cupy_rmm

from cuml.dask.common.base import BaseEstimator
from cuml.dask.common.base import DelayedTransformMixin
from cuml.dask.common.base import DelayedInverseTransformMixin

from toolz import first

from collections.abc import Sequence
from dask_cudf.core import DataFrame as dcDataFrame
from dask_cudf.core import Series as daskSeries
from cuml.preprocessing import LabelEncoder


class OneHotEncoder(BaseEstimator, DelayedTransformMixin,
                    DelayedInverseTransformMixin):
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
    categories : 'auto' a dask.array or a dask_cudf.DataFrame, default='auto'
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
    """

    def __init__(self, client=None, verbose=False, **kwargs):
        super(OneHotEncoder, self).__init__(client=client,
                                            verbose=verbose,
                                            **kwargs)

    @property
    def categories_(self):
        """
        Returns categories used for the one hot encoding in the correct order.
        """
        return self.local_model.categories_

    def _mnmg_fit(self, enc, X):
        def _mnmg_check_input(X, is_categories=False):
            if isinstance(X, dask.array.core.Array):
                enc._set_input_type('array')
                if is_categories:
                    X = X.transpose()
                return to_dask_cudf(X, client=self.client)
            else:
                enc._set_input_type('df')
                return X

        enc._validate_keywords()
        X = _mnmg_check_input(X)
        if type(enc.categories) is str and enc.categories == 'auto':
            enc._features = X.columns
            enc._encoders = {
                feature: LabelEncoder(
                    handle_unknown=enc.handle_unknown).fit(
                    X[feature].unique().compute())
                for feature in enc._features
            }
        else:
            enc.categories = _mnmg_check_input(enc.categories, True)
            enc._features = enc.categories.columns
            if len(enc._features) != X.shape[1]:
                raise ValueError(
                    "Shape mismatch: if categories is not 'auto',"
                    " it has to be of shape (n_features, _).")
            enc._encoders = dict()
            for feature in enc._features:
                le = LabelEncoder(handle_unknown=enc.handle_unknown)
                enc._encoders[feature] = le.fit(
                    enc.categories[feature].unique().compute())
                if enc.handle_unknown == 'error':
                    print(type(X), type(enc._encoders[feature].classes_))
                    if not X[feature].isin(
                            enc._encoders[feature].classes_).all().compute():
                        msg = ("Found unknown categories in column {0}"
                               " during fit".format(feature))
                        raise KeyError(msg)
        enc.drop_idx_ = enc._compute_drop_idx()
        enc._fitted = True
        return enc

    @with_cupy_rmm
    def fit(self, X):
        """
        Fit a multi-node multi-gpu OneHotEncoder to X.
        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """
        from cuml.preprocessing.encoders import OneHotEncoder as cumlOHE

        el = first(X) if isinstance(X, Sequence) else X
        self.datatype = ('cudf' if isinstance(el, (dcDataFrame, daskSeries))
                         else 'cupy')

        self.local_model = cumlOHE(**self.kwargs)

        self._mnmg_fit(self.local_model, X)

        return self

    def fit_transform(self, X, delayed=True):
        """
        Fit OneHotEncoder to X, then transform X.
        Equivalent to fit(X).transform(X).

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            The data to encode.
        delayed : bool (default = True)
            Whether to execute as a delayed task or eager.
        Returns
        -------
        out : Dask cuDF DataFrame or CuPy backed Dask Array
            Distributed object containing the transformed data
        """
        return self.fit(X).transform(X, delayed=delayed)

    @with_cupy_rmm
    def transform(self, X, delayed=True):
        """
        Transform X using one-hot encoding.
        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            The data to encode.
        delayed : bool (default = True)
            Whether to execute as a delayed task or eager.
        Returns
        -------
        out : Dask cuDF DataFrame or CuPy backed Dask Array
            Distributed object containing the transformed input.
        """
        return self._transform(X, n_dims=2, delayed=delayed,
                               output_dtype=self.local_model.dtype,
                               output_collection_type='cupy')

    @with_cupy_rmm
    def inverse_transform(self, X, delayed=True):
        """
        Convert the data back to the original representation.
        In case unknown categories are encountered (all zeros in the
        one-hot encoding), ``None`` is used to represent this category.
        Parameters
        ----------
        X : CuPy backed Dask Array, shape [n_samples, n_encoded_features]
            The transformed data.
        delayed : bool (default = True)
            Whether to execute as a delayed task or eager.
        Returns
        -------
        X_tr : Dask cuDF DataFrame or CuPy backed Dask Array
            Distributed object containing the inverse transformed array.
        """
        return self._inverse_transform(X, n_dims=2, delayed=delayed,
                                       output_dtype=self.local_model.dtype,
                                       output_collection_type=self.datatype)
