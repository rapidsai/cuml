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
from cuml.preprocessing.encoders import OneHotEncoder as cumlOHE


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

    class LocalOneHotEncoder(cumlOHE):
        def __init__(self, client=None, **kwargs):
            super().__init__(**kwargs)
            self.client = client

        def _check_input_fit(self, X, is_categories=False):
            """Helper function to check input of fit within the local model"""
            if isinstance(X, dask.array.core.Array):
                self._set_input_type('array')
                if is_categories:
                    X = X.transpose()
                return to_dask_cudf(X, client=self.client)
            else:
                self._set_input_type('df')
                return X

        @staticmethod
        def _unique(inp):
            return inp.unique().compute()

        @staticmethod
        def _has_unknown(X_cat, encoder_cat):
            return not X_cat.isin(encoder_cat).all().compute()

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
        el = first(X) if isinstance(X, Sequence) else X
        self.datatype = ('cudf' if isinstance(el, (dcDataFrame, daskSeries))
                         else 'cupy')

        self.local_model = self.LocalOneHotEncoder(**self.kwargs).fit(X)

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
