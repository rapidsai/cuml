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

from cuml.common.memory_utils import with_cupy_rmm
from sklearn.exceptions import NotFittedError


class LabelEncoder(object):
    """
    An nvcategory based implementation of ordinal label encoding

    Parameters
    ----------
    handle_unknown : {'error', 'ignore'}, default='error'
        Whether to raise an error or ignore if an unknown categorical feature
        is present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform or inverse transform, the resulting encoding will be null.

    Examples
    --------
    Converting a categorical implementation to a numerical one

    .. code-block:: python

        from cudf import DataFrame, Series

        data = DataFrame({'category': ['a', 'b', 'c', 'd']})

        # There are two functionally equivalent ways to do this
        le = LabelEncoder()
        le.fit(data.category)  # le = le.fit(data.category) also works
        encoded = le.transform(data.category)

        print(encoded)

        # This method is preferred
        le = LabelEncoder()
        encoded = le.fit_transform(data.category)

        print(encoded)

        # We can assign this to a new column
        data = data.assign(encoded=encoded)
        print(data.head())

        # We can also encode more data
        test_data = Series(['c', 'a'])
        encoded = le.transform(test_data)
        print(encoded)

        # After train, ordinal label can be inverse_transform() back to
        # string labels
        ord_label = cudf.Series([0, 0, 1, 2, 1])
        ord_label = dask_cudf.from_cudf(data, npartitions=2)
        str_label = le.inverse_transform(ord_label)
        print(str_label)

    Output:

    .. code-block:: python

        0    0
        1    1
        2    2
        3    3
        dtype: int64

        0    0
        1    1
        2    2
        3    3
        dtype: int32

        category  encoded
        0         a        0
        1         b        1
        2         c        2
        3         d        3

        0    2
        1    0
        dtype: int64

        0    a
        1    a
        2    b
        3    c
        4    b
        dtype: object

    """

    def __init__(self, handle_unknown='error'):
        self.classes_ = None
        self.dtype = None
        self._fitted: bool = False
        self.handle_unknown = handle_unknown

    def _check_is_fitted(self):
        if not self._fitted:
            msg = ("This LabelEncoder instance is not fitted yet. Call 'fit' "
                   "with appropriate arguments before using this estimator.")
            raise NotFittedError(msg)

    def _validate_keywords(self):
        if self.handle_unknown not in ('error', 'ignore'):
            msg = ("handle_unknown should be either 'error' or 'ignore', "
                   "got {0}.".format(self.handle_unknown))
            raise ValueError(msg)

    @with_cupy_rmm
    def fit(self, y):
        """
        Fit a LabelEncoder (nvcategory) instance to a set of categories

        Parameters
        ----------
        y : cudf.Series
            Series containing the categories to be encoded. It's elements
            may or may not be unique

        Returns
        -------
        self : LabelEncoder
            A fitted instance of itself to allow method chaining
        """
        self._validate_keywords()
        self.dtype = y.dtype if y.dtype != cp.dtype('O') else str

        self.classes_ = y.unique()  # dedupe and sort

        self._fitted = True
        return self

    def transform(self, y: cudf.Series) -> cudf.Series:
        """
        Transform an input into its categorical keys.

        This is intended for use with small inputs relative to the size of the
        dataset. For fitting and transforming an entire dataset, prefer
        `fit_transform`.

        Parameters
        ----------
        y : cudf.Series
            Input keys to be transformed. Its values should match the
            categories given to `fit`

        Returns
        -------
        encoded : cudf.Series
            The ordinally encoded input series

        Raises
        ------
        KeyError
            if a category appears that was not seen in `fit`
        """
        self._check_is_fitted()

        y = y.astype('category')

        encoded = y.cat.set_categories(self.classes_)._column.codes

        encoded = cudf.Series(encoded)

        if encoded.has_nulls and self.handle_unknown == 'error':
            raise KeyError("Attempted to encode unseen key")

        return encoded

    def fit_transform(self, y: cudf.Series) -> cudf.Series:
        """
        Simultaneously fit and transform an input

        This is functionally equivalent to (but faster than)
        `LabelEncoder().fit(y).transform(y)`
        """
        self.dtype = y.dtype if y.dtype != cp.dtype('O') else str

        y = y.astype('category')
        self.classes_ = y._column.categories

        self._fitted = True
        return cudf.Series(y._column.codes)

    @with_cupy_rmm
    def inverse_transform(self, y: cudf.Series) -> cudf.Series:
        """
        Revert ordinal label to original label

        Parameters
        ----------
        y : cudf.Series, dtype=int32
            Ordinal labels to be reverted

        Returns
        -------
        reverted : cudf.Series
            Reverted labels
        """
        # check LabelEncoder is fitted
        self._check_is_fitted()
        # check input type is cudf.Series
        if not isinstance(y, cudf.Series):
            raise TypeError(
                'Input of type {} is not cudf.Series'.format(type(y)))

        # check if ord_label out of bound
        ord_label = y.unique()
        category_num = len(self.classes_)
        if self.handle_unknown == 'error':
            for ordi in ord_label:
                if ordi < 0 or ordi >= category_num:
                    raise ValueError(
                        'y contains previously unseen label {}'.format(ordi))

        y = y.astype(self.dtype)

        ran_idx = cudf.Series(cp.arange(len(self.classes_))).astype(self.dtype)

        reverted = y._column.find_and_replace(ran_idx, self.classes_, False)

        return cudf.Series(reverted)
