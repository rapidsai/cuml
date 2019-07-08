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
import nvcategory
from librmm_cffi import librmm
import numpy as np

import warnings


def _enforce_str(y: cudf.Series) -> cudf.Series:
    """
    Ensure that nvcategory is being given strings
    """
    if y.dtype != "object":
        return y.astype("str")
    return y


def _check_npint32(y: cudf.Series) -> cudf.Series:
    if y.dtype != np.int32:
        return y.astype(np.int32)
    return y


class LabelEncoder(object):
    """
    An nvcategory based implementation of ordinal label encoding

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

    def __init__(self, *args, **kwargs):
        self._cats: nvcategory.nvcategory = None
        self._dtype = None
        self._fitted: bool = False

    def _check_is_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model must first be .fit()")

    def fit(self, y: cudf.Series) -> "LabelEncoder":
        """
        Fit a LabelEncoder (nvcategory) instance to a set of categories

        Parameters
        ---------
        y : cudf.Series
            Series containing the categories to be encoded. It's elements
            may or may not be unique

        Returns
        -------
        self : LabelEncoder
            A fitted instance of itself to allow method chaining
        """
        self._dtype = y.dtype

        y = _enforce_str(y)

        self._cats = nvcategory.from_strings(y.data)
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
        ------
        encoded : cudf.Series
            The ordinally encoded input series

        Raises
        ------
        KeyError
            if a category appears that was not seen in `fit`
        """
        self._check_is_fitted()
        y = _enforce_str(y)
        encoded = cudf.Series(
            nvcategory.from_strings(y.data)
            .set_keys(self._cats.keys())
            .values()
        )
        if -1 in encoded:
            raise KeyError("Attempted to encode unseen key")
        return encoded

    def fit_transform(self, y: cudf.Series) -> cudf.Series:
        """
        Simultaneously fit and transform an input

        This is functionally equivalent to (but faster than)
        `LabelEncoder().fit(y).transform(y)`
        """
        self._dtype = y.dtype

        # Convert y to nvstrings series, if it isn't one
        y = _enforce_str(y)

        # Bottleneck is here, despite everything being done on the device
        self._cats = nvcategory.from_strings(y.data)

        self._fitted = True
        arr: librmm.device_array = librmm.device_array(
            y.data.size(), dtype=np.int32
        )
        self._cats.values(devptr=arr.device_ctypes_pointer.value)
        return cudf.Series(arr)

    def inverse_transform(self, y: cudf.Series) -> cudf.Series:
        ''' Revert ordinal label to original label

        Parameters
        ----------
        y : cudf.Series, dtype=int32
            Ordinal labels to be reverted

        Returns
        -------
        reverted : cudf.Series
            Reverted labels
        '''
        # check LabelEncoder is fitted
        self._check_is_fitted()
        # check input type is cudf.Series
        if not isinstance(y, cudf.Series):
            raise TypeError(
                'Input of type {} is not cudf.Series'.format(type(y)))

        # check if y's dtype is np.int32, otherwise convert it
        y = _check_npint32(y)

        # check if ord_label out of bound
        ord_label = y.unique()
        category_num = len(self._cats.keys())
        for ordi in ord_label:
            if ordi < 0 or ordi >= category_num:
                raise ValueError(
                    'y contains previously unseen label {}'.format(ordi))
        # convert ordinal label to string label
        reverted = self._cats.gather_strings(
            y.data.mem.device_ctypes_pointer.value, len(y))

        # convert to original datatype?
        # if y.dtype != self._dtype:
        #     y = y.astype(self._dtype)

        return reverted
