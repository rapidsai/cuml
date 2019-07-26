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

import numpy as np
import cudf
import dask_cudf

import nvcategory
from librmm_cffi import librmm


def _enforce_str(y):
    ''' Check and convert to string '''
    if y.dtype != "object":
        return y.astype("str")
    return y


def _check_npint32(y: cudf.Series) -> cudf.Series:
    if y.dtype != np.int32:
        return y.astype(np.int32)
    return y


def _trans(ser, categories):
    ''' Helper function to encode cudf.Series with keys provided in nvcategory

    Parameters
    ----------
    ser : cudf.Series
        The series to be encoded
    categories : nvcategory.nvcategory
        Nvcategory that contains the keys to encode ser

    Returns
    -------
    ser : cudf.Series
        Encoded series
    '''
    encoded = (nvcategory
               .from_strings(ser.data)
               .set_keys(categories.keys()))
    device_array = librmm.device_array(len(ser.data), dtype=np.int32)
    encoded.values(devptr=device_array.device_ctypes_pointer.value)
    ser = cudf.Series(device_array)
    return ser


# def _trans_back(ser, categories, orig_dtype):
#     ''' Helper function to revert encoded label to original label

#     Parameters
#     ----------
#     ser : cudf.Series, dtype=int32
#         The series to be reverted
#     categories : nvcategory.nvcategory
#         Nvcategory that contains the keys to encoding

#     Returns
#     -------
#     reverted : cudf.Series
#         Reverted series
#     '''
#     # Since inverse_transform is done by replacing ordinal label with
#     # corresponding string label, it is important to sort the ordinal
#   # from high to low, and process in this order. Otherwise, the ordinal label
#     # may be messed up.
#     # e.g. if ordinal label '0' is replaced first, '10' will be messed up
#     # and become '1label_of_zero' instead of 'label_of_ten'
#     sorted_ord_label = ser.unique().sort_values(ascending=False)

#     # nvstrings.replace() doesn't take nvstrings as param, so need to_host()
#     keys = categories.keys().to_host()
#     # convert ordinal labels to nvstrings, and apply .replace() later
#     reverted = ser.astype('str').data

#     for ord_int in sorted_ord_label:
#         ord_str = str(ord_int)
#         if ord_int < 0 or ord_int >= len(categories.keys()):
#            raise ValueError('Input label {} is out of bound'.format(ord_int))
#         reverted = reverted.replace(ord_str, keys[ord_int])

#     reverted = cudf.Series(reverted, dtype=orig_dtype)
#     return reverted


class LabelEncoder(object):
    ''' Encode labels with value between 0 and n_classes-1

    Examples
    --------
    >>> from dask_cuml.preprocessing import LabelEncoder
    >>> import dask_cudf
    >>> import cudf

    >>> data = cudf.Series(['a', 'b', 'a', 'b', 'c', 'd', 'a'])
    >>> data = dask_cudf.from_cudf(data, npartitions=2)

    >>> le = LabelEncoder()
    >>> le.fit(data)
    >>> print(le._cats.keys())
    ['a', 'b', 'c', 'd']

    >>> encoded = le.transform(data)
    >>> print(encoded.compute())
    0    0
    1    1
    2    0
    3    1
    0    2
    1    3
    2    0
    dtype: int32

    >>> ord_label = cudf.Series([0, 0, 1, 2, 1])
    >>> ord_label = dask_cudf.from_cudf(data, npartitions=2)
    >>> str_label = le.inverse_transform(ord_label)
    >>> print(type(str_label))
    <class 'cudf.dataframe.series.Series'>

    >>> print(str_label)
    0    a
    1    a
    2    b
    3    c
    4    b
    dtype: object
    '''

    def __init__(self, *args, **kwargs):
        self._cats = None
        self._dtype = None
        self._fitted = False

    def _check_is_fitted(self):
        ''' Check whether the LabelEncoder object has been fitted and
        is ready to transform, raise a ValueError otherwise.
        '''
        if (not self._fitted) or (self._cats is None):
            raise ValueError('LabelEncoder must be fit first')

    def fit(self, y):
        ''' Fit label encoder

        Parameters
        ----------
        y : dask_cudf.Series or cudf.Series
            Target values

        Returns
        -------
        self : returns an instance of self
        '''
        if isinstance(y, dask_cudf.Series):
            y = y.map_partitions(_enforce_str)
            self._cats = nvcategory.from_strings(y.compute().data)
        elif isinstance(y, cudf.Series):
            y = _enforce_str(y)
            self._cats = nvcategory.from_strings(y.data)
        else:
            raise TypeError('Input of type {} is not dask_cudf.Series '
                            + 'or cudf.Series'.format(type(y)))
        self._fitted = True
        self._dtype = y.dtype

        return self

    def transform(self, y):
        ''' Transform labels to normalized encoding

        Parameters
        ----------
        y : dask_cudf.Series or cudf.Series
            Target values

        Returns
        -------
        encoded : dask_cudf.Series or cudf.Series
            Encoded labels
        '''
        self._check_is_fitted()

        if isinstance(y, dask_cudf.Series):
            y = y.map_partitions(_enforce_str)
            encoded = y.map_partitions(_trans, self._cats)
            if len(encoded[encoded == -1].compute()) != 0:
                raise ValueError('contains previously unseen labels')

        elif isinstance(y, cudf.Series):
            y = _enforce_str(y)
            encoded = _trans(y, self._cats)
            if -1 in encoded:
                raise ValueError('contains previously unseen labels')

        else:
            raise TypeError(
                'Input of type {} is not dask_cudf.Series '
                + 'or cudf.Series'.format(type(y)))
        return encoded

    def fit_transform(self, y):
        ''' Fit label encoder and return encoded labels

        Parameters
        ----------
        y : dask_cudf.Series or cudf.Series
            Target values

        Returns
        -------
        encoded : dask_cudf.Series or cudf.Series
            Encoded labels
        '''
        if isinstance(y, dask_cudf.Series):
            y = y.map_partitions(_enforce_str)
            self._cats = nvcategory.from_strings(y.compute().data)
            self._fitted = True

            encoded = y.map_partitions(_trans, self._cats)
            if len(encoded[encoded == -1].compute()) != 0:
                raise ValueError('contains previously unseen labels')

        elif isinstance(y, cudf.Series):
            y = _enforce_str(y)
            self._cats = nvcategory.from_strings(y.data)
            self._fitted = True

            encoded = _trans(y, self._cats)
            if -1 in encoded:
                raise ValueError('contains previously unseen labels')
        else:
            raise TypeError(
                'Input of type {} is not dask_cudf.Series '
                + 'or cudf.Series'.format(type(y)))

        self._dtype = y.dtype
        return encoded

    def inverse_transform(self, y):
        ''' Revert ordinal label to original label

        Parameters
        ----------
        y : dask_cudf.Series or cudf.Series, dtype=int32
            Ordinal labels waited to be reverted

        Returns
        -------
        reverted : cudf.Series
            Reverted labels
        '''
        self._check_is_fitted()

        if isinstance(y, dask_cudf.Series):
            y = y.compute()         # convert to cudf.Series

        if isinstance(y, cudf.Series):
            # check if ord_label out of bound
            ord_label = y.unique()
            category_num = len(self._cats.keys())
            for ordi in ord_label:
                if ordi < 0 or ordi >= category_num:
                    raise ValueError(
                        'y contains previously unseen label {}'.format(ordi))

            # check if y's dtype is np.int32, otherwise convert it
            y = _check_npint32(y)
            # convert ordinal label to string label
            reverted = cudf.Series(self._cats.gather_strings(
                y.data.mem.device_ctypes_pointer.value, len(y)))

        else:
            raise TypeError(
                'Input of type {} is not dask_cudf.Series '
                + 'or cudf.Series'.format(type(y)))

        return reverted
