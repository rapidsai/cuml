#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
import numpy as np
from cuml import Base
import cuml
from cuml.common import input_to_cuml_array
from pandas import Series as pdSeries

from cuml.common.exceptions import NotFittedError


class LabelEncoder(Base):
    """
    An nvcategory based implementation of ordinal label encoding

    Parameters
    ----------
    handle_unknown : {'error', 'ignore'}, default='error'
        Whether to raise an error or ignore if an unknown categorical feature
        is present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform or inverse transform, the resulting encoding will be null.
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

    def __init__(self, *,
                 handle_unknown='error',
                 handle=None,
                 verbose=False,
                 output_type=None):

        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)

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

    def fit(self, y, _classes=None):
        """
        Fit a LabelEncoder (nvcategory) instance to a set of categories

        Parameters
        ----------
        y : cudf.Series, pandas.Series, cupy.ndarray or numpy.ndarray
            Series containing the categories to be encoded. It's elements
            may or may not be unique

        _classes: int or None.
            Passed by the dask client when dask LabelEncoder is used.

        Returns
        -------
        self : LabelEncoder
            A fitted instance of itself to allow method chaining

        """
        if _classes is None:
            y, _ = self._to_cudf_series(y)

        self._validate_keywords()

        self.dtype = y.dtype if y.dtype != cp.dtype('O') else str
        if _classes is not None:
            self.classes_ = _classes
        else:
            self.classes_ = y.unique()  # dedupe and sort

        self._fitted = True
        return self

    @cuml.internals.api_base_return_array(get_output_dtype=True)
    def transform(self, y):
        """
        Transform an input into its categorical keys.

        This is intended for use with small inputs relative to the size of the
        dataset. For fitting and transforming an entire dataset, prefer
        `fit_transform`.

        Parameters
        ----------
        y : cudf.Series, pandas.Series, cupy.ndarray or numpy.ndarray
            Input keys to be transformed. Its values should match the
            categories given to `fit`

        Returns
        -------
        encoded : the same type as y
            The ordinally encoded input series

        Raises
        ------
        KeyError
            if a category appears that was not seen in `fit`
        """
        y, output_type = self._to_cudf_series(y)

        self._check_is_fitted()

        y = y.astype('category')

        encoded = y.cat.set_categories(self.classes_)._column.codes
        encoded = cudf.Series(encoded, index=y.index).astype('int32')

        if encoded.has_nulls and self.handle_unknown == 'error':
            raise KeyError("Attempted to encode unseen key")

        return input_to_cuml_array(encoded).array

    @cuml.internals.api_base_return_array(get_output_dtype=True)
    def fit_transform(self, y, z=None) -> cudf.Series:
        """
        Simultaneously fit and transform an input

        This is functionally equivalent to (but faster than)
        `LabelEncoder().fit(y).transform(y)`
        """

        y, output_type = self._to_cudf_series(y)
        self.dtype = y.dtype if y.dtype != cp.dtype('O') else str

        y = y.astype('category')
        self.classes_ = y._column.categories

        self._fitted = True
        res = cudf.Series(y._column.codes, index=y.index).astype('int32')
        return input_to_cuml_array(res).array

    def inverse_transform(self, y: cudf.Series) -> cudf.Series:
        """
        Revert ordinal label to original label

        Parameters
        ----------
        y : cudf.Series, pandas.Series, cupy.ndarray or numpy.ndarray
            dtype=int32
            Ordinal labels to be reverted

        Returns
        -------
        reverted : the same type as y
            Reverted labels
        """
        # check LabelEncoder is fitted
        self._check_is_fitted()
        # check input type is cudf.Series
        y, output_type = self._to_cudf_series(y)

        # check if ord_label out of bound
        ord_label = y.unique()
        category_num = len(self.classes_)
        if self.handle_unknown == 'error':
            for ordi in ord_label.values_host:
                if ordi < 0 or ordi >= category_num:
                    raise ValueError(
                        'y contains previously unseen label {}'.format(ordi))

        y = y.astype(self.dtype)

        ran_idx = cudf.Series(cp.arange(len(self.classes_))).astype(self.dtype)

        reverted = y._column.find_and_replace(ran_idx, self.classes_, False)

        res = cudf.Series(reverted)
        return res

    def get_param_names(self):
        return super().get_param_names() + [
            "handle_unknown",
        ]

    def _to_cudf_series(self, y):
        if isinstance(y, pdSeries):
            y = cudf.from_pandas(y)
            output_type = 'pandas'
        elif isinstance(y, cp.ndarray):
            y = cudf.Series(y)
            output_type = 'cupy'
        elif isinstance(y, np.ndarray):
            y = cudf.Series(y)
            output_type = 'numpy'
        elif isinstance(y, cudf.Series):
            output_type = 'cudf'
        else:
            msg = ("input should be either 'cupy.ndarray'"
                   " or 'numpy.ndarray' or 'pandas.Series',"
                   " or 'cudf.Series'"
                   "got {0}.".format(type(y)))
            raise TypeError(msg)
        return y, output_type
