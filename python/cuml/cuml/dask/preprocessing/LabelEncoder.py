# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
from collections.abc import Sequence

from dask_cudf import DataFrame as dcDataFrame
from dask_cudf import Series as dcSeries
from toolz import first

from cuml.common.exceptions import NotFittedError
from cuml.dask.common.base import (
    BaseEstimator,
    DelayedInverseTransformMixin,
    DelayedTransformMixin,
)
from cuml.preprocessing import LabelEncoder as LE


class LabelEncoder(
    BaseEstimator, DelayedTransformMixin, DelayedInverseTransformMixin
):
    """
    A cuDF-based implementation of ordinal label encoding

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

        >>> from dask_cuda import LocalCUDACluster
        >>> from dask.distributed import Client
        >>> import cudf
        >>> import dask_cudf
        >>> from cuml.dask.preprocessing import LabelEncoder

        >>> import pandas as pd
        >>> pd.set_option('display.max_colwidth', 2000)

        >>> cluster = LocalCUDACluster(threads_per_worker=1)
        >>> client = Client(cluster)
        >>> df = cudf.DataFrame({'num_col':[10, 20, 30, 30, 30],
        ...                    'cat_col':['a','b','c','a','a']})
        >>> ddf = dask_cudf.from_cudf(df, npartitions=2)

        >>> # There are two functionally equivalent ways to do this
        >>> le = LabelEncoder()
        >>> le.fit(ddf.cat_col)  # le = le.fit(data.category) also works
        <cuml.dask.preprocessing.LabelEncoder.LabelEncoder object at 0x...>
        >>> encoded = le.transform(ddf.cat_col)
        >>> print(encoded.compute())
        0    0
        1    1
        2    2
        3    0
        4    0
        dtype: uint8

        >>> # This method is preferred
        >>> le = LabelEncoder()
        >>> encoded = le.fit_transform(ddf.cat_col)
        >>> print(encoded.compute())
        0    0
        1    1
        2    2
        3    0
        4    0
        dtype: uint8

        >>> # We can assign this to a new column
        >>> ddf = ddf.assign(encoded=encoded.values)
        >>> print(ddf.compute())
        num_col cat_col  encoded
        0       10       a        0
        1       20       b        1
        2       30       c        2
        3       30       a        0
        4       30       a        0
        >>> # We can also encode more data
        >>> test_data = cudf.Series(['c', 'a'])
        >>> encoded = le.transform(dask_cudf.from_cudf(test_data,
        ...                                            npartitions=2))
        >>> print(encoded.compute())
        0    2
        1    0
        dtype: uint8

        >>> # After train, ordinal label can be inverse_transform() back to
        >>> # string labels
        >>> ord_label = cudf.Series([0, 0, 1, 2, 1])
        >>> ord_label = le.inverse_transform(
        ...    dask_cudf.from_cudf(ord_label,npartitions=2))

        >>> print(ord_label.compute())
        0    a
        1    a
        2    b
        3    c
        4    b
        dtype: object
        >>> client.close()
        >>> cluster.close()

    """

    def __init__(self, *, client=None, verbose=False, **kwargs):
        super().__init__(client=client, verbose=verbose, **kwargs)

    def fit(self, y):
        """
        Fit a LabelEncoder instance to a set of categories

        Parameters
        ----------
        y : dask_cudf.Series
            Series containing the categories to be encoded. Its elements
            may or may not be unique

        Returns
        -------
        self : LabelEncoder
            A fitted instance of itself to allow method chaining

        Notes
        -----
        Number of unique classes will be collected at the client. It'll
        consume memory proportional to the number of unique classes.
        """
        classes = y.unique().compute().sort_values(ignore_index=True)
        el = first(y) if isinstance(y, Sequence) else y
        self.datatype = (
            "cudf" if isinstance(el, (dcDataFrame, dcSeries)) else "cupy"
        )
        self._set_internal_model(LE(**self.kwargs)._fit(y, classes=classes))
        return self

    def fit_transform(self, y, delayed=True):
        """
        Simultaneously fit and transform an input

        This is functionally equivalent to (but faster than)
        LabelEncoder().fit(y).transform(y)
        """
        return self.fit(y).transform(y, delayed=delayed)

    def transform(self, y, delayed=True):
        """
        Transform an input into its categorical keys.

        This is intended for use with small inputs relative to the size of the
        dataset. For fitting and transforming an entire dataset, prefer
        `fit_transform`.

        Parameters
        ----------
        y : dask_cudf.Series
            Input keys to be transformed. Its values should match the
            categories given to `fit`

        Returns
        -------
        encoded : dask_cudf.Series
            The ordinally encoded input series

        Raises
        ------
        KeyError
            if a category appears that was not seen in `fit`
        """
        if self._get_internal_model() is not None:
            return self._transform(
                y,
                delayed=delayed,
                output_dtype="int32",
                output_collection_type="cudf",
            )
        else:
            msg = (
                "This LabelEncoder instance is not fitted yet. Call 'fit' "
                "with appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg)

    def inverse_transform(self, y, delayed=True):
        """
        Convert the data back to the original representation.
        In case unknown categories are encountered (all zeros in the
        one-hot encoding), ``None`` is used to represent this category.

        Parameters
        ----------
        X : dask_cudf Series
            The string representation of the categories.
        delayed : bool (default = True)
            Whether to execute as a delayed task or eager.

        Returns
        -------
        X_tr : dask_cudf.Series
            Distributed object containing the inverse transformed array.
        """
        if self._get_internal_model() is not None:
            return self._inverse_transform(
                y, delayed=delayed, output_collection_type="cudf"
            )
        else:
            msg = (
                "This LabelEncoder instance is not fitted yet. Call 'fit' "
                "with appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg)
