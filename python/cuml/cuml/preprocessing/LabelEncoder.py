#
# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

from typing import TYPE_CHECKING

from cuml import Base
from cuml._thirdparty.sklearn.utils.validation import check_is_fitted
from cuml.common.exceptions import NotFittedError
from cuml.internals.safe_imports import (
    cpu_only_import,
    cpu_only_import_from,
    gpu_only_import,
)

if TYPE_CHECKING:
    import cudf
    import cupy as cp
    import numpy as np
    from pandas import Series as pdSeries
else:
    cudf = gpu_only_import("cudf")
    cp = gpu_only_import("cupy")
    np = cpu_only_import("numpy")
    pdSeries = cpu_only_import_from("pandas", "Series")


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
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Examples
    --------

    Converting a categorical implementation to a numerical one

    >>> from cudf import DataFrame, Series
    >>> from cuml.preprocessing import LabelEncoder
    >>> data = DataFrame({'category': ['a', 'b', 'c', 'd']})

    >>> # There are two functionally equivalent ways to do this
    >>> le = LabelEncoder()
    >>> le.fit(data.category)  # le = le.fit(data.category) also works
    LabelEncoder()
    >>> encoded = le.transform(data.category)

    >>> print(encoded)
    0    0
    1    1
    2    2
    3    3
    dtype: uint8

    >>> # This method is preferred
    >>> le = LabelEncoder()
    >>> encoded = le.fit_transform(data.category)

    >>> print(encoded)
    0    0
    1    1
    2    2
    3    3
    dtype: uint8

    >>> # We can assign this to a new column
    >>> data = data.assign(encoded=encoded)
    >>> print(data.head())
    category  encoded
    0         a        0
    1         b        1
    2         c        2
    3         d        3

    >>> # We can also encode more data
    >>> test_data = Series(['c', 'a'])
    >>> encoded = le.transform(test_data)
    >>> print(encoded)
    0    2
    1    0
    dtype: uint8

    >>> # After train, ordinal label can be inverse_transform() back to
    >>> # string labels
    >>> ord_label = cudf.Series([0, 0, 1, 2, 1])
    >>> str_label = le.inverse_transform(ord_label)
    >>> print(str_label)
    0    a
    1    a
    2    b
    3    c
    4    b
    dtype: object

    """

    def __init__(
        self,
        *,
        handle_unknown="error",
        handle=None,
        verbose=False,
        output_type=None,
    ) -> None:

        super().__init__(
            handle=handle, verbose=verbose, output_type=output_type
        )

        self.classes_ = None
        self.dtype = None
        self._fitted: bool = False
        self.handle_unknown = handle_unknown

    def __sklearn_is_fitted__(self) -> bool:
        return self.classes_ is not None

    def _validate_keywords(self):
        if self.handle_unknown not in ("error", "ignore"):
            msg = (
                "handle_unknown should be either 'error' or 'ignore', "
                "got {0}.".format(self.handle_unknown)
            )
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
        self._validate_keywords()

        if _classes is None:
            # dedupe and sort
            y = cudf.Series(y).drop_duplicates().sort_values(ignore_index=True)
            self.classes_ = y
        else:
            self.classes_ = _classes

        self.dtype = y.dtype if y.dtype != cp.dtype("O") else str
        return self

    def transform(self, y) -> cudf.Series:
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
        encoded : cudf.Series
            The ordinally encoded input series

        Raises
        ------
        KeyError
            if a category appears that was not seen in `fit`
        """
        check_is_fitted(self)

        y = cudf.Series(y, dtype="category")

        encoded = y.cat.set_categories(self.classes_).cat.codes

        if encoded.hasnans and self.handle_unknown == "error":
            raise KeyError("Attempted to encode unseen key")

        return encoded

    def fit_transform(self, y, z=None) -> cudf.Series:
        """
        Simultaneously fit and transform an input

        This is functionally equivalent to (but faster than)
        `LabelEncoder().fit(y).transform(y)`
        """

        y = cudf.Series(y)
        self.dtype = y.dtype if y.dtype != cp.dtype("O") else str

        y = y.astype("category")
        self.classes_ = y.cat.categories

        return y.cat.codes

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
        check_is_fitted(self)
        # check input type is cudf.Series
        y = cudf.Series(y)

        # check if ord_label out of bound
        ord_label = y.unique()
        category_num = len(self.classes_)
        if self.handle_unknown == "error":
            if not isinstance(ord_label, (cp.ndarray, np.ndarray)):
                ord_label = ord_label.values_host
            for ordi in ord_label:
                if ordi < 0 or ordi >= category_num:
                    raise ValueError(
                        "y contains previously unseen label {}".format(ordi)
                    )

        y = y.astype(self.dtype)

        # TODO: Remove ._column once .replace correctly accepts cudf.Index
        ran_idx = (
            cudf.Index(cp.arange(len(self.classes_)))
            .astype(self.dtype)
            ._column
        )
        res = y.replace(ran_idx, self.classes_)

        return res

    def get_param_names(self):
        return super().get_param_names() + [
            "handle_unknown",
        ]
