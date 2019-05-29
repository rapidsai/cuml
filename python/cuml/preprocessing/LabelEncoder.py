import cudf
import nvcategory

from librmm_cffi import librmm
import numpy as np


def _enforce_str(y: cudf.Series) -> cudf.Series:
    """
    Ensure that nvcategory is being given strings
    """
    if y.dtype != "object":
        return y.astype("str")
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
    """

    def __init__(self, *args, **kwargs):
        self._cats: nvcategory.nvcategory = None
        self._dtype = None
        self._fitted: bool = False

    def _check_is_fitted(self):
        if not self._fitted:
            raise TypeError("Model must first be .fit()")

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
        raise NotImplementedError
