from cuml.dask.common.base import BaseEstimator
from cuml.dask.common.base import DelayedTransformMixin
from cuml.dask.common.base import DelayedInverseTransformMixin

from toolz import first

from collections.abc import Sequence
from dask_cudf.core import DataFrame as dcDataFrame
from dask_cudf.core import Series as daskSeries
from cuml.common.exceptions import NotFittedError
from cuml.preprocessing import LabelEncoder as LE


class LabelEncoder(BaseEstimator,
                   DelayedTransformMixin,
                   DelayedInverseTransformMixin):
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

        import cudf
        import dask_cudf
        df = cudf.DataFrame({'num_col':[10, 20, 30, 30, 30],
                           'cat_col':['a','b','c','a','a']})
        ddf = dask_cudf.from_cudf(df, npartitions=2)

        # There are two functionally equivalent ways to do this
        le = LabelEncoder()
        le.fit(ddf.cat_col)  # le = le.fit(data.category) also works
        encoded = le.transform(ddf.cat_col)

        print(encoded.compute())

        # This method is preferred
        le = LabelEncoder()
        encoded = le.fit_transform(ddf.cat_col)

        print(encoded.compute())

        # We can assign this to a new column
        ddf = ddf.assign(encoded=encoded.values)
        print(ddf.compute())

        # We can also encode more data
        test_data = cudf.Series(['c', 'a'])
        encoded = le.transform(dask_cudf.from_cudf(test_data, npartitions=2))
        print(encoded.compute())

        # After train, ordinal label can be inverse_transform() back to
        # string labels
        ord_label = cudf.Series([0, 0, 1, 2, 1])
        ord_label = dask_cudf.from_cudf(ord_label, npartitions=2)

        print(ord_label.compute())

    Output:

    .. code-block:: python

        [0 1 2 0 0]

        [0 1 2 0 0]

           num_col cat_col  encoded
        0       10       a        0
        1       20       b        1
        2       30       c        2
        3       30       a        0
        4       30       a        0

        [2 0]

        0    a
        1    a
        2    b
        0    c
        1    b
        dtype: object

    """
    def __init__(self, client=None, verbose=False, **kwargs):
        super(LabelEncoder, self).__init__(client=client,
                                           verbose=verbose,
                                           **kwargs)

    def fit(self, y):
        """
        Fit a LabelEncoder (nvcategory) instance to a set of categories

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
        --------
        Number of unique classes will be collected at the client. It'll
        consume memory proportional to the number of unique classes.
        """
        _classes = y.unique().compute()
        el = first(y) if isinstance(y, Sequence) else y
        self.datatype = ('cudf' if isinstance(el, (dcDataFrame, daskSeries))
                         else 'cupy')
        self._set_internal_model(LE(**self.kwargs).fit(y, _classes=_classes))
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
            return self._transform(y,
                                   delayed=delayed,
                                   output_dtype='int32',
                                   output_collection_type='cudf')
        else:
            msg = ("This LabelEncoder instance is not fitted yet. Call 'fit' "
                   "with appropriate arguments before using this estimator.")
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
            return self._inverse_transform(y,
                                           delayed=delayed,
                                           output_collection_type='cudf')
        else:
            msg = ("This LabelEncoder instance is not fitted yet. Call 'fit' "
                   "with appropriate arguments before using this estimator.")
            raise NotFittedError(msg)
