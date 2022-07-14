# Original authors from Sckit-Learn:
#          Arnaud Joly,
#          Joel Nothman,
#          Hamzeh Alsalhi
# License: BSD 3 clause


# This code originates from the Scikit-Learn library,
# it was since modified to allow GPU acceleration.
# This code is under BSD 3 clause license.
# Authors mentioned above do not endorse or promote this production.

from collections.abc import Sequence
from itertools import chain
import warnings

import cupy as cp
import cupyx.scipy.sparse as sp


def _is_integral_float(y):
    return y.dtype.kind == "f" and cp.all(y.astype(int) == y)


def is_multilabel(y):
    """Check if ``y`` is in a multilabel format.
    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Target values.
    Returns
    -------
    out : bool
        Return ``True``, if ``y`` is in a multilabel format, else ```False``.
    Examples
    --------
    >>> import numpy as cp
    >>> from sklearn.utils.multiclass import is_multilabel
    >>> is_multilabel([0, 1, 0, 1])
    False
    >>> is_multilabel([[1], [0, 2], []])
    False
    >>> is_multilabel(cp.array([[1, 0], [0, 0]]))
    True
    >>> is_multilabel(cp.array([[1], [0], [0]]))
    False
    >>> is_multilabel(cp.array([[1, 0, 0]]))
    True
    """
    if hasattr(y, "__array__") or isinstance(y, Sequence):
        # DeprecationWarning will be replaced by ValueError, see NEP 34
        # https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
        with warnings.catch_warnings():
            warnings.simplefilter("error", cp.VisibleDeprecationWarning)
            try:
                y = cp.asarray(y)
            except cp.VisibleDeprecationWarning:
                # dtype=object should be provided explicitly for ragged arrays,
                # see NEP 34
                y = cp.array(y, dtype=object)

    if not (hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1):
        return False

    if sp.issparse(y):
        if isinstance(y, (sp.dok_matrix, sp.lil_matrix)):
            y = y.tocsr()
        return (
            len(y.data) == 0
            or cp.unique(y.data).size == 1
            and (
                y.dtype.kind in "biu"
                or _is_integral_float(cp.unique(y.data))  # bool, int, uint
            )
        )
    else:
        labels = cp.unique(y)

        return len(labels) < 3 and (
            y.dtype.kind in "biu" or _is_integral_float(labels)  # bool, int, uint
        )


def type_of_target(y):
    """Determine the type of data indicated by the target.
    Note that this type is the most specific type that can be inferred.
    For example:
        * ``binary`` is more specific but compatible with ``multiclass``.
        * ``multiclass`` of integers is more specific but compatible with
          ``continuous``.
        * ``multilabel-indicator`` is more specific but compatible with
          ``multiclass-multioutput``.
    Parameters
    ----------
    y : array-like
    Returns
    -------
    target_type : str
        One of:
        * 'continuous': `y` is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': `y` is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': `y` contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': `y` contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': `y` is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': `y` is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': `y` is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.
    Examples
    --------
    >>> from sklearn.utils.multiclass import type_of_target
    >>> import numpy as cp
    >>> type_of_target([0.1, 0.6])
    'continuous'
    >>> type_of_target([1, -1, -1, 1])
    'binary'
    >>> type_of_target(['a', 'b', 'a'])
    'binary'
    >>> type_of_target([1.0, 2.0])
    'binary'
    >>> type_of_target([1, 0, 2])
    'multiclass'
    >>> type_of_target([1.0, 0.0, 3.0])
    'multiclass'
    >>> type_of_target(['a', 'b', 'c'])
    'multiclass'
    >>> type_of_target(cp.array([[1, 2], [3, 1]]))
    'multiclass-multioutput'
    >>> type_of_target([[1, 2]])
    'multilabel-indicator'
    >>> type_of_target(cp.array([[1.5, 2.0], [3.0, 1.6]]))
    'continuous-multioutput'
    >>> type_of_target(cp.array([[0, 1], [1, 1]]))
    'multilabel-indicator'
    """
    valid = (
        isinstance(y, Sequence) or sp.issparse(y) or hasattr(y, "__array__")
    ) and not isinstance(y, str)

    if not valid:
        raise ValueError(
            "Expected array-like (array or non-string sequence), got %r" % y
        )

    sparse_pandas = y.__class__.__name__ in ["SparseSeries", "SparseArray"]
    if sparse_pandas:
        raise ValueError("y cannot be class 'SparseSeries' or 'SparseArray'")

    if is_multilabel(y):
        return "multilabel-indicator"

    # DeprecationWarning will be replaced by ValueError, see NEP 34
    # https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
    with warnings.catch_warnings():
        warnings.simplefilter("error", cp.VisibleDeprecationWarning)
        try:
            y = cp.asarray(y)
        except cp.VisibleDeprecationWarning:
            # dtype=object should be provided explicitly for ragged arrays,
            # see NEP 34
            y = cp.asarray(y, dtype=object)

    # The old sequence of sequences format
    try:
        if (
            not hasattr(y[0], "__array__")
            and isinstance(y[0], Sequence)
            and not isinstance(y[0], str)
        ):
            raise ValueError(
                "You appear to be using a legacy multi-label data"
                " representation. Sequence of sequences are no"
                " longer supported; use a binary array or sparse"
                " matrix instead - the MultiLabelBinarizer"
                " transformer can convert to this format."
            )
    except IndexError:
        pass

    # Invalid inputs
    if y.ndim > 2 or (y.dtype == object and len(y) and not isinstance(y.flat[0], str)):
        return "unknown"  # [[[1, 2]]] or [obj_1] and not ["label_1"]

    if y.ndim == 2 and y.shape[1] == 0:
        return "unknown"  # [[]]

    if y.ndim == 2 and y.shape[1] > 1:
        suffix = "-multioutput"  # [[1, 2], [1, 2]]
    else:
        suffix = ""  # [1, 2, 3] or [[1], [2], [3]]

    # check float and contains non-integer float values
    if y.dtype.kind == "f" and cp.any(y != y.astype(int)):
        # [.1, .2, 3] or [[.1, .2, 3]] or [[1., .2]] and not [1., 2., 3.]
        _assert_all_finite(y)
        return "continuous" + suffix

    if (len(cp.unique(y)) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
        return "multiclass" + suffix  # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
    else:
        return "binary"  # [1, 2] or [["a"], ["b"]]
