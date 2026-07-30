# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import cudf
import cupy as cp
import cupyx.scipy.sparse as cp_sp
import numpy as np
import scipy.sparse as sp

from cuml.internals.base import Base
from cuml.internals.interop import InteropMixin
from cuml.internals.outputs import ClassLabels, mlfunc
from cuml.internals.validation import (
    check_array,
    check_classification_targets,
    check_is_fitted,
)


def _label_binarize(
    y,
    *,
    classes=...,
    neg_label=0,
    pos_label=1,
    sparse_output=False,
    accept_multilabel=True,
):
    """A helper used to implement `label_binarize` and `LabelBinarizer`"""
    if neg_label >= pos_label:
        raise ValueError(
            f"{neg_label=} must be strictly less than {pos_label=}."
        )

    if sparse_output and (pos_label == 0 or neg_label != 0):
        raise ValueError(
            "Sparse binarization is only supported with non "
            "zero pos_label and zero neg_label, got "
            f"{pos_label=} and {neg_label=}"
        )

    if classes is not ...:
        if hasattr(classes, "__cuda_array_interface__"):
            classes = cp.asarray(classes)
        else:
            classes = np.asarray(classes)

    # To account for pos_label == 0 in the dense case
    if pos_switch := pos_label == 0:
        pos_label = -neg_label

    is_multioutput = False

    if sparse_input := (cp_sp.issparse(y) or sp.issparse(y)):
        # Coerce to cupyx.scipy.sparse.csr_matrix
        y = check_array(
            y,
            dtype=("float32", "float64"),
            accept_sparse="csr",
            ensure_min_samples=0,
            ensure_all_finite=False,
        )
        # Ensure y is integral and finite
        check_classification_targets(y.data)
        is_multioutput = len(cp.unique(y.data)) > 2
    else:
        # cudf may coerce the dtype, store the original so we can cast back later
        input_dtype = y.dtype if isinstance(y, np.ndarray) else None

        if not isinstance(y, (cudf.DataFrame, cudf.Series)):
            y = check_array(
                y,
                mem_type=None,
                ensure_2d=False,
                ensure_min_samples=0,
                ensure_all_finite=False,
                input_name="y",
            )
            # If no original dtype found on input, use the coerced one instead
            if input_dtype is None:
                input_dtype = y.dtype

            y = (cudf.DataFrame if y.ndim == 2 else cudf.Series)(
                y, dtype=(np.dtype("O") if y.dtype.kind in "U" else None)
            )
        else:
            y = y.reset_index(drop=True)

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.iloc[:, 0]
        if y.ndim == 1:
            check_classification_targets(y)
        elif y.ndim == 2:
            if y.select_dtypes(exclude=["number", "bool"]).shape[1]:
                is_multioutput = True
            else:
                y = y.to_cupy()
                check_classification_targets(y)
                is_multioutput = len(cp.unique(y)) > 2

    if is_multioutput:
        raise ValueError(
            "Multioutput target data is not supported with label binarization"
        )

    if y.shape[0] == 0:
        raise ValueError("y has 0 samples, while a minimum of 1 is required")

    has_unseen = False
    if y.ndim == 1:
        # binary or multiclass
        # y is a cudf.Series
        y = y.astype("category")
        if classes is ...:
            classes = y.cat.categories
            # XXX: cudf's to_numpy doesn't support conversions for all
            # dtypes. Roundtrip through object dtype when necessary.
            try:
                classes = classes.to_numpy(dtype=input_dtype)
            except NotImplementedError:
                classes = classes.to_numpy(dtype="object")
            # cudf will sometimes translate non-numeric dtypes. Coerce back to
            # the input dtype if the input was originally a numpy array.
            if input_dtype is not None:
                classes = classes.astype(input_dtype, copy=False)
            indices = cp.asarray(y.cat.codes)
            indptr = cp.arange(len(y) + 1)
        else:
            y = y.cat.set_categories(classes)
            if has_unseen := y.has_nulls:
                mask = ~y.isnull()
                indices = cp.asarray(y[mask].cat.codes)
                indptr = cp.concatenate([cp.array([0]), mask.cumsum()])
            else:
                indices = cp.asarray(y.cat.codes)
                indptr = cp.arange(len(y) + 1)

        if len(classes) == 1:
            # Special case binary with 1 class -> neg_label
            y_type = "binary"
            out = cp_sp.csr_matrix((len(y), 1), dtype="float32")
        else:
            y_type = "binary" if len(classes) <= 2 else "multiclass"
            data = cp.full(len(indices), pos_label, dtype="float32")
            out = cp_sp.csr_matrix(
                (data, indices, indptr), shape=(len(y), len(classes))
            )
        if not sparse_output:
            out = out.toarray()
    else:
        # multilabel-indicator
        # y is a cupy.ndarray or cupyx.scipy.sparse.csr_matrix
        y_type = "multilabel-indicator"

        if not accept_multilabel:
            raise ValueError(
                "The object was not fitted with multilabel input."
            )

        if classes is ...:
            classes = np.arange(y.shape[1])
        elif len(classes) != y.shape[1]:
            raise ValueError(
                f"classes {classes} mismatch with the labels "
                f"{np.arange(y.shape[1])} found in the data"
            )

        if sparse_output:
            out = cp_sp.csr_matrix(y.astype("float32"))
            if pos_label != 1:
                out.data = cp.full_like(out.data, pos_label)
        else:
            out = y.toarray() if cp_sp.issparse(y) else y.copy()
            if pos_label != 1:
                out[out != 0] = pos_label

    if not sparse_output:
        if neg_label != 0:
            out[out == 0] = neg_label

        if pos_switch:
            out[out == pos_label] = 0

        out = out.astype("int32", copy=False)

    # XXX: In a binary problem with unseen labels we return a matrix of shape
    # (n_samples, 2), while sklearn returns (n_samples, 1). This is an edge
    # case (binary inputs for `LabelBinarizer` are a bit odd, as are unseen
    # classes. We view the sklearn behavior as a bug (see
    # https://github.com/scikit-learn/scikit-learn/issues/13674), since with
    # their encoding unseen labels are conflated with label 0 rather than
    # encoded as missing via all 0s (as in the multiclass case).
    if y_type == "binary" and not has_unseen:
        out = out[:, [-1]]

    return out, classes, y_type, sparse_input


@mlfunc(preserve_index=True)
def label_binarize(y, classes, neg_label=0, pos_label=1, sparse_output=False):
    """
    Binarize labels in a one-vs-all fashion.

    Parameters
    ----------
    y : array-like or sparse matrix, shape (n_samples,) or (n_samples, n_classes)
        Target values. The 2-d matrix should only contain 0 and 1, in the
        multilabel-indicator format.
    classes : array-like of shape (n_classes,)
        The class labels for each class.
    neg_label : int, default=0
        The value to use for encoding negative labels.
    pos_label : int, default=1
        The value to use for encoding positive labels.
    sparse_output : bool, default=False
        If true, a sparse CSR matrix is returned.

    Returns
    -------
    y : array or sparse matrix, shape (n_samples, n_classes)
        The encoded labels. Will be a sparse matrix if ``sparse_output=True``.
        Shape will be (n_samples, n_classes) for multiclass problems,
        (n_samples, 1) for binary problems with no unseen classes, and
        (n_samples, 2) for binary problems with unseen classes (a minor,
        intentional deviation from sklearn).

    See Also
    --------
    LabelBinarizer : A class version of this function.

    Examples
    --------
    >>> from cuml.preprocessing import label_binarize
    >>> label_binarize([1, 6], classes=[1, 2, 4, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]], dtype=int32)

    Binary targets result in a column vector:

    >>> label_binarize(['a', 'b', 'b', 'a'], classes=['a', 'b'])
    array([[0],
           [1],
           [1],
           [0]], dtype=int32)
    """
    out, _, _, _ = _label_binarize(
        y,
        classes=classes,
        neg_label=neg_label,
        pos_label=pos_label,
        sparse_output=sparse_output,
    )
    return out


class LabelBinarizer(InteropMixin, Base):
    """
    Binarize labels in a one-vs-all fashion.

    Parameters
    ----------
    neg_label : int, default=0
        The value to use for encoding negative labels.
    pos_label : int, default=1
        The value to use for encoding positive labels.
    sparse_output : bool, default=False
        If true, a sparse CSR matrix is returned from ``transform``.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {None, 'input', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (n_classes,)
        Holds the label for each class.
    y_type_ : {'binary', 'multiclass', 'multilabel-indicator'}
        The type of the target data.
    sparse_input_ : bool
        Whether the input data to `fit` was a sparse matrix.

    See Also
    --------
    label_binarize : A function version of this class.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.preprocessing import LabelBinarizer
    >>> y = cp.array([1, 2, 6, 4, 2])
    >>> lb = LabelBinarizer().fit(y)
    >>> lb.classes_
    array([1, 2, 4, 6])
    >>> lb.transform(cp.array([1, 6]))
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]], dtype=int32)

    Binary targets result in a column vector:

    >>> import numpy as np
    >>> lb = LabelBinarizer()
    >>> lb.fit_transform(np.array(['a', 'b', 'b', 'a']))
    array([[0],
           [1],
           [1],
           [0]], dtype=int32)
    """

    _cpu_class_path = "sklearn.preprocessing.LabelBinarizer"

    def __init__(
        self,
        *,
        neg_label=0,
        pos_label=1,
        sparse_output=False,
        verbose=False,
        output_type=None,
    ):
        super().__init__(verbose=verbose, output_type=output_type)
        self.neg_label = neg_label
        self.pos_label = pos_label
        self.sparse_output = sparse_output

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "neg_label",
            "pos_label",
            "sparse_output",
        ]

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "classes_")

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.one_d_array = False
        tags.input_tags.two_d_array = False
        tags.target_tags.one_d_labels = True
        return tags

    @classmethod
    def _params_from_cpu(cls, model):
        return {
            "neg_label": model.neg_label,
            "pos_label": model.pos_label,
            "sparse_output": model.sparse_output,
        }

    def _params_to_cpu(self):
        return {
            "neg_label": self.neg_label,
            "pos_label": self.pos_label,
            "sparse_output": self.sparse_output,
        }

    def _attrs_from_cpu(self, model):
        return {
            "y_type_": model.y_type_,
            "sparse_input_": model.sparse_input_,
            "classes_": model.classes_,
        }

    def _attrs_to_cpu(self, model):
        return {
            "y_type_": self.y_type_,
            "sparse_input_": self.sparse_input_,
            "classes_": self.classes_,
        }

    @mlfunc(set_input_type=True)
    def fit(self, y) -> "LabelBinarizer":
        """
        Fit label binarizer.

        Parameters
        ----------
        y : array of shape [n_samples,] or [n_samples, n_classes]
            Target values. The 2-d matrix should only contain 0 and 1,
            in the multilabel-indicator format.

        Returns
        -------
        self : LabelBinarizer
            Returns the instance itself.
        """
        self.fit_transform(y)
        return self

    @mlfunc(set_input_type=True, preserve_index=True)
    def fit_transform(self, y):
        """
        Fit label binarizer and transform labels to binary labels.

        Parameters
        ----------
        y : array-like or sparse matrix, shape (n_samples,) or (n_samples, n_classes)
            Target values. The 2-d matrix should only contain 0 and 1, in the
            multilabel-indicator format.

        Returns
        -------
        y : array or sparse matrix
            The encoded labels. Shape will be (n_samples, 1) for binary
            classification problems. Will be a sparse matrix if
            ``sparse_output=True``.
        """
        out, classes, y_type, sparse_input = _label_binarize(
            y,
            neg_label=self.neg_label,
            pos_label=self.pos_label,
            sparse_output=self.sparse_output,
        )
        self.classes_ = classes
        self.y_type_ = y_type
        self.sparse_input_ = sparse_input
        return out

    @mlfunc(preserve_index=True)
    def transform(self, y):
        """
        Transform labels to binary labels.

        Parameters
        ----------
        y : array-like or sparse matrix, shape (n_samples,) or (n_samples, n_classes)
            Target values. The 2-d matrix should only contain 0 and 1, in the
            multilabel-indicator format.

        Returns
        -------
        y : array or sparse matrix
            The encoded labels. Will be a sparse matrix if
            ``sparse_output=True``. Shape will be (n_samples, n_classes) for
            multiclass problems, (n_samples, 1) for binary problems with no
            unseen classes, and (n_samples, 2) for binary problems with unseen
            classes (a minor, intentional deviation from sklearn).
        """
        check_is_fitted(self)
        out, _, _, _ = _label_binarize(
            y,
            classes=self.classes_,
            neg_label=self.neg_label,
            pos_label=self.pos_label,
            sparse_output=self.sparse_output,
            accept_multilabel=self.y_type_ == "multilabel-indicator",
        )
        return out

    @mlfunc(preserve_index=True)
    def inverse_transform(self, y, *, threshold=None):
        """
        Transform binary labels back to original labels.

        Parameters
        ----------
        y : array-like or sparse matrix, shape (n_samples, n_classes)
            The encoded target values.
        threshold : float, default=None
            Threshold used in the binary and multilabel-indicator cases.
            If None, the threshold is assumed to be half way between
            ``neg_label`` and ``pos_label``.

        Returns
        -------
        y : array or sparse matrix, shape (n_samples,) or (n_samples, n_classes)
            The original target values.
        """
        check_is_fitted(self)

        # Validate and normalize y to a cupy array or csr_matrix
        y = check_array(
            y,
            ensure_2d=False,
            ensure_min_samples=0,
            accept_sparse="csr",
        )
        if y.ndim == 1:
            y = y[:, None]

        # Ensure shape is valid with fit classes
        n_classes = len(self.classes_)
        if n_classes == 2 and y.shape[1] not in (1, 2):
            raise ValueError(
                f"Expected `y` with 1 or 2 columns, but got {y.shape[1]}"
            )
        elif n_classes != 2 and y.shape[1] != n_classes:
            raise ValueError(
                f"Expected `y` with {n_classes} columns, but got {y.shape[1]}"
            )

        # Transform back to original input
        if self.y_type_ == "multiclass":
            if cp_sp.issparse(y):
                indices = y.argmax(1).flatten()
            else:
                indices = y.argmax(axis=1)
        else:
            if threshold is None:
                threshold = (self.pos_label + self.neg_label) / 2.0

            if cp_sp.issparse(y):
                if threshold > 0:
                    indices = (y > threshold).toarray().view("int8")
                else:
                    # Results in a fully dense output, pre-densify to save memory
                    indices = (y.toarray() > threshold).view("int8")
            else:
                indices = (y > threshold).view("int8")

            if self.y_type_ == "binary":
                if cp_sp.issparse(indices):
                    indices = indices.toarray()
                if y.ndim == 2 and y.shape[1] == 2:
                    indices = indices[:, 1].flatten()
                elif n_classes == 1:
                    indices = cp.zeros(len(y), dtype="int8")
                else:
                    indices = indices.flatten()

        if self.y_type_ in ("binary", "multiclass"):
            return ClassLabels(indices, self.classes_)
        elif self.sparse_input_:
            return cp_sp.csr_matrix(indices.astype("float32", copy=False))
        else:
            return indices.astype("int32", copy=False)
