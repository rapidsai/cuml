#
# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np

from cuml.common.classification import decode_labels
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.interop import InteropMixin, UnsupportedOnCPU
from cuml.internals.outputs import (
    exit_internal_context,
    reflect,
    run_in_internal_context,
)
from cuml.internals.validation import check_cudf, check_is_fitted, check_y


class LabelEncoder(InteropMixin, Base):
    """Encode target labels with values between 0 and n_classes - 1.

    This transformer should be used to encode target values (`y`) and not the
    input `X`.

    Parameters
    ----------
    handle_unknown : {'error', 'ignore'}, default='error'
        Whether to raise an error or ignore if an unknown categorical feature
        is present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform or inverse transform, the resulting encoding will be null.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    classes_ : numpy.ndarray of shape (n_classes,)
        Holds the label for each class.

    Examples
    --------
    >>> import numpy as np
    >>> from cuml.preprocessing import LabelEncoder
    >>> y = np.array(["apple", "apple", "banana", "grape"])
    >>> le = LabelEncoder()
    >>> le.fit_transform(y)
    array([0, 0, 1, 2], dtype=uint8)
    >>> le.classes_
    array(['apple', 'banana', 'grape'], dtype='<U6')
    """

    _cpu_class_path = "sklearn.preprocessing.LabelEncoder"

    def __init__(
        self,
        *,
        handle_unknown="error",
        verbose=False,
        output_type=None,
    ) -> None:
        super().__init__(verbose=verbose, output_type=output_type)
        self.handle_unknown = handle_unknown

    @classmethod
    def _get_param_names(cls):
        return ["handle_unknown", *super()._get_param_names()]

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
        return {}

    def _params_to_cpu(self):
        if self.handle_unknown != "error":
            raise UnsupportedOnCPU(
                f"`handle_unknown={self.handle_unknown}` is not supported"
            )
        return {}

    def _attrs_from_cpu(self, model):
        return {"classes_": model.classes_}

    def _attrs_to_cpu(self, model):
        return {"classes_": self.classes_}

    def _validate_keywords(self):
        if self.handle_unknown not in ("error", "ignore"):
            msg = (
                "handle_unknown should be either 'error' or 'ignore', "
                "got {0}.".format(self.handle_unknown)
            )
            raise ValueError(msg)

    @reflect(reset=True)
    @generate_docstring(
        y="dense_anydtype",
        y_shape="n_samples",
        return_values={
            "name": "y",
            "type": "dense",
            "shape": "n_samples",
            "description": "Encoded labels.",
        },
    )
    def fit_transform(self, y):
        """
        Simultaneously fit and transform an input.

        This is functionally equivalent to (but faster than)
        ``LabelEncoder().fit(y).transform(y)``.
        """
        self._validate_keywords()

        y, classes, index = check_y(
            y,
            ensure_discrete_classes=False,
            return_classes=True,
            return_index=True,
        )
        self.classes_ = classes
        return CumlArray(data=y, index=index)

    @reflect(reset=True)
    @generate_docstring(
        y="dense_anydtype",
        y_shape="n_samples",
        return_values={
            "name": "self",
            "type": "LabelEncoder",
            "description": "Fitted label encoder.",
        },
    )
    def fit(self, y):
        """Fit a LabelEncoder instance to a set of categories."""
        self.fit_transform(y)
        return self

    @reflect
    @generate_docstring(
        y="dense_anydtype",
        y_shape="n_samples",
        return_values={
            "name": "y",
            "type": "dense",
            "shape": "n_samples",
            "description": "Encoded labels.",
        },
    )
    def transform(self, y):
        """
        Transform an input into its categorical keys.

        This is intended for use with small inputs relative to the size of the
        dataset. For fitting and transforming an entire dataset, prefer
        `fit_transform`.
        """
        check_is_fitted(self)

        y = check_cudf(
            y,
            ensure_ndim=1,
            coerce_ndim="warn",
            ensure_min_samples=0,
            input_name="y",
        )
        y = y.astype("category")
        encoded = y.cat.set_categories(self.classes_).cat.codes

        if (
            encoded.hasnans or encoded.has_nulls
        ) and self.handle_unknown == "error":
            y = y.cat.categories
            diff = y[~y.isin(self.classes_)].to_numpy(dtype=object)
            raise ValueError(f"y contains previously unseen labels: {diff!s}")

        return CumlArray(data=encoded.to_cupy(), index=y.index)

    @run_in_internal_context
    @generate_docstring(
        y="dense_anydtype",
        y_shape="n_samples",
        return_values={
            "name": "y_original",
            "type": "dense",
            "shape": "n_samples",
            "description": "Original encoding.",
        },
    )
    def inverse_transform(self, y):
        """Transform labels back to original encoding."""
        check_is_fitted(self)

        codes, index = check_y(
            y, dtype=("i2", "i4", "i8", "u2", "u4", "u8"), return_index=True
        )
        classes = self.classes_
        n_classes = len(self.classes_)

        if len(codes) and (codes.min() < 0 or codes.max() >= n_classes):
            if self.handle_unknown == "error":
                diff = cp.setdiff1d(
                    codes, cp.arange(n_classes, dtype=codes.dtype)
                )
                raise ValueError(
                    f"y contains previously unseen labels: {diff!s}"
                )
            else:
                codes = cp.where(
                    (codes < 0) | (codes >= n_classes),
                    -1,
                    codes,
                )
                classes = np.concatenate([classes, [None]])

        with exit_internal_context():
            output_type = self._get_output_type(y)

        return decode_labels(
            codes, classes, output_type=output_type, index=index
        )
