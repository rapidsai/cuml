# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import warnings

import cuml
from cuml.common import (
    input_to_host_array,
    input_to_host_array_with_sparse_support,
)
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.mixins import ClassifierMixin


class _BaseMulticlassClassifier(Base, ClassifierMixin):
    """Shared base class for multiclass classifiers"""

    def __init__(
        self,
        estimator,
        *,
        handle=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(
            handle=handle, verbose=verbose, output_type=output_type
        )
        self.estimator = estimator

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "estimator"]

    @property
    @cuml.internals.reflect
    def classes_(self):
        return self.multiclass_estimator.classes_

    @generate_docstring(y="dense_anydtype")
    @cuml.internals.reflect(reset=True)
    def fit(self, X, y) -> "_BaseMulticlassClassifier":
        """
        Fit a multiclass classifier.
        """
        import sklearn.multiclass

        opts = {
            "ovo": sklearn.multiclass.OneVsOneClassifier,
            "ovr": sklearn.multiclass.OneVsRestClassifier,
        }
        if (cls := opts.get(self.strategy)) is None:
            raise ValueError(
                f"Expected `strategy` to be one of {list(opts)}, got {self.strategy}"
            )
        X = input_to_host_array_with_sparse_support(X)
        y = input_to_host_array(y).array

        with cuml.internals.exit_internal_api():
            wrapper = cls(self.estimator, n_jobs=None).fit(X, y)

        self.multiclass_estimator = wrapper
        return self

    @generate_docstring(
        return_values={
            "name": "preds",
            "type": "dense",
            "description": "Predicted values",
            "shape": "(n_samples, 1)",
        }
    )
    @cuml.internals.reflect
    def predict(self, X) -> CumlArray:
        """
        Predict using multi class classifier.
        """
        X = input_to_host_array_with_sparse_support(X)

        with cuml.internals.exit_internal_api():
            return self.multiclass_estimator.predict(X)

    @generate_docstring(
        return_values={
            "name": "results",
            "type": "dense",
            "description": "Decision function values",
            "shape": "(n_samples, 1)",
        }
    )
    @cuml.internals.reflect
    def decision_function(self, X) -> CumlArray:
        """
        Calculate the decision function.
        """
        X = input_to_host_array_with_sparse_support(X)
        with cuml.internals.exit_internal_api():
            return self.multiclass_estimator.decision_function(X)


class MulticlassClassifier(_BaseMulticlassClassifier):
    """
    Wrapper around scikit-learn multiclass classifiers that allows to
    choose different multiclass strategies.

    .. deprecated:: 25.12

        This estimator was deprecated in 25.12 and will be removed in 26.02.
        Please use OneVsOneClassifier or OneVsRestClassifier directly instead.

    The input can be any kind of cuML compatible array, and the output type
    follows cuML's output type configuration rules.

    Before passing the data to scikit-learn, it is converted to host (numpy)
    array. Under the hood the data is partitioned for binary classification,
    and it is transformed back to the device by the cuML estimator. These
    copies back and forth the device and the host have some overhead. For more
    details see issue https://github.com/rapidsai/cuml/issues/2876.

    Parameters
    ----------
    estimator : cuML estimator
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
    strategy: string {'ovr', 'ovo'}, default='ovr'
        Multiclass classification strategy: 'ovr': one vs. rest or 'ovo': one
        vs. one

    Attributes
    ----------
    classes_ : float, shape (`n_classes_`)
        Array of class labels.
    n_classes_ : int
        Number of classes.

    Examples
    --------
    >>> from cuml.linear_model import LogisticRegression
    >>> from cuml.multiclass import MulticlassClassifier
    >>> from cuml.datasets.classification import make_classification

    >>> X, y = make_classification(n_samples=10, n_features=6,
    ...                            n_informative=4, n_classes=3,
    ...                            random_state=137)

    >>> cls = MulticlassClassifier(LogisticRegression(), strategy='ovo')
    >>> cls.fit(X, y)
    MulticlassClassifier(estimator=LogisticRegression())
    >>> cls.predict(X)
    array([1, 1, 0, 1, 1, 1, 2, 2, 1, 2])
    """

    def __init__(
        self,
        estimator,
        *,
        handle=None,
        verbose=False,
        output_type=None,
        strategy="ovr",
    ):
        warnings.warn(
            "MulticlassClassifier was deprecated in version 25.12 and will be "
            "removed in version 26.02. Please use OneVsOneClassifier or "
            "OneVsRestClassifier directly instead.",
            FutureWarning,
        )

        super().__init__(
            estimator, handle=handle, verbose=verbose, output_type=output_type
        )
        self.strategy = strategy

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "strategy"]


class OneVsRestClassifier(_BaseMulticlassClassifier):
    """
    Wrapper around Sckit-learn's class with the same name. The input can be
    any kind of cuML compatible array, and the output type follows cuML's
    output type configuration rules.

    Before passing the data to scikit-learn, it is converted to host (numpy)
    array. Under the hood the data is partitioned for binary classification,
    and it is transformed back to the device by the cuML estimator. These
    copies back and forth the device and the host have some overhead. For more
    details see issue https://github.com/rapidsai/cuml/issues/2876.

    For documentation see `scikit-learn's OneVsRestClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html>`_.

    Parameters
    ----------
    estimator : cuML estimator
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
    >>> from cuml.linear_model import LogisticRegression
    >>> from cuml.multiclass import OneVsRestClassifier
    >>> from cuml.datasets.classification import make_classification

    >>> X, y = make_classification(n_samples=10, n_features=6,
    ...                            n_informative=4, n_classes=3,
    ...                            random_state=137)

    >>> cls = OneVsRestClassifier(LogisticRegression())
    >>> cls.fit(X, y)
    OneVsRestClassifier(estimator=LogisticRegression())
    >>> cls.predict(X)
    array([1, 1, 0, 1, 1, 1, 2, 2, 1, 2])
    """

    strategy = "ovr"


class OneVsOneClassifier(_BaseMulticlassClassifier):
    """
    Wrapper around Sckit-learn's class with the same name. The input can be
    any kind of cuML compatible array, and the output type follows cuML's
    output type configuration rules.

    Before passing the data to scikit-learn, it is converted to host (numpy)
    array. Under the hood the data is partitioned for binary classification,
    and it is transformed back to the device by the cuML estimator. These
    copies back and forth the device and the host have some overhead. For more
    details see issue https://github.com/rapidsai/cuml/issues/2876.

    For documentation see `scikit-learn's OneVsOneClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html>`_.

    Parameters
    ----------
    estimator : cuML estimator
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
    >>> from cuml.linear_model import LogisticRegression
    >>> from cuml.multiclass import OneVsOneClassifier
    >>> from cuml.datasets.classification import make_classification

    >>> X, y = make_classification(n_samples=10, n_features=6,
    ...                            n_informative=4, n_classes=3,
    ...                            random_state=137)

    >>> cls = OneVsOneClassifier(LogisticRegression())
    >>> cls.fit(X, y)
    OneVsOneClassifier(estimator=LogisticRegression())
    >>> cls.predict(X)
    array([1, 1, 0, 1, 1, 1, 2, 2, 1, 2])
    """

    strategy = "ovo"
