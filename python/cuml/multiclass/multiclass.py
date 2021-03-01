# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import cuml.internals
import sklearn.multiclass

from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.mixins import ClassifierMixin
from cuml.common.doc_utils import generate_docstring
from cuml.common import input_to_host_array


class MulticlassClassifier(Base, ClassifierMixin):
    """
    Wrapper around scikit-learn multiclass classifiers that allows to
    choose different multiclass strategies.

    The input can be any kind of cuML compatible array, and the output type
    follows cuML's output type configuration rules.

    Berofe passing the data to scikit-learn, it is converted to host (numpy)
    array. Under the hood the data is partitioned for binary classification,
    and it is transformed back to the device by the cuML estimator. These
    copies back and forth the device and the host have some overhead. For more
    details see issue https://github.com/rapidsai/cuml/issues/2876.

    Examples
    --------

    >>> from cuml.linear_model import LogisticRegression
    >>> from cuml.multiclass import MulticlassClassifier
    >>> from cuml.datasets.classification import make_classification
    >>>
    >>> X, y = make_classification(n_samples=10, n_features=6, n_informative=4,
    ...                            n_classes=3, random_state=137)
    >>>
    >>> cls = MulticlassClassifier(LogisticRegression(), strategy='ovo')
    >>> cls.fit(X,y)
    >>> cls.predict(X)
    array([1, 1, 1, 0, 0, 2, 2, 2, 0, 1])

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
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.
    strategy: string {'ovr', 'ovo'}, default='ovr'
        Multiclass classification strategy: 'ovr': one vs. rest or 'ovo': one
        vs. one

    Attributes
    ----------
    classes_ : float, shape (`n_classes_`)
        Array of class labels.
    n_classes_ : int
        Number of classes.

    """
    def __init__(self,
                 estimator,
                 handle=None,
                 verbose=False,
                 output_type=None,
                 strategy='ovr'):
        super().__init__(handle=handle, verbose=verbose,
                         output_type=output_type)
        self.strategy = strategy
        self.estimator = estimator

    @property
    @cuml.internals.api_base_return_array_skipall
    def classes_(self):
        return self.multiclass_estimator.classes_

    @property
    @cuml.internals.api_base_return_any_skipall
    def n_classes_(self):
        return self.multiclass_estimator.n_classes_

    @generate_docstring(y='dense_anydtype')
    def fit(self, X, y) -> 'MulticlassClassifier':
        """
        Fit a multiclass classifier.
        """
        if self.strategy == 'ovr':
            self.multiclass_estimator = sklearn.multiclass.\
                OneVsRestClassifier(self.estimator, n_jobs=None)
        elif self.strategy == 'ovo':
            self.multiclass_estimator = \
                sklearn.multiclass.OneVsOneClassifier(
                    self.estimator, n_jobs=None)
        else:
            raise ValueError('Invalid multiclass strategy ' +
                             str(self.strategy) + ', must be one of '
                             '{"ovr", "ovo"}')
        X, _, _, _, _ = input_to_host_array(X)
        y, _, _, _, _ = input_to_host_array(y)
        with cuml.internals.exit_internal_api():
            self.multiclass_estimator.fit(X, y)
            return self

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, 1)'})
    def predict(self, X) -> CumlArray:
        """
        Predict using multi class classifier.
        """
        X, _, _, _, _ = input_to_host_array(X)
        with cuml.internals.exit_internal_api():
            return self.multiclass_estimator.predict(X)

    @generate_docstring(return_values={'name': 'results',
                                       'type': 'dense',
                                       'description': 'Decision function \
                                       values',
                                       'shape': '(n_samples, 1)'})
    def decision_function(self, X) -> CumlArray:
        """
        Calculate the decision function.
        """
        X, _, _, _, _ = input_to_host_array(X)
        with cuml.internals.exit_internal_api():
            return self.multiclass_estimator.decision_function(X)

    def get_param_names(self):
        return super().get_param_names() + ['estimator', 'strategy']


class OneVsRestClassifier(MulticlassClassifier):
    """
    Wrapper around Sckit-learn's class with the same name. The input can be
    any kind of cuML compatible array, and the output type follows cuML's
    output type configuration rules.

    Berofe passing the data to scikit-learn, it is converted to host (numpy)
    array. Under the hood the data is partitioned for binary classification,
    and it is transformed back to the device by the cuML estimator. These
    copies back and forth the device and the host have some overhead. For more
    details see issue https://github.com/rapidsai/cuml/issues/2876.

    For documentation see `scikit-learn's OneVsRestClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html>`_.

    Examples
    --------

    >>> from cuml.linear_model import LogisticRegression
    >>> from cuml.multiclass import OneVsRestClassifier
    >>> from cuml.datasets.classification import make_classification
    >>>
    >>> X, y = make_classification(n_samples=10, n_features=6, n_informative=4,
    ...                            n_classes=3, random_state=137)
    >>>
    >>> cls = OneVsRestClassifier(LogisticRegression())
    >>> cls.fit(X,y)
    >>> cls.predict(X)
    array([1, 1, 1, 0, 1, 2, 2, 2, 0, 1])


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
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.
    """
    def __init__(self,
                 estimator,
                 *args,
                 handle=None,
                 verbose=False,
                 output_type=None):
        super().__init__(
            estimator, *args, handle=handle, verbose=verbose,
            output_type=output_type, strategy='ovr')

    def get_param_names(self):
        param_names = super().get_param_names()
        param_names.remove('strategy')
        return param_names


class OneVsOneClassifier(MulticlassClassifier):
    """
    Wrapper around Sckit-learn's class with the same name. The input can be
    any kind of cuML compatible array, and the output type follows cuML's
    output type configuration rules.

    Berofe passing the data to scikit-learn, it is converted to host (numpy)
    array. Under the hood the data is partitioned for binary classification,
    and it is transformed back to the device by the cuML estimator. These
    copies back and forth the device and the host have some overhead. For more
    details see issue https://github.com/rapidsai/cuml/issues/2876.

    For documentation see `scikit-learn's OneVsOneClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html>`_.

    Examples
    --------

    >>> from cuml.linear_model import LogisticRegression
    >>> from cuml.multiclass import OneVsOneClassifier
    >>> from cuml.datasets.classification import make_classification
    >>>
    >>> X, y = make_classification(n_samples=10, n_features=6, n_informative=4,
    ...                            n_classes=3, random_state=137)
    >>>
    >>> cls = OneVsOneClassifier(LogisticRegression())
    >>> cls.fit(X,y)
    >>> cls.predict(X)
    array([1, 1, 1, 0, 0, 2, 2, 2, 0, 1])

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
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.
    """
    def __init__(self,
                 estimator,
                 *args,
                 handle=None,
                 verbose=False,
                 output_type=None):
        super().__init__(
            estimator, *args, handle=handle, verbose=verbose,
            output_type=output_type, strategy='ovo')

    def get_param_names(self):
        param_names = super().get_param_names()
        param_names.remove('strategy')
        return param_names
