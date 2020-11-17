# Copyright (c) 2020, NVIDIA CORPORATION.
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
from cuml.common.base import Base, ClassifierMixin
from cuml.common import input_to_host_array


class MulticlassClassifier(Base, ClassifierMixin):
    """ Wrapper around scikit-learn multiclass classifiers that allows to
    choose different multiclass strategies.

    See issue https://github.com/rapidsai/cuml/issues/2876 for more info about
    using Sklearn meta estimators.
    """
    def __init__(self, estimator, handle=None, verbose=False,
                 output_type=None, n_jobs=None, strategy='ovr'):
        super().__init__(handle=handle, verbose=verbose,
                         output_type=output_type)
        self.strategy = strategy
        self.estimator = estimator
        self.n_jobs = n_jobs

    @property
    @cuml.internals.api_base_return_array_skipall
    def classes_(self):
        return self.multiclass_estimator.classes_

    @property
    @cuml.internals.api_base_return_any_skipall
    def n_classes_(self):
        return self.multiclass_estimator.n_classes_

    def fit(self, X, y) -> 'MulticlassClassifier':
        if self.strategy == 'ovr':
            self.multiclass_estimator = sklearn.multiclass.\
                OneVsRestClassifier(self.estimator, n_jobs=self.n_jobs)
        elif self.strategy == 'ovo':
            self.multiclass_estimator = \
                sklearn.multiclass.OneVsOneClassifier(
                    self.estimator, n_jobs=self.n_jobs)
        else:
            raise ValueError('Invalid multiclass strategy ' +
                             str(self.strategy) + ', must be one of '
                             '{"ovr", "ovo"}')
        X, _, _, _, _ = input_to_host_array(X)
        y, _, _, _, _ = input_to_host_array(y)
        with cuml.internals.exit_internal_api():
            return self.multiclass_estimator.fit(X, y)

    def predict(self, X) -> CumlArray:
        X, _, _, _, _ = input_to_host_array(X)
        with cuml.internals.exit_internal_api():
            preds = self.multiclass_estimator.predict(X)
        return preds

    def decision_function(self, X) -> CumlArray:
        X, _, _, _, _ = input_to_host_array(X)
        with cuml.internals.exit_internal_api():
            df = self.multiclass_estimator.decision_function(X)
        return df

    def get_param_names(self):
        return super().get_param_names() + ['estimator', 'strategy', 'n_jobs']


class OneVsRestClassifier(MulticlassClassifier):
    """ Wrapper around Sckit-learn's class with the same name. This wrapper
    accepts any array type supported by cuML and converts them to numpy if
    needed to call the corresponding sklearn routine.

    For documentation see `scikit-learn's OneVsOneClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html>`_. # noqa: E501

    See issue https://github.com/rapidsai/cuml/issues/2876 for more info about
    using Sklearn meta estimators.
    """
    def __init__(self, estimator, *args, handle=None, verbose=False,
                 output_type=None, n_jobs=None):
        super(OneVsRestClassifier, self).__init__(
            estimator, *args, handle=handle, verbose=verbose,
            output_type=output_type, n_jobs=n_jobs, strategy='ovr')


class OneVsOneClassifier(MulticlassClassifier):
    """ Wrapper around Sckit-learn's class with the same name. This wrapper
    accepts any array type supported by cuML and converts them to numpy if
    needed to call the corresponding sklearn routine.

    For documentation see `scikit-learn's OneVsOneClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html>`_. # noqa: E501

    See issue https://github.com/rapidsai/cuml/issues/2876 for more info about
    using Sklearn meta estimators.
    """
    def __init__(self, estimator, *args, handle=None, verbose=False,
                 output_type=None, n_jobs=None):
        super(OneVsOneClassifier, self).__init__(
            estimator, *args, handle=handle, verbose=verbose,
            output_type=output_type, n_jobs=n_jobs, strategy='ovo')
