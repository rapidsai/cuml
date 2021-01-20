#
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

from cuml.common.doc_utils import generate_docstring


class RegressorMixin:
    """Mixin class for regression estimators in cuML"""

    _estimator_type = "regressor"

    @generate_docstring(
        return_values={
            'name': 'score',
            'type': 'float',
            'description': 'R^2 of self.predict(X) '
                           'wrt. y.'
        })
    @cuml.internals.api_base_return_any_skipall
    def score(self, X, y, **kwargs):
        """
        Scoring function for regression estimators

        Returns the coefficient of determination R^2 of the prediction.

        """
        from cuml.metrics.regression import r2_score

        if hasattr(self, 'handle'):
            handle = self.handle
        else:
            handle = None

        preds = self.predict(X, **kwargs)
        return r2_score(y, preds, handle=handle)

    @staticmethod
    def _more_static_tags():
        return {
            'requires_y': True
        }


class ClassifierMixin:
    """Mixin class for classifier estimators in cuML"""

    _estimator_type = "classifier"

    @generate_docstring(
        return_values={
            'name':
                'score',
            'type':
                'float',
            'description': ('Accuracy of self.predict(X) wrt. y '
                            '(fraction where y == pred_y)')
        })
    @cuml.internals.api_base_return_any_skipall
    def score(self, X, y, **kwargs):
        """
        Scoring function for classifier estimators based on mean accuracy.

        """
        from cuml.metrics.accuracy import accuracy_score

        if hasattr(self, 'handle'):
            handle = self.handle
        else:
            handle = None

        preds = self.predict(X, **kwargs)
        return accuracy_score(y, preds, handle=handle)

    @staticmethod
    def _more_static_tags():
        return {
            'requires_y': True
        }


class FMajorInputTagMixin:

    @staticmethod
    def _more_static_tags():
        return {
            'preferred_input_order': 'F'
        }


class CMajorInputTagMixin:

    @staticmethod
    def _more_static_tags():
        return {
            'preferred_input_order': 'C'
        }


class SparseInputTagMixin:

    @staticmethod
    def _more_static_tags():
        return {
            'X_types_gpu': ['2darray', 'sparse'],
            'X_types': ['2darray', 'sparse']
        }
