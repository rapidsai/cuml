#
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import inspect

from copy import deepcopy
from cuml.common.doc_utils import generate_docstring
from cuml.internals.api_decorators import api_base_return_any_skipall
from cuml.internals.base_helpers import _tags_class_and_instance
from cuml.internals.api_decorators import enable_device_interop


###############################################################################
#                          Tag Functionality Mixin                            #
###############################################################################


# Default tags for estimators inheritting from Base.
# tag system based on experimental tag system from Scikit-learn >=0.21
# https://scikit-learn.org/stable/developers/develop.html#estimator-tags
_default_tags = {
    # cuML specific tags
    "preferred_input_order": None,
    "X_types_gpu": ["2darray"],
    # Scikit-learn API standard tags
    "allow_nan": False,
    "binary_only": False,
    "multilabel": False,
    "multioutput": False,
    "multioutput_only": False,
    "no_validation": False,
    "non_deterministic": False,
    "pairwise": False,
    "poor_score": False,
    "preserves_dtype": [],
    "requires_fit": True,
    "requires_positive_X": False,
    "requires_positive_y": False,
    "requires_y": False,
    "stateless": False,
    "X_types": ["2darray"],
    "_skip_test": False,
    "_xfail_checks": False,
}


class TagsMixin:
    @_tags_class_and_instance
    def _get_tags(cls):
        """
        Method that collects all the static tags associated to any
        inheritting class. The Base class for cuML's estimators already
        uses this mixin, so most estimators don't need to use this Mixin
        directly.

        - Tags usage:

        In general, inheriting classes can use the appropriate Mixins defined
        in this file. Additional static tags can be defined by the
        `_get_static_tags` method like:

        ```
        @staticmethod
        def _more_static_tags():
           return {
                "requires_y": True
           }
        ```

        The method traverses the MRO in reverse
        order, i.e. the closer the parent to the final class will be
        explored later, so that children classes can overwrite their
        parent tags.

        - Mixin Usage

        If your class is not inheritting from cuml's Base
        then your class can use composition from this Mixin to get the tags
        behavior. If you want your class to have default tags different than
        the ones defined in this file, then implement the `_default_tags`
        method that returns a dictionary, like:

        class BaseClassWithTags(TagMixin)
            @staticmethod
            def _default_tags():
                return {'tag1': True, 'tag2': False}

        Method and code based on scikit-learn 0.21 _get_tags functionality:
        https://scikit-learn.org/stable/developers/develop.html#estimator-tags

        Examples
        --------

        >>> import cuml
        >>>
        >>> cuml.DBSCAN._get_tags()
        {'preferred_input_order': 'C', 'X_types_gpu': ['2darray'],
        'non_deterministic': False, 'requires_positive_X': False,
        'requires_positive_y': False, 'X_types': ['2darray'],
        'poor_score': False, 'no_validation': False, 'multioutput': False,
        'allow_nan': False, 'stateless': False, 'multilabel': False,
        '_skip_test': False, '_xfail_checks': False, 'multioutput_only': False,
        'binary_only': False, 'requires_fit': True, 'requires_y': False,
        'pairwise': False}

        """
        if hasattr(cls, "_default_tags"):
            tags = cls._default_tags()
        else:
            tags = deepcopy(_default_tags)
        for cl in reversed(inspect.getmro(cls)):
            if hasattr(cl, "_more_static_tags"):
                more_tags = cl._more_static_tags()
                tags.update(more_tags)

        return tags

    @_get_tags.instance_method
    def _get_tags(self):
        """
        Method to add dynamic tags capability to objects. Useful for cases
        where a tag depends on a value of an instantiated object. Dynamic tags
        will override class static tags, and can be defined with the
        _more_tags method in inheritting classes like:

        def _more_tags(self):
            return {'no_validation': not self.validate}

        Follows the same logic regarding the MRO as the static _get_tags.
        First it collects all the static tags of the reversed MRO, and then
        collects the dynamic tags and overwrites the corresponding static
        ones.

        Examples
        --------

        >>> import cuml
        >>>
        >>> estimator = cuml.DBSCAN()
        >>> estimator._get_tags()
        {'preferred_input_order': 'C', 'X_types_gpu': ['2darray'],
        'non_deterministic': False, 'requires_positive_X': False,
        'requires_positive_y': False, 'X_types': ['2darray'],
        'poor_score': False, 'no_validation': False, 'multioutput': False,
        'allow_nan': False, 'stateless': False, 'multilabel': False,
        '_skip_test': False, '_xfail_checks': False, 'multioutput_only': False,
        'binary_only': False, 'requires_fit': True, 'requires_y': False,
        'pairwise': False}

        """
        if hasattr(self, "_default_tags"):
            tags = self._default_tags()
        else:
            tags = deepcopy(_default_tags)
        dynamic_tags = {}
        for cl in reversed(inspect.getmro(self.__class__)):
            if hasattr(cl, "_more_static_tags"):
                more_tags = cl._more_static_tags()
                tags.update(more_tags)
            if hasattr(cl, "_more_tags"):
                more_tags = cl._more_tags(self)
                dynamic_tags.update(more_tags)
        tags.update(dynamic_tags)

        return tags


###############################################################################
#                          Estimator Type Mixins                              #
#                 Estimators should only use one of these.                    #
###############################################################################


class RegressorMixin:
    """
    Mixin class for regression estimators in cuML
    """

    _estimator_type = "regressor"

    @generate_docstring(
        return_values={
            "name": "score",
            "type": "float",
            "description": "R^2 of self.predict(X) " "wrt. y.",
        }
    )
    @api_base_return_any_skipall
    @enable_device_interop
    def score(self, X, y, sample_weight=None, **kwargs):
        """
        Scoring function for regression estimators

        Returns the coefficient of determination R^2 of the prediction.

        """
        from cuml.metrics.regression import r2_score

        preds = self.predict(X, **kwargs)
        return r2_score(y, preds, sample_weight=sample_weight)

    @staticmethod
    def _more_static_tags():
        return {"requires_y": True}


class ClassifierMixin:
    """
    Mixin class for classifier estimators in cuML
    """

    _estimator_type = "classifier"

    @generate_docstring(
        return_values={
            "name": "score",
            "type": "float",
            "description": (
                "Accuracy of self.predict(X) wrt. y "
                "(fraction where y == pred_y)"
            ),
        }
    )
    @api_base_return_any_skipall
    @enable_device_interop
    def score(self, X, y, sample_weight=None, **kwargs):
        """
        Scoring function for classifier estimators based on mean accuracy.

        """
        from cuml.metrics import accuracy_score

        preds = self.predict(X, **kwargs)
        return accuracy_score(y, preds, sample_weight=sample_weight)

    @staticmethod
    def _more_static_tags():
        return {"requires_y": True}


class ClusterMixin:
    """
    Mixin class for clustering estimators in cuML.
    """

    _estimator_type = "clusterer"

    @staticmethod
    def _more_static_tags():
        return {"requires_y": False}


###############################################################################
#                              Input Mixins                                   #
#               Estimators can use as many of these as needed.                #
###############################################################################


class FMajorInputTagMixin:
    """
    Mixin class for estimators that prefer inputs in F (column major) order.
    """

    @staticmethod
    def _more_static_tags():
        return {"preferred_input_order": "F"}


class CMajorInputTagMixin:
    """
    Mixin class for estimators that prefer inputs in C (row major) order.
    """

    @staticmethod
    def _more_static_tags():
        return {"preferred_input_order": "C"}


class SparseInputTagMixin:
    """
    Mixin class for estimators that can take (GPU and host) sparse inputs.
    """

    @staticmethod
    def _more_static_tags():
        return {
            "X_types_gpu": ["2darray", "sparse"],
            "X_types": ["2darray", "sparse"],
        }


class StringInputTagMixin:
    """
    Mixin class for estimators that can take (GPU and host) string inputs.
    """

    @staticmethod
    def _more_static_tags():
        return {
            "X_types_gpu": ["2darray", "string"],
            "X_types": ["2darray", "string"],
        }


class AllowNaNTagMixin:
    """
    Mixin class for estimators that allow NaNs in their inputs.
    """

    @staticmethod
    def _more_static_tags():
        return {"allow_nan": True}


###############################################################################
#                              Other Mixins                                   #
###############################################################################


class StatelessTagMixin:
    """
    Mixin class for estimators that are stateless.
    """

    @staticmethod
    def _more_static_tags():
        return {"stateless": True}
