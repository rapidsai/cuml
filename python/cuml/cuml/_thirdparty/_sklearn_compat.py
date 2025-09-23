# Copyright (c) 2025, NVIDIA CORPORATION.  #
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
# Portions provided under the following terms:
# BSD 3-Clause License

# Copyright (c) 2024, Guillaume Lemaitre

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""This file was vendored from `sklearn-compat` version 0.1.3, and then culled to
only include the parts we needed."""

from __future__ import annotations

import sklearn
from sklearn.utils.fixes import parse_version

sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)


########################################################################################
# The following code does not depend on the sklearn version
########################################################################################

def get_tags(estimator):
    """Get estimator tags in a consistent format across different sklearn versions.

    This function provides compatibility between sklearn versions before and after 1.6.
    It returns either a Tags object (sklearn >= 1.6) or a converted Tags object from
    the dictionary format (sklearn < 1.6) containing metadata about the estimator's
    requirements and capabilities.

    Parameters
    ----------
    estimator : estimator object
        A scikit-learn estimator instance.

    Returns
    -------
    tags : Tags
        An object containing metadata about the estimator's requirements and
        capabilities (e.g., input types, fitting requirements, classifier/regressor
        specific tags).
    """
    try:
        from sklearn.utils._tags import get_tags

        return get_tags(estimator)
    except ImportError:
        from sklearn.utils._tags import _safe_tags

        return _to_new_tags(_safe_tags(estimator), estimator)


def _to_new_tags(old_tags, estimator=None):
    """Utility function convert old tags (dictionary) to new tags (dataclass)."""
    input_tags = InputTags(
        one_d_array="1darray" in old_tags["X_types"],
        two_d_array="2darray" in old_tags["X_types"],
        three_d_array="3darray" in old_tags["X_types"],
        sparse="sparse" in old_tags["X_types"],
        categorical="categorical" in old_tags["X_types"],
        string="string" in old_tags["X_types"],
        dict="dict" in old_tags["X_types"],
        positive_only=old_tags["requires_positive_X"],
        allow_nan=old_tags["allow_nan"],
        pairwise=old_tags["pairwise"],
    )
    target_tags = TargetTags(
        required=old_tags["requires_y"],
        one_d_labels="1dlabels" in old_tags["X_types"],
        two_d_labels="2dlabels" in old_tags["X_types"],
        positive_only=old_tags["requires_positive_y"],
        multi_output=old_tags["multioutput"] or old_tags["multioutput_only"],
        single_output=not old_tags["multioutput_only"],
    )
    if estimator is not None and (
        hasattr(estimator, "transform") or hasattr(estimator, "fit_transform")
    ):
        transformer_tags = TransformerTags(
            preserves_dtype=old_tags["preserves_dtype"],
        )
    else:
        transformer_tags = None
    estimator_type = getattr(estimator, "_estimator_type", None)
    if estimator_type == "classifier":
        classifier_tags = ClassifierTags(
            poor_score=old_tags["poor_score"],
            multi_class=not old_tags["binary_only"],
            multi_label=old_tags["multilabel"],
        )
    else:
        classifier_tags = None
    if estimator_type == "regressor":
        regressor_tags = RegressorTags(
            poor_score=old_tags["poor_score"],
        )
    else:
        regressor_tags = None
    return Tags(
        estimator_type=estimator_type,
        target_tags=target_tags,
        transformer_tags=transformer_tags,
        classifier_tags=classifier_tags,
        regressor_tags=regressor_tags,
        input_tags=input_tags,
        # Array-API was introduced in 1.3, we need to default to False if not inside
        # the old-tags.
        array_api_support=old_tags.get("array_api_support", False),
        no_validation=old_tags["no_validation"],
        non_deterministic=old_tags["non_deterministic"],
        requires_fit=old_tags["requires_fit"],
        _skip_test=old_tags["_skip_test"],
    )


if sklearn_version < parse_version("1.6"):
    import sys

    from dataclasses import dataclass, field

    def _dataclass_args():
        if sys.version_info < (3, 10):
            return {}
        return {"slots": True}

    # tags infrastructure
    @dataclass(**_dataclass_args())
    class InputTags:
        """Tags for the input data.

        Parameters
        ----------
        one_d_array : bool, default=False
            Whether the input can be a 1D array.

        two_d_array : bool, default=True
            Whether the input can be a 2D array. Note that most common
            tests currently run only if this flag is set to ``True``.

        three_d_array : bool, default=False
            Whether the input can be a 3D array.

        sparse : bool, default=False
            Whether the input can be a sparse matrix.

        categorical : bool, default=False
            Whether the input can be categorical.

        string : bool, default=False
            Whether the input can be an array-like of strings.

        dict : bool, default=False
            Whether the input can be a dictionary.

        positive_only : bool, default=False
            Whether the estimator requires positive X.

        allow_nan : bool, default=False
            Whether the estimator supports data with missing values encoded as `np.nan`.

        pairwise : bool, default=False
            This boolean attribute indicates whether the data (`X`),
            :term:`fit` and similar methods consists of pairwise measures
            over samples rather than a feature representation for each
            sample.  It is usually `True` where an estimator has a
            `metric` or `affinity` or `kernel` parameter with value
            'precomputed'. Its primary purpose is to support a
            :term:`meta-estimator` or a cross validation procedure that
            extracts a sub-sample of data intended for a pairwise
            estimator, where the data needs to be indexed on both axes.
            Specifically, this tag is used by
            `sklearn.utils.metaestimators._safe_split` to slice rows and
            columns.
        """

        one_d_array: bool = False
        two_d_array: bool = True
        three_d_array: bool = False
        sparse: bool = False
        categorical: bool = False
        string: bool = False
        dict: bool = False
        positive_only: bool = False
        allow_nan: bool = False
        pairwise: bool = False

    @dataclass(**_dataclass_args())
    class TargetTags:
        """Tags for the target data.

        Parameters
        ----------
        required : bool
            Whether the estimator requires y to be passed to `fit`,
            `fit_predict` or `fit_transform` methods. The tag is ``True``
            for estimators inheriting from `~sklearn.base.RegressorMixin`
            and `~sklearn.base.ClassifierMixin`.

        one_d_labels : bool, default=False
            Whether the input is a 1D labels (y).

        two_d_labels : bool, default=False
            Whether the input is a 2D labels (y).

        positive_only : bool, default=False
            Whether the estimator requires a positive y (only applicable
            for regression).

        multi_output : bool, default=False
            Whether a regressor supports multi-target outputs or a classifier supports
            multi-class multi-output.

        single_output : bool, default=True
            Whether the target can be single-output. This can be ``False`` if the
            estimator supports only multi-output cases.
        """

        required: bool
        one_d_labels: bool = False
        two_d_labels: bool = False
        positive_only: bool = False
        multi_output: bool = False
        single_output: bool = True

    @dataclass(**_dataclass_args())
    class TransformerTags:
        """Tags for the transformer.

        Parameters
        ----------
        preserves_dtype : list[str], default=["float64"]
            Applies only on transformers. It corresponds to the data types
            which will be preserved such that `X_trans.dtype` is the same
            as `X.dtype` after calling `transformer.transform(X)`. If this
            list is empty, then the transformer is not expected to
            preserve the data type. The first value in the list is
            considered as the default data type, corresponding to the data
            type of the output when the input data type is not going to be
            preserved.
        """

        preserves_dtype: list[str] = field(default_factory=lambda: ["float64"])

    @dataclass(**_dataclass_args())
    class ClassifierTags:
        """Tags for the classifier.

        Parameters
        ----------
        poor_score : bool, default=False
            Whether the estimator fails to provide a "reasonable" test-set
            score, which currently for classification is an accuracy of
            0.83 on ``make_blobs(n_samples=300, random_state=0)``. The
            datasets and values are based on current estimators in scikit-learn
            and might be replaced by something more systematic.

        multi_class : bool, default=True
            Whether the classifier can handle multi-class
            classification. Note that all classifiers support binary
            classification. Therefore this flag indicates whether the
            classifier is a binary-classifier-only or not.

        multi_label : bool, default=False
            Whether the classifier supports multi-label output.
        """

        poor_score: bool = False
        multi_class: bool = True
        multi_label: bool = False

    @dataclass(**_dataclass_args())
    class RegressorTags:
        """Tags for the regressor.

        Parameters
        ----------
        poor_score : bool, default=False
            Whether the estimator fails to provide a "reasonable" test-set
            score, which currently for regression is an R2 of 0.5 on
            ``make_regression(n_samples=200, n_features=10,
            n_informative=1, bias=5.0, noise=20, random_state=42)``. The
            dataset and values are based on current estimators in scikit-learn
            and might be replaced by something more systematic.
        """

        poor_score: bool = False

    @dataclass(**_dataclass_args())
    class Tags:
        """Tags for the estimator.

        See :ref:`estimator_tags` for more information.

        Parameters
        ----------
        estimator_type : str or None
            The type of the estimator. Can be one of:
            - "classifier"
            - "regressor"
            - "transformer"
            - "clusterer"
            - "outlier_detector"
            - "density_estimator"

        target_tags : :class:`TargetTags`
            The target(y) tags.

        transformer_tags : :class:`TransformerTags` or None
            The transformer tags.

        classifier_tags : :class:`ClassifierTags` or None
            The classifier tags.

        regressor_tags : :class:`RegressorTags` or None
            The regressor tags.

        array_api_support : bool, default=False
            Whether the estimator supports Array API compatible inputs.

        no_validation : bool, default=False
            Whether the estimator skips input-validation. This is only meant for
            stateless and dummy transformers!

        non_deterministic : bool, default=False
            Whether the estimator is not deterministic given a fixed ``random_state``.

        requires_fit : bool, default=True
            Whether the estimator requires to be fitted before calling one of
            `transform`, `predict`, `predict_proba`, or `decision_function`.

        _skip_test : bool, default=False
            Whether to skip common tests entirely. Don't use this unless
            you have a *very good* reason.

        input_tags : :class:`InputTags`
            The input data(X) tags.
        """

        estimator_type: str | None
        target_tags: TargetTags
        transformer_tags: TransformerTags | None = None
        classifier_tags: ClassifierTags | None = None
        regressor_tags: RegressorTags | None = None
        array_api_support: bool = False
        no_validation: bool = False
        non_deterministic: bool = False
        requires_fit: bool = True
        _skip_test: bool = False
        input_tags: InputTags = field(default_factory=InputTags)

else:
    # test_common
    # tags infrastructure
    from sklearn.utils import (
        ClassifierTags,
        InputTags,
        RegressorTags,
        Tags,
        TargetTags,
        TransformerTags,
    )
