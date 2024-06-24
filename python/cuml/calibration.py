#
# Copyright (c) 2024, NVIDIA CORPORATION.
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
from copy import deepcopy
from functools import partial
from inspect import signature
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from cupyx.scipy.special import expit, xlogy
from sklearn.base import clone
from sklearn.calibration import MetaEstimatorMixin

from cuml.common.exceptions import NotFittedError
from cuml.internals.base import Base
from cuml.internals.mixins import ClassifierMixin, RegressorMixin
from cuml.internals.safe_imports import cpu_only_import, gpu_only_import
from cuml.model_selection import cross_val_predict
from cuml.preprocessing.label import label_binarize
from cuml.preprocessing.LabelEncoder import LabelEncoder
from cuml.solvers.lbfgs import _fmin_lbfgs

from ._thirdparty.sklearn.model_selection._validation import (
    _safe_indexing,
    _split,
    check_cv,
)

if TYPE_CHECKING:
    import cupy as cp
    import numpy as np
else:
    np = cpu_only_import("numpy")
    cp = gpu_only_import("cupy")


class _SigmoidModel:
    def __init__(self, slope, intercept) -> None:
        self.slope = slope
        self.intercept = intercept

    def predict(self, X):
        return expit(-(self.slope * X + self.intercept))


def _estimate_sigmoid(predictions, y, sample_weight) -> _SigmoidModel:
    """Implements [1] for calibrating classification models.

    .. math::

        p(y = 1 | f) = \\frac{1}{1 + \\exp{a f + b}}

    where :math:`f` is the model output (predictions).

    Returns
    -------
    The estimated slope and intercept: :math:`a` and :math:`b`.

    References
    ----------
    .. [1] Platt, "Probabilistic Outputs for Support Vector Machines and Comparisons to
           Regularized Likelihood Methods." 1999
    """
    if not isinstance(predictions.dtype, cp.floating):
        predictions = predictions.astype(cp.float64)

    neg_mask = y <= 0
    if sample_weight is None:
        sample_weight = cp.ones_like(predictions, dtype=predictions.dtype)

    n_negative = sample_weight[neg_mask].sum()
    n_positive = sample_weight[~neg_mask].sum()

    # Define soft label, cannot use standard logistic regression in cuml.
    target = cp.zeros_like(y, dtype=predictions.dtype)
    target[y > 0] = (n_positive + 1.0) / (n_positive + 2.0)  # Eq (13)
    target[y <= 0] = 1.0 / (n_negative + 2.0)  # Eq (14)
    # cache to avoid repeated calculation.
    m_target = 1.0 - target

    def objective(x: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
        # loss
        p = expit(-(x[0] * predictions + x[1]))
        loss = -(xlogy(target, p) + xlogy(m_target, 1.0 - p))
        loss = loss @ sample_weight
        assert loss.size == 1
        # grad
        t_p = (target - p) * sample_weight
        da = t_p @ predictions
        da = cp.dot(t_p, predictions)
        db = t_p.sum()
        return loss, cp.array([da, db])

    a0 = 0.0
    b0 = float(cp.log((n_negative + 1.0) / (n_positive + 1.0)))
    x0 = cp.array([a0, b0], dtype=predictions.dtype)
    res = _fmin_lbfgs(objective, x0)
    return _SigmoidModel(res.x[0], res.x[1])


def _get_predict_fn(estimator) -> Tuple[Callable, str]:
    for name in ("decision_function", "predict_proba"):
        if hasattr(estimator, name):
            predict_fn = getattr(estimator, name)
            predict_name = name
            return predict_fn, predict_name
    raise ValueError("Invalid estimator.")


class _Calibrator:
    """A calibrator the the input classifier. Internally, it estimates one
    sub-calibrator for each class when multi-class classification is used.

    """

    def __init__(
        self,
        estimator,
        classes,
        output_type,
        predict_fn: Optional[Callable] = None,
    ) -> None:
        self.estimator = estimator
        self.classes = classes
        self.calibrators: List[_SigmoidModel] = []
        self.output_type = output_type

        self.predict_fn = predict_fn

        fn, self.predict_name = _get_predict_fn(self.estimator)
        if self.predict_fn is None:
            self.predict_fn = fn

        if self.predict_fn is None:
            raise TypeError(
                "Invalid estimator type, either predict_proba or decision_function is required."
            )

    @property
    def classes_(self):
        return cp.asarray(self.classes)

    def _predict(self, X):
        classes = self.classes_
        n_classes = len(classes)
        predictions = cp.asarray(self.predict_fn(X=X))

        if self.predict_name == "decision_function":
            if predictions.ndim == 1:
                predictions = predictions[:, cp.newaxis]
        elif self.predict_name == "predict_proba":
            # indexing trick, [:, 1:] returns a vector with shape (n, 1), while [:, 1]
            # returns a vector with shape (n, )
            if n_classes == 2:
                predictions = predictions[:, 1:]
                assert predictions.ndim == 2
        else:
            raise ValueError("Invalid prediction name.")
        if not isinstance(predictions.dtype, cp.floating):
            predictions = predictions.astype(cp.float64)
        return predictions

    def fit(self, X, y, sample_weight=None) -> "_Calibrator":
        """Fit a calibrator based on an existing estimator."""
        classes = self.classes_
        predictions = self._predict(X)

        Y = cp.asarray(label_binarize(y, classes=classes.astype(y.dtype)))
        if len(classes) == 2:
            # Unlike scikit-learn, the binarizer in cuML returns full matrix for binary
            # classes.
            # https://github.com/rapidsai/cuml/issues/5740
            assert Y.ndim == 2 and Y.shape[1] == 2
            Y = Y[:, 1].reshape(Y.shape[0], 1)

        label_encoder = LabelEncoder(output_type=self.output_type).fit(classes)
        # cuml label encoder returns a cuDF dataframe
        pos_class_indices = label_encoder.transform(classes)
        assert pos_class_indices.ndim == 1 or pos_class_indices.shape[1] == 1
        if hasattr(pos_class_indices, "iloc"):
            pos_class_indices = pos_class_indices.values

        for class_idx, this_pred in zip(pos_class_indices, predictions.T):
            calibrator = _estimate_sigmoid(
                this_pred, Y[:, class_idx], sample_weight
            )
            self.calibrators.append(calibrator)
        return self

    def predict_proba(self, X) -> cp.ndarray:
        if not self.calibrators:
            raise NotFittedError()

        p = self._predict(X)
        classes = self.classes_
        pos_class_indices = (
            LabelEncoder(output_type=self.output_type)
            .fit(self.classes_)
            .transform(cp.asarray(self.estimator.classes_))
        )
        if hasattr(pos_class_indices, "iloc"):
            pos_class_indices = pos_class_indices.values

        n_classes = len(classes)
        proba = cp.zeros((X.shape[0], n_classes), dtype=p.dtype)

        for i in range(p.shape[1]):
            idx = pos_class_indices[i]
            if n_classes == 2:
                idx += 1
            proba[:, idx] = self.calibrators[i].predict(p[:, i])

        if n_classes == 2:
            proba[:, 0] = 1.0 - proba[:, 1]
        else:
            denominator = cp.sum(proba, axis=1)[:, np.newaxis]
            # In the edge case where for each class calibrator returns a null
            # probability for a given sample, use the uniform distribution
            # instead.
            uniform_proba = cp.full_like(proba, 1.0 / n_classes)
            mask = denominator != 0
            proba[mask.flatten(), :] = proba[mask.flatten(), :] / denominator

        proba = cp.clip(proba, 0.0, 1.0)
        return proba


def _check_support_weight(estimator):
    # Similar to what skl does, we skip sample weight for CV when the underlying
    # estimator doesn't support it.
    fit_parameters = signature(estimator.fit).parameters
    supports_sw = "sample_weight" in fit_parameters
    return supports_sw

# In the spirit of sklearn cv function naming.
def _cross_val_calibrate(
    estimator,
    X,
    y,
    cv,
    fit_params,
    sample_weight,
    classes,
    output_type,
):
    cv = check_cv(cv)
    calibrators = []

    for train, test in cv.split(X, y):
        est = clone(estimator)

        X_train, y_train = _split(X, y, train)
        X_test, y_test = _split(X, y, test)
        if sample_weight is not None:
            w_train = _safe_indexing(sample_weight, train)
        else:
            w_train = None

        if _check_support_weight(estimator):
            est.fit(X_train, y_train, sample_weight=w_train, **fit_params)
        else:
            est.fit(X_train, y_train, **fit_params)

        calibrator = _Calibrator(est, classes=classes, output_type=output_type)

        if sample_weight is not None:
            w_test = _safe_indexing(sample_weight, test)
        else:
            w_test = None
        calibrator.fit(X_test, y_test, sample_weight=w_test)
        # The calibrator instance itself can have multiple calibrators, one for each
        # class. Then we have one such calibrator for each CV fold.
        calibrators.append(calibrator)

    return calibrators


def cuml_clone(estimator, output_type):
    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in new_object_params.items():
        new_object_params[name] = clone(param, safe=False)
    output_type = estimator.output_type
    del new_object_params["output_type"]
    new_object = klass(**new_object_params, output_type=output_type)
    params_set = new_object.get_params(deep=False)

    # quick sanity check of the parameters of the clone
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is not param2:
            raise RuntimeError(
                "Cannot clone object %s, as the constructor "
                "either does not set or modifies parameter %s"
                % (estimator, name)
            )

    # _sklearn_output_config is used by `set_output` to configure the output
    # container of an estimator.
    if hasattr(estimator, "_sklearn_output_config"):
        new_object._sklearn_output_config = copy.deepcopy(
            estimator._sklearn_output_config
        )
    return new_object


class CalibratedClassifierCV(ClassifierMixin, MetaEstimatorMixin, Base):
    """Probability calibration with logistic regression."""

    def __init__(
        self,
        *,
        estimator=None,
        method="sigmoid",
        cv=None,
        ensemble=True,
        output_type=None,
    ) -> None:
        super().__init__(output_type=output_type)

        self.estimator = estimator
        self.method = method
        self.cv = cv
        self.ensemble = ensemble

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "calibrated_classifiers_")

    def fit(
        self, X, y, sample_weight=None, **fit_params
    ) -> "CalibratedClassifierCV":
        estimator = self.estimator

        self.calibrated_classifiers_ = []

        if self.cv == "prefit":
            self.classes_ = cp.asarray(self.estimator.classes_)
            estimator = self.estimator
            predictions = estimator.predict(X)
            calibrator = _Calibrator(
                self.estimator, classes=self.classes_, output_type=self.output_type
            )
            calibrator.fit(X, y, sample_weight=sample_weight)
            self.calibrated_classifiers_.append(calibrator)
        else:
            # We obtain all the classes upfront in case of when a fold doesn't contain
            # all the classes.
            label_encoder_ = LabelEncoder(output_type=self.output_type).fit(y)
            classes = label_encoder_.classes_
            self.classes_ = classes.values if hasattr(classes, "iloc") else classes

            n_classes = len(self.classes_)
            cv = check_cv(self.cv, classifier=True)

            if self.ensemble:
                self.calibrated_classifiers_ = _cross_val_calibrate(
                    estimator,
                    X,
                    y,
                    self.cv,
                    fit_params,
                    sample_weight=sample_weight,
                    classes=self.classes_,
                    output_type=self.output_type,
                )
            else:
                estimator.set_params(output_type=self.output_type)
                est = cuml_clone(estimator, self.output_type)

                if sample_weight is not None and _check_support_weight(est):
                    est.fit(X, y, sample_weight=sample_weight)
                else:
                    est.fit(X, y)

                fit_params = {}

                _, predict_name = _get_predict_fn(estimator)
                predict_fn = partial(
                    cross_val_predict,
                    estimator=cuml_clone(estimator, self.output_type),
                    X=X,
                    y=y,
                    sample_weight=sample_weight if _check_support_weight(est) else None,
                    cv=cv,
                    method=predict_name,
                    fit_params=fit_params,
                )
                calibrator = _Calibrator(
                    est,
                    classes=self.classes_,
                    output_type=self.output_type,
                    predict_fn=predict_fn,
                )
                calibrator.fit(X, y, sample_weight=sample_weight)

                self.calibrated_classifiers_.append(calibrator)
        return self

    def predict_proba(self, X):
        # Use mean for CV ensemble.
        mean = cp.zeros((X.shape[0], len(self.classes_)))
        for c in self.calibrated_classifiers_:
            proba = c.predict_proba(X)
            mean += proba

        mean /= len(self.calibrated_classifiers_)

        return mean

    def predict(self, X):
        return self.classes_[cp.argmax(self.predict_proba(X), axis=1)]
