# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Raghav RV <rvraghav93@gmail.com>
#         Michal Karbownik <michakarbownik@gmail.com>
# License: BSD 3 clause
import numbers
import warnings
from sklearn.base import clone
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

import cupy as cp
import numpy as np
from cuml.internals.base import Base, is_classifier
from cuml.preprocessing.LabelEncoder import LabelEncoder
from cupyx.scipy import sparse as cusp

from ..utils import _safe_indexing


def _clf_dft_values(dtype=None):
    dtype = np.float64 if dtype is None else dtype
    return {
        "decision_function": np.finfo(dtype).min,
        "predict_proba": 0,
        "predict_log_proba": np.finfo(dtype).min,
    }


def _need_encode(method: str, y: Optional[Any]) -> bool:
    return method in _clf_dft_values() and y is not None


def _split(X, y, indices):
    X_train = _safe_indexing(X, indices)
    if y is not None:
        y_train = _safe_indexing(y, indices)
    else:
        y_train = None
    return X_train, y_train


def _enforce_prediction_order(
    fold_classes: cp.ndarray,
    predictions: cp.ndarray,
    n_total_classes: int,
    method: str,
) -> cp.ndarray:
    """Handle a special case where some classes are missing in a CV fold."""
    if n_total_classes == len(fold_classes):
        # The fold contains all the classes
        return predictions

    msg = (
        "Some classes are missing the training fold, which might lead to incorrect or"
        " sub-optimal results."
    )
    warnings.warn(msg, UserWarning)

    # Fill in the default values for missing classes
    full = cp.full(
        (predictions.shape[0], n_total_classes),
        _clf_dft_values(predictions.dtype)[method],
        dtype=predictions.dtype,
    )
    # Fill in the values for classes that are covered by the current fold
    full[:, fold_classes] = predictions
    return full


def _encode_y(y, output_type):
    y = cp.asarray(y)
    if y.ndim == 1:
        return LabelEncoder(verbose=0, output_type=output_type).fit_transform(y)
    elif y.ndim == 2:
        y_enc = cp.zeros_like(y, dtype=cp.int64)
        for y_i in range(y.shape[1]):
            y_enc[:, y_i] = LabelEncoder().fit_transform(y[:, y_i])
        return y_enc
    else:
        raise ValueError(f"Invalid shape for y: {y.shape}")


def check_cv(cv=5, classifier=False):
    from cuml.model_selection import KFold, StratifiedKFold
    from cuml.model_selection._split import KFoldBase

    cv = 5 if cv is None else cv
    if isinstance(cv, numbers.Integral):
        if classifier:
            # sklearn checks y as well before making the decision, here we skip the data
            # lengthy inspection for now since we know the estimator type with
            # cv_predict.
            return StratifiedKFold(cv)
        else:
            return KFold(cv)
    elif hasattr(cv, "split") or isinstance(cv, KFoldBase):
        return cv
    else:
        raise TypeError(f"Unsupported type for cv: {type(cv)}")


def cross_val_predict(
    estimator: Base,
    X,
    y=None,
    *,
    sample_weight=None,
    cv=None,
    verbose=0,
    fit_params=None,
    method="predict",
) -> cp.ndarray:
    fit_params = {} if fit_params is None else fit_params

    cv = check_cv(cv, is_classifier(estimator))
    splits = list(cv.split(X, y))

    # Get a contiguous test data set for prediction.
    test_indices = cp.concatenate([test for _, test in splits])

    need_encode = _need_encode(method, y)
    if need_encode:
        y = _encode_y(y, estimator.output_type)

    predictions = []

    for train, test in splits:
        est = clone(estimator)

        X_train, y_train = _split(X, y, train)
        X_test, _ = _split(X, y, test)
        if sample_weight is not None:
            w_train = _safe_indexing(sample_weight, train)
        else:
            w_train = None

        # Train an estimator
        if y_train is None:
            if w_train is not None:
                est.fit(X_train, sample_weight=w_train, **fit_params)
            else:
                est.fit(X_train, **fit_params)
        else:
            if w_train is not None:
                est.fit(X_train, y_train, sample_weight=w_train, **fit_params)
            else:
                est.fit(X_train, y_train, **fit_params)

        # Obtain its prediction on the test dataset
        predict_fn = getattr(est, method)
        predt = cp.asarray(predict_fn(X_test))

        if not need_encode:
            predictions.append(predt)
            continue

        # Get the classes contained in the current fold. Need to fill in the values for
        # prediction output in case if there are missing classes, which can happen if
        # the CV is not stratified.
        fold_classes = est.classes_
        if isinstance(predt, list):
            # multi-class-multi-target
            assert len(predt) == len(fold_classes)
            predt = [
                _enforce_prediction_order(
                    fold_classes=fold_classes[y_i],
                    predictions=predt,
                    n_total_classes=len(cp.unique(y[:, y_i])),
                    method=method,
                )
                for y_i in range(len(predt))
            ]
        else:
            predt = _enforce_prediction_order(
                fold_classes=fold_classes,
                predictions=predt,
                n_total_classes=(
                    len(cp.unique(y)) if y.ndim == 1 else y.shape[1]
                ),
                method=method,
            )
        predictions.append(predt)

    inv_test_indices = cp.empty(len(test_indices), dtype=cp.int64)
    inv_test_indices[test_indices] = cp.arange(len(test_indices))

    if cusp.issparse(predictions[0]):
        predictions = cusp.vstack(predictions, format=predictions[0].format)
    elif need_encode and isinstance(predictions[0], list):
        n_labels = y.shape[1]
        concat_pred = []
        for y_i in range(n_labels):
            label_preds = cp.concatenate([p[y_i] for p in predictions])
            concat_pred.append(label_preds)
        predictions = concat_pred
    else:
        predictions = cp.concatenate(predictions)

    if isinstance(predictions, list):
        return [p[inv_test_indices] for p in predictions]
    else:
        return predictions[inv_test_indices]
