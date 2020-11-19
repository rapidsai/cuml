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

import cupy as cp
import numpy as np
import cuml.internals
from cuml.common.input_utils import input_to_cupy_array


@cuml.internals.api_return_any()
def log_loss(y_true,
             y_pred,
             eps=1e-15,
             normalize=True,
             sample_weight=None) -> float:
    """ Log loss, aka logistic loss or cross-entropy loss.
    This is the loss function used in (multinomial) logistic regression
    and extensions of it such as neural networks, defined as the negative
    log-likelihood of a logistic model that returns ``y_pred`` probabilities
    for its training data ``y_true``.
    The log loss is only defined for two or more labels.

    Parameters
    ----------
    y_true : array-like, shape = (n_samples,)
    y_pred : array-like of float,
        shape = (n_samples, n_classes) or (n_samples,)
    eps : float
        Log loss is undefined for p=0 or p=1, so probabilities are
        clipped to max(eps, min(1 - eps, p)).
    normalize : bool, optional (default=True)
        If true, return the mean loss per sample.
        Otherwise, return the sum of the per-sample losses.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    loss : float

    Examples
    --------
    >>> from cuml.metrics import log_loss
    >>> import numpy as np
    >>> log_loss(np.array([1, 0, 0, 1]),
    ...          np.array([[.1, .9], [.9, .1], [.8, .2], [.35, .65]]))
    0.21616...

    References
    ----------
    C.M. Bishop (2006). Pattern Recognition and Machine Learning. Springer,
    p. 209.

    Notes
    -----
    The logarithm used is the natural logarithm (base-e).

    """
    y_true, n_rows, n_cols, ytype = \
        input_to_cupy_array(y_true, check_dtype=[np.int32, np.int64,
                                                 np.float32, np.float64])

    if y_true.dtype.kind == 'f' and np.any(y_true != y_true.astype(int)):
        raise ValueError("'y_true' can only have integer values")
    if y_true.min() < 0:
        raise ValueError("'y_true' cannot have negative values")

    y_pred, _, _, _ = \
        input_to_cupy_array(y_pred, check_dtype=[np.int32, np.int64,
                                                 np.float32, np.float64],
                            check_rows=n_rows)

    y_true_max = y_true.max()
    if (y_pred.ndim == 1 and y_true_max > 1) \
       or (y_pred.ndim > 1 and y_pred.shape[1] <= y_true_max):
        raise ValueError("The shape of y_pred doesn't "
                         "match the number of classes")

    y_true = y_true.astype('int32')
    y_pred = cp.clip(y_pred, eps, 1 - eps)
    if y_pred.ndim == 1:
        y_pred = cp.expand_dims(y_pred, axis=1)
    if y_pred.shape[1] == 1:
        y_pred = cp.hstack([1 - y_pred, y_pred])

    y_pred /= cp.sum(y_pred, axis=1, keepdims=True)
    loss = -cp.log(y_pred)[cp.arange(y_pred.shape[0]), y_true]
    return _weighted_sum(loss, sample_weight, normalize).item()


def _weighted_sum(sample_score, sample_weight, normalize):
    if normalize:
        return cp.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return cp.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()
