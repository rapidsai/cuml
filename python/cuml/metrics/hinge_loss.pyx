#
# Copyright (c) 2021, NVIDIA CORPORATION.
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
from cuml.preprocessing import LabelEncoder, LabelBinarizer
import cudf


def cython_hinge_loss(y_true, pred_decision, labels=None):
    y_true_unique = cp.unique(labels.values if labels is not None else y_true)
    
    if y_true_unique.size > 2:
        if (labels is None and pred_decision.ndim > 1 and
                (cp.size(y_true_unique) != pred_decision.shape[1])):
            raise ValueError("Please include all labels in y_true "
                             "or pass labels as third argument")
        if labels is None:
            labels = y_true_unique
        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        mask = cp.ones_like(pred_decision, dtype=bool)
        mask[cp.arange(y_true.shape[0]), y_true.values] = False
        margin = pred_decision[~mask]
        margin -= cp.max(pred_decision[mask].reshape(y_true.shape[0], -1),
                         axis=1)
    else:
        # Handles binary class case
        # this code assumes that positive and negative labels
        # are encoded as +1 and -1 respectively
        pred_decision = cp.ravel(pred_decision)

        lbin = LabelBinarizer(neg_label=-1)
        y_true = lbin.fit_transform(y_true)[:, 0]

        try:
            margin = y_true * pred_decision
        except TypeError:
            raise TypeError("pred_decision should be an array of floats.")

    losses = 1 - margin
    # The hinge_loss doesn't penalize good enough predictions.
    cp.clip(losses, 0, None, out=losses)
    return cp.average(losses)