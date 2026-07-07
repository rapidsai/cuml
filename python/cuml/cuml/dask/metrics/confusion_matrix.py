#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import dask.array as da
import numpy as np

from cuml.dask.common.input_utils import DistributedDataHandler
from cuml.dask.common.utils import get_client
from cuml.metrics.confusion_matrix import confusion_matrix as _confusion_matrix


def _local_cm(inputs, labels, use_sample_weight):
    if use_sample_weight:
        y_true, y_pred, sample_weight = inputs
    else:
        y_true, y_pred = inputs
        sample_weight = cp.ones(y_true.shape[0], dtype=y_true.dtype)

    return _confusion_matrix(
        y_true, y_pred, labels=labels, sample_weight=sample_weight
    )


def confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    normalize=None,
    sample_weight=None,
    client=None,
):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Parameters
    ----------
    y_true : dask.Array (device or host) shape = (n_samples,)
        Ground truth (correct) target values.
    y_pred : dask.Array (device or host) shape = (n_samples,)
        Estimated target values.
    labels : array-like (device or host) shape = (n_classes,), optional
        List of labels to index the matrix. This may be used to reorder or
        select a subset of labels. If None is given, those that appear at least
        once in y_true or y_pred are used in sorted order.
    sample_weight : dask.Array (device or host) shape = (n_samples,), optional
        Sample weights.
    normalize : string in ['true', 'pred', 'all']
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.
    client : dask.distributed.Client, optional
        Dask client to use. Will use the default client if None.

    Returns
    -------
    C : array-like (device or host) shape = (n_classes, n_classes)
        Confusion matrix.
    """
    client = get_client(client)

    if labels is None:
        labels = da.unique(
            da.concatenate([da.unique(y_true), da.unique(y_pred)])
        ).compute()

    if normalize not in ["true", "pred", "all", None]:
        msg = (
            "normalize must be one of "
            f"{{'true', 'pred', 'all', None}}, got {normalize}."
        )
        raise ValueError(msg)

    use_sample_weight = bool(sample_weight is not None)
    dask_arrays = (
        [y_true, y_pred, sample_weight]
        if use_sample_weight
        else [y_true, y_pred]
    )

    # run cm computation on each partition.
    data = DistributedDataHandler.create(dask_arrays, client=client)
    cms = [
        client.submit(
            _local_cm, p, labels, use_sample_weight, workers=[w]
        ).result()
        for w, p in data.gpu_futures
    ]

    # reduce each partition's result into one cupy matrix
    cm = sum(cms)

    with np.errstate(all="ignore"):
        if normalize == "true":
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm / cm.sum()
        cm = np.nan_to_num(cm)

    return cm
