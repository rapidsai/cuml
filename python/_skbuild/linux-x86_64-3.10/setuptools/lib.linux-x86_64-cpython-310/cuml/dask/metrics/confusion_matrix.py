#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from cuml.prims.label import make_monotonic
from cuml.dask.metrics.utils import sorted_unique_labels
from cuml.internals.memory_utils import with_cupy_rmm
from cuml.dask.common.utils import get_client
from cuml.dask.common.input_utils import DistributedDataHandler
from cuml.internals.safe_imports import gpu_only_import
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")


@with_cupy_rmm
def _local_cm(inputs, labels, use_sample_weight):
    if use_sample_weight:
        y_true, y_pred, sample_weight = inputs
    else:
        y_true, y_pred = inputs
        sample_weight = cp.ones(y_true.shape[0], dtype=y_true.dtype)

    y_true, _ = make_monotonic(y_true, labels, copy=True)
    y_pred, _ = make_monotonic(y_pred, labels, copy=True)

    n_labels = labels.size

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = cp.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]
    sample_weight = sample_weight[ind]
    cm = cupyx.scipy.sparse.coo_matrix(
        (sample_weight, (y_true, y_pred)),
        shape=(n_labels, n_labels),
        dtype=cp.float64,
    ).toarray()
    return cp.nan_to_num(cm)


@with_cupy_rmm
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
        or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : dask.Array (device or host) shape = (n_samples,)
        or (n_samples, n_outputs)
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
        labels = sorted_unique_labels(y_true, y_pred)

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
