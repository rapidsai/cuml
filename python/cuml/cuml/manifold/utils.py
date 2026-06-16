#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import math

import cupy as cp
import numpy as np

from cuml.common.kernel_utils import cuda_kernel_factory
from cuml.common.sparse_utils import is_sparse
from cuml.internals.validation import check_array

_is_axis_1_sorted_kernel = cp.RawKernel(
    """
    extern "C" __global__
    void is_axis_1_sorted(const float* arr, int n_rows, int n_cols, int *sorted) {
        int row = blockDim.x * blockIdx.x + threadIdx.x;

        if (row >= n_rows) return;

        int start = row * n_cols;
        int end = start + n_cols - 1;
        for (int i = start; i < end; i++) {
            if (arr[i] > arr[i + 1]) {
                *sorted = 0;
                return;
            }
        }
    }
    """,
    "is_axis_1_sorted",
)


def _is_axis_1_sorted(X):
    """Checks if axis 1 (every row) is sorted in X."""
    # Safety checks
    assert X.flags.c_contiguous
    assert X.dtype == "float32"
    assert X.ndim == 2
    # XXX: ensure on device just for this routine - zero copy unless a
    # host->device transfer needed.
    X = cp.asarray(X, order="C")
    is_sorted = cp.ones(1, dtype="int32")
    _is_axis_1_sorted_kernel(
        (math.ceil(X.shape[0] / 32),),
        (32,),
        (X, X.shape[0], X.shape[1], is_sorted),
    )
    return bool(is_sorted.item())


def _has_self_references(indices):
    """Checks if indices has self references in column 0."""
    # Safety checks
    assert indices.flags.c_contiguous
    assert indices.ndim == 2
    n_rows, n_cols = indices.shape
    if isinstance(indices, cp.ndarray):
        out = cp.ones(1, dtype="int32")
        kernel = cuda_kernel_factory(
            """
            (const {0}* arr, int n_rows, int n_cols, int *out) {
                int row = blockDim.x * blockIdx.x + threadIdx.x;

                if (row >= n_rows) return;
                if (arr[row * n_cols] != row) {
                    *out = 0;
                }
                return;
            }
            """,
            (indices.dtype,),
            "has_self_references",
        )
        kernel(
            (math.ceil(n_rows / 32),), (32,), (indices, n_rows, n_cols, out)
        )
        return bool(out.item())
    return (indices[:, 0] == np.arange(n_rows, dtype=indices.dtype)).all()


def extract_knn_graph(
    knn_info,
    n_neighbors,
    mem_type="device",
    indices_dtype="int64",
):
    """Extract the KNN graph indices and distances.

    Parameters
    ----------
    knn_info : array, sparse-matrix, or tuple[array, array]
        - Tuple (indices, distances) of arrays of shape (n_samples,
          n_neighbors). Should contain self references (i.e. the closest sample
          to a row is the row itself).
        - Pairwise distances dense array of shape (n_samples, n_samples).
        - KNN graph sparse array. This is most efficient if the graph is in CSR
          format and contains 0 entries in `data` for all diagonal elements.
    n_neighbors: int
        Number of nearest neighbors
    mem_type : {"device", "host", None}, default="device"
        The desired output memory type.
    indices_dtype : dtype, default='int64'
        The dtype to use for the output indices.

    Returns
    -------
    indices : cupy.ndarray or numpy.ndarray
        The KNN indices, shape=n_samples * n_neighbors, dtype=indices_dtype.
    distances : cupy.ndarray or numpy.ndarray
        The KNN distances, shape=n_samples * n_neighbors, dtype=float32.
    """
    # The initial mem_type to coerce to. When possible we only coerce to device
    # if the output is known to be device, otherwise we leave as is until the
    # final coercion.
    mem_type_init = "device" if mem_type == "device" else None
    if isinstance(knn_info, tuple):
        # (indices, distances), each with shape=(n_samples, orig_n_neighbors)
        indices, distances = knn_info
        indices = check_array(
            indices, dtype=indices_dtype, order="C", mem_type=mem_type_init
        )
        distances = check_array(
            distances, dtype="float32", order="C", mem_type=mem_type_init
        )
        if not indices.shape == distances.shape:
            raise ValueError(
                f"Expected indices and distances to have shape=(n_samples, "
                f"n_neighbors), got indices.shape={indices.shape}, "
                f"distances.shape={distances.shape}"
            )
        if not _has_self_references(indices):
            raise ValueError(
                "Expected indices and distances to include self references (i.e. "
                "the closest sample to each row is itself). If using "
                "`NearestNeighbors.kneighbors` to precompute the KNN, pass in "
                "the training data to both `NearestNeighbors.fit` and "
                "`NearestNeighbors.kneighbors`."
            )
    elif is_sparse(knn_info):
        # Sparse KNN graph
        # - shape=(n_samples, n_samples)
        # - nnz=n_samples * orig_n_neighbors
        if not (knn_info.ndim == 2 and knn_info.shape[0] == knn_info.shape[1]):
            raise ValueError(
                f"Expected a sparse array of shape=(n_samples, n_samples), "
                f"got shape={knn_info.shape}"
            )

        # Coerce to CSR. If the input was already CSR this is zero-copy and
        # avoids reordering indices (leaving `.data` in the initial order).
        # This ensures the case of passing a direct `kneighbors_graph` output
        # can be done zero-copy.
        knn_info = knn_info.tocsr()

        # XXX: Here we try to do some rudimentary inference and validation of
        # the original K on the provided KNN graph. Since sparse inputs may
        # drop 0 entries, some diagonal entries may be _close_ to zero but not
        # exactly zero, and the graph may or may not be canonicalized this is
        # only a best-effort. Some invalid graphs may sneak through here, but
        # we try to catch common errors and issues and raise them.
        n_samples = knn_info.shape[0]
        if (remainder := knn_info.nnz % n_samples) != 0:
            # This might still be valid if _some_ diagonal distances are
            # _close_ to zero but not actually zero, _and_ the actually zero
            # elements were dropped. Before doing a copy, check if dropping
            # the diagonal will still result in an invalid graph.
            if (knn_info.diagonal() != 0).sum() != remainder:
                raise ValueError(
                    f"Precomputed KNN graph has {knn_info.nnz} nonzero elements which "
                    f"is not evenly divisible by {n_samples} samples."
                )
            knn_info = knn_info.copy()
            knn_info.setdiag(knn_info.dtype.type(0))
            knn_info.eliminate_zeros()
        orig_n_neighbors = knn_info.nnz // n_samples

        indices = check_array(
            knn_info.indices.reshape((n_samples, orig_n_neighbors)),
            dtype=indices_dtype,
            order="C",
            mem_type=mem_type_init,
        )
        distances = check_array(
            knn_info.data.reshape((n_samples, orig_n_neighbors)),
            dtype="float32",
            order="C",
            mem_type=mem_type_init,
        )
        xp = cp if isinstance(distances, cp.ndarray) else np
        # Reorder by distances if not already sorted. This is necessary for KNN
        # graph inputs, since a canonical sparse matrix will not have the data
        # sorted as we need it. Both sklearn and cuml's `kneighbors_graph`
        # returns a matrix with `.data` sorted as required (not canonicalized),
        # so we optimistically check for sortedness before doing the sorting.
        if not _is_axis_1_sorted(distances):
            new_order = distances.argsort()
            all_rows = xp.arange(distances.shape[0])[:, None]
            indices = indices[all_rows, new_order]
            distances = distances[all_rows, new_order]
            del new_order

        if not _has_self_references(indices):
            # No self references present in the data. The KNN graph was either
            # computed without self references, or 0 elements were dropped
            # during a canonicalization step at some point.
            temp = xp.zeros(
                (n_samples, orig_n_neighbors + 1),
                dtype=distances.dtype,
                order="C",
            )
            temp[:, 1:] = distances
            distances = temp
            temp = xp.empty(
                (n_samples, orig_n_neighbors + 1),
                dtype=indices.dtype,
                order="C",
            )
            temp[:, 0] = xp.arange(n_samples, dtype=indices.dtype)
            temp[:, 1:] = indices
            indices = temp
            orig_n_neighbors += 1
    else:
        # Dense pairwise distance matrix, shape=(n_samples, n_samples)
        knn_info = check_array(
            knn_info, dtype="float32", mem_type=mem_type_init
        )
        if knn_info.shape[0] != knn_info.shape[1]:
            raise ValueError(
                f"Expected a dense array of shape=(n_samples, n_samples), "
                f"got shape={knn_info.shape}"
            )
        n_samples = knn_info.shape[0]
        if n_samples < n_neighbors:
            raise ValueError(
                f"Precomputed KNN data requires n_samples >= n_neighbors. "
                f"Got {n_neighbors=}, {n_samples=}"
            )

        # Convert pairwise distance matrix to KNN graph
        xp = cp if isinstance(knn_info, cp.ndarray) else np
        # Partition indices to select the nearest `n_neighbors`
        indices = xp.argpartition(knn_info, n_neighbors - 1, axis=1)
        indices = indices[:, :n_neighbors]
        # Reorder and subset indices and distances appropriately
        all_rows = xp.arange(n_samples)[:, None]
        indices = indices[all_rows, xp.argsort(knn_info[all_rows, indices])]
        distances = knn_info[all_rows, indices]

    # Validate shape and n_neighbors
    if indices.shape[1] < n_neighbors:
        raise ValueError(
            f"Precomputed KNN data has {indices.shape[1]} neighbors per "
            f"sample, but {n_neighbors=} was specified. Please provide KNN data "
            f"with at least {n_neighbors} neighbors per sample."
        )

    # Trim arrays to n_neighbors if necessary
    if indices.shape[1] > n_neighbors:
        indices = indices[:, :n_neighbors]
        distances = distances[:, :n_neighbors]

    # Reshape and coerce to proper dtype and mem_type.
    indices = check_array(
        indices.reshape(-1),
        dtype=indices_dtype,
        mem_type=mem_type,
        ensure_2d=False,
    )
    distances = check_array(
        distances.reshape(-1),
        dtype="float32",
        mem_type=("device" if isinstance(indices, cp.ndarray) else "host"),
        ensure_2d=False,
        ensure_all_finite=False,
    )
    return indices, distances
