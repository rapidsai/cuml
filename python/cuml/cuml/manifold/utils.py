#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import math

import cupy as cp
import numpy as np

from cuml.common.kernel_utils import cuda_kernel_factory
from cuml.common.sparse import is_sparse
from cuml.internals.validation import check_array


def _check_indices_per_row(indptr, n_neighbors):
    """Check if indptr indicates n_neighbors per row"""
    if isinstance(indptr, cp.ndarray):
        out = cp.ones(1, dtype="int32")
        kernel = cuda_kernel_factory(
            """
            (const {0}* arr, int n_rows, int n_neighbors, int *out) {
                int row = blockDim.x * blockIdx.x + threadIdx.x;

                if (row + 1 >= n_rows) return;
                if (arr[row + 1] - arr[row] != n_neighbors) {
                    *out = 0;
                }
                return;
            }
            """,
            (indptr.dtype,),
            "has_n_neighbors_per_row",
        )
        kernel(
            (math.ceil(indptr.shape[0] / 32),),
            (32,),
            (indptr, indptr.shape[0], n_neighbors, out),
        )
        return bool(out.item())
    return (np.diff(indptr) == n_neighbors).all()


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


def _check_distances_sorted(dist, orig_n_neighbors):
    """Checks if axis 1 (every row) is sorted appropriately in dist."""
    # Safety check
    assert dist.dtype == "float32"
    # XXX: ensure on device just for this routine - zero copy unless a
    # host->device transfer needed.
    dist = cp.asarray(dist, order="C")
    is_sorted = cp.ones(1, dtype="int32")
    _is_axis_1_sorted_kernel(
        (math.ceil(dist.shape[0] / 32),),
        (32,),
        (dist, dist.shape[0] // orig_n_neighbors, orig_n_neighbors, is_sorted),
    )
    return bool(is_sorted.item())


def _check_self_references(indices, orig_n_neighbors=None):
    """Checks if indices has self references in column 0."""
    if orig_n_neighbors is not None:
        indices = indices.reshape((-1, orig_n_neighbors))
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
    n_samples,
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
    n_samples: int
        Number of samples expected.
    n_neighbors: int
        Number of nearest neighbors required.
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
        if indices.shape[0] != n_samples or indices.shape != distances.shape:
            raise ValueError(
                f"Expected indices and distances to have shape=(n_samples, "
                f"n_neighbors) where {n_samples=}, got "
                f"indices.shape={indices.shape}, distances.shape={distances.shape}"
            )
        if not _check_self_references(indices):
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
        if not (
            knn_info.ndim == 2
            and knn_info.shape[0] == knn_info.shape[1] == n_samples
        ):
            raise ValueError(
                f"Expected a sparse array of shape=(n_samples, n_samples) where "
                f"{n_samples=}, got shape={knn_info.shape}"
            )

        # Coerce to CSR. If the input was already CSR this is zero-copy and
        # avoids reordering indices (leaving `.data` in the initial order).
        # This ensures the case of passing a direct `kneighbors_graph` output
        # can be done zero-copy.
        knn_info = check_array(
            knn_info,
            accept_sparse=["csr"],
            accept_large_sparse=(indices_dtype == "int64"),
            mem_type=mem_type_init,
            dtype="float32",
        )
        xp = cp if isinstance(knn_info.data, cp.ndarray) else np

        orig_n_neighbors, remainder = divmod(knn_info.nnz, n_samples)
        if (
            remainder == 0
            and _check_indices_per_row(knn_info.indptr, orig_n_neighbors)
            and _check_distances_sorted(knn_info.data, orig_n_neighbors)
            and _check_self_references(knn_info.indices, orig_n_neighbors)
        ):
            # The input graph is usable as is with no copies needed.
            distances = knn_info.data.reshape((n_samples, orig_n_neighbors))
            indices = knn_info.indices.reshape((n_samples, orig_n_neighbors))
        else:
            # Graph `data` and `indices` aren't in the correct format, we
            # need to copy and massage the data.
            knn_info = knn_info.copy()

            # Set an explicit value for all diagonal elements, ensuring self
            # references are present. We use -1 to ensure that self references
            # sort earlier than any other 0 distance elements.
            knn_info.setdiag(knn_info.dtype.type(-1))

            # Recompute and validate orig_n_neighbors. An error here indicates
            # either an invalid KNN graph or one where some samples had 0
            # distance to each other _and_ the 0 entries were dropped. We cannot
            # recover that information and have to error.
            orig_n_neighbors, remainder = divmod(knn_info.nnz, n_samples)
            if not (
                remainder == 0
                and _check_indices_per_row(knn_info.indptr, orig_n_neighbors)
            ):
                raise ValueError(
                    f"Precomputed KNN graph has {knn_info.nnz - n_samples} "
                    f"nonzero elements which is not evenly divisible by "
                    f"{n_samples} samples. The graph may be malformed, or may "
                    f"have contained 0-distance samples that were removed "
                    f"during sparse matrix canonicalization."
                )

            # Extract distances and indices into individual arrays
            distances = knn_info.data.reshape((n_samples, orig_n_neighbors))
            indices = knn_info.indices.reshape((n_samples, orig_n_neighbors))

            # Sort each row by distance.
            new_order = distances.argsort()
            all_rows = xp.arange(distances.shape[0])[:, None]
            indices = indices[all_rows, new_order]
            distances = distances[all_rows, new_order]
            del new_order

            # Finally swap -1 distances for self references back to 0
            distances[:, 0] = 0
    else:
        # Dense pairwise distance matrix, shape=(n_samples, n_samples)
        knn_info = check_array(
            knn_info, dtype="float32", mem_type=mem_type_init
        )
        if not (knn_info.shape[0] == knn_info.shape[1] == n_samples):
            raise ValueError(
                f"Expected a dense array of shape=(n_samples, n_samples) where "
                f"{n_samples=}, got shape={knn_info.shape}"
            )
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
