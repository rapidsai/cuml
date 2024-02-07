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

import math
import cuml
from cuml.internals.input_utils import input_to_cuml_array, input_to_cupy_array
from cuml.internals.memory_utils import with_cupy_rmm
from cuml.internals.import_utils import has_scipy
from cuml.common.kernel_utils import cuda_kernel_factory
from cuml.internals.safe_imports import cpu_only_import
from cuml.internals.safe_imports import gpu_only_import
from cuml.internals.safe_imports import gpu_only_import_from

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")
cp_csr_matrix = gpu_only_import_from("cupyx.scipy.sparse", "csr_matrix")
cp_coo_matrix = gpu_only_import_from("cupyx.scipy.sparse", "coo_matrix")
cp_csc_matrix = gpu_only_import_from("cupyx.scipy.sparse", "csc_matrix")


if has_scipy():
    from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
else:
    from cuml.common.import_utils import DummyClass

    csr_matrix = DummyClass
    coo_matrix = DummyClass
    csc_matrix = DummyClass


def _map_l1_norm_kernel(dtype):
    """Creates cupy RawKernel for csr_raw_normalize_l1 function."""

    map_kernel_str = r"""
    ({0} *data, {1} *indices, {2} *indptr, int n_samples) {

      int tid = blockDim.x * blockIdx.x + threadIdx.x;

      if(tid >= n_samples) return;
      {0} sum = 0.0;


      for(int i = indptr[tid]; i < indptr[tid+1]; i++) {
        sum += fabs(data[i]);
      }


      if(sum == 0) return;

      for(int i = indptr[tid]; i < indptr[tid+1]; i++) {
        data[i] /= sum;
      }
    }
    """
    return cuda_kernel_factory(map_kernel_str, dtype, "map_l1_norm_kernel")


def _map_l2_norm_kernel(dtype):
    """Creates cupy RawKernel for csr_raw_normalize_l2 function."""

    map_kernel_str = r"""
    ({0} *data, {1} *indices, {2} *indptr, int n_samples) {

      int tid = blockDim.x * blockIdx.x + threadIdx.x;

      if(tid >= n_samples) return;
      {0} sum = 0.0;

      for(int i = indptr[tid]; i < indptr[tid+1]; i++) {
        sum += (data[i] * data[i]);
      }

      if(sum == 0) return;

      sum = sqrt(sum);

      for(int i = indptr[tid]; i < indptr[tid+1]; i++) {
        data[i] /= sum;
      }
    }
    """
    return cuda_kernel_factory(map_kernel_str, dtype, "map_l2_norm_kernel")


@cuml.internals.api_return_any()
def csr_row_normalize_l1(X, inplace=True):
    """Row normalize for csr matrix using the l1 norm"""
    if not inplace:
        X = X.copy()

    kernel = _map_l1_norm_kernel((X.dtype, X.indices.dtype, X.indptr.dtype))
    kernel(
        (math.ceil(X.shape[0] / 32),),
        (32,),
        (X.data, X.indices, X.indptr, X.shape[0]),
    )

    return X


@cuml.internals.api_return_any()
def csr_row_normalize_l2(X, inplace=True):
    """Row normalize for csr matrix using the l2 norm"""
    if not inplace:
        X = X.copy()

    kernel = _map_l2_norm_kernel((X.dtype, X.indices.dtype, X.indptr.dtype))
    kernel(
        (math.ceil(X.shape[0] / 32),),
        (32,),
        (X.data, X.indices, X.indptr, X.shape[0]),
    )

    return X


@cuml.internals.api_return_any()
def csr_diag_mul(X, y, inplace=True):
    """Multiply a sparse X matrix with diagonal matrix y"""
    if not inplace:
        X = X.copy()
    # grab underlying dense ar from y
    y = y.data[0]
    X.data *= y[X.indices]
    return X


@cuml.internals.api_return_any()
def create_csr_matrix_from_count_df(
    count_df, empty_doc_ids, n_doc, n_features, dtype=np.float32
):
    """
    Create a sparse matrix from the count of tokens by document

    Parameters
    ----------
        count_df = cudf.DataFrame({'count':..., 'doc_id':.., 'token':.. })
                    sorted by doc_id and token
        empty_doc_ids = cupy array containing doc_ids with no tokens
        n_doc: Total number of documents
        n_features: Number of features
        dtype: Output dtype
    """
    data = count_df["count"].values
    indices = count_df["token"].values

    doc_token_counts = count_df["doc_id"].value_counts().reset_index()
    del count_df

    doc_token_counts = doc_token_counts.rename(
        {"count": "token_counts"}, axis=1
    ).sort_values(by="doc_id")

    token_counts = _insert_zeros(
        doc_token_counts["token_counts"], empty_doc_ids
    )
    indptr = token_counts.cumsum()
    indptr = cp.pad(indptr, (1, 0), "constant")

    return cupyx.scipy.sparse.csr_matrix(
        arg1=(data, indices, indptr), dtype=dtype, shape=(n_doc, n_features)
    )


def _insert_zeros(ary, zero_indices):
    """
    Create a new array of len(ary + zero_indices) where zero_indices
    indicates indexes of 0s in the new array. Ary is used to fill the rest.

    Examples
    --------
        _insert_zeros([1, 2, 3], [1, 3]) => [1, 0, 2, 0, 3]
    """
    if len(zero_indices) == 0:
        return ary.values

    new_ary = cp.zeros((len(ary) + len(zero_indices)), dtype=cp.int32)

    # getting mask of non-zeros
    data_mask = ~cp.in1d(
        cp.arange(0, len(new_ary), dtype=cp.int32), zero_indices
    )

    new_ary[data_mask] = ary
    return new_ary


@with_cupy_rmm
def extract_knn_graph(knn_graph):
    """
    Converts KNN graph from CSR, COO and CSC formats into separate
    distance and indice arrays. Input can be a cupy sparse graph (device)
    or a numpy sparse graph (host).
    """
    if isinstance(knn_graph, (csc_matrix, cp_csc_matrix)):
        knn_graph = cupyx.scipy.sparse.csr_matrix(knn_graph)
        n_samples = knn_graph.shape[0]
        reordering = knn_graph.data.reshape((n_samples, -1))
        reordering = reordering.argsort()
        n_neighbors = reordering.shape[1]
        reordering += (cp.arange(n_samples) * n_neighbors)[:, np.newaxis]
        reordering = reordering.flatten()
        knn_graph.indices = knn_graph.indices[reordering]
        knn_graph.data = knn_graph.data[reordering]

    knn_indices = None
    if isinstance(knn_graph, (csr_matrix, cp_csr_matrix)):
        knn_indices = knn_graph.indices
    elif isinstance(knn_graph, (coo_matrix, cp_coo_matrix)):
        knn_indices = knn_graph.col

    if knn_indices is not None:
        knn_dists = knn_graph.data
        return knn_indices, knn_dists
    else:
        return None


@with_cupy_rmm
def extract_pairwise_dists(pw_dists, n_neighbors):
    """
    Extract the nearest neighbors distances and indices
    from a pairwise distance matrix.

    Parameters
    ----------
        pw_dists: paiwise distances matrix of shape (n_samples, n_samples)
        n_neighbors: number of nearest neighbors

    (inspired from Scikit-Learn code)
    """
    pw_dists, _, _, _ = input_to_cupy_array(pw_dists)

    n_rows = pw_dists.shape[0]
    sample_range = cp.arange(n_rows)[:, None]
    knn_indices = cp.argpartition(pw_dists, n_neighbors - 1, axis=1)
    knn_indices = knn_indices[:, :n_neighbors]
    argdist = cp.argsort(pw_dists[sample_range, knn_indices])
    knn_indices = knn_indices[sample_range, argdist]
    knn_dists = pw_dists[sample_range, knn_indices]
    return knn_indices, knn_dists


@with_cupy_rmm
def extract_knn_infos(knn_info, n_neighbors):
    """
    Extract the nearest neighbors distances and indices
    from the knn_info parameter.

    Parameters
    ----------
        knn_info : array / sparse array / tuple, optional (device or host)
        Either one of :
            - Tuple (indices, distances) of arrays of
              shape (n_samples, n_neighbors)
            - Pairwise distances dense array of shape (n_samples, n_samples)
            - KNN graph sparse array (preferably CSR/COO)
        n_neighbors: number of nearest neighbors
    """
    if knn_info is None:
        # no KNN was provided
        return None

    deepcopy = False
    if isinstance(knn_info, tuple):
        # dists and indices provided as a tuple
        results = knn_info
    else:
        isaKNNGraph = isinstance(
            knn_info,
            (
                csr_matrix,
                coo_matrix,
                csc_matrix,
                cp_csr_matrix,
                cp_coo_matrix,
                cp_csc_matrix,
            ),
        )
        if isaKNNGraph:
            # extract dists and indices from a KNN graph
            deepcopy = True
            results = extract_knn_graph(knn_info)
        else:
            # extract dists and indices from a pairwise distance matrix
            results = extract_pairwise_dists(knn_info, n_neighbors)

    if results is not None:
        knn_indices, knn_dists = results

        knn_indices_m, _, _, _ = input_to_cuml_array(
            knn_indices.flatten(),
            order="C",
            deepcopy=deepcopy,
            check_dtype=np.int64,
            convert_to_dtype=np.int64,
        )

        knn_dists_m, _, _, _ = input_to_cuml_array(
            knn_dists.flatten(),
            order="C",
            deepcopy=deepcopy,
            check_dtype=np.float32,
            convert_to_dtype=np.float32,
        )

        return knn_indices_m, knn_dists_m
    else:
        return None
