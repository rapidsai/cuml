from cuml.internals.safe_imports import cpu_only_import
np = cpu_only_import('numpy')
pd = cpu_only_import('pandas')
import warnings
from cuml.internals.safe_imports import gpu_only_import
cupy = gpu_only_import('cupy')

import cuml.internals
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.base import Base
from pylibraft.common.handle cimport handle_t
from pylibraft.common.handle import Handle
import cuml.internals.logger as logger

from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.common.sparse_utils import is_sparse
from cuml.common.doc_utils import generate_docstring
from cuml.common import input_to_cuml_array
from cuml.internals.mixins import CMajorInputTagMixin
from cuml.common.sparsefuncs import extract_knn_infos
from cuml.metrics.distance_type cimport DistanceType
rmm = gpu_only_import('rmm')

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdint cimport int64_t
from libc.stdlib cimport free
from cython.operator cimport dereference as deref

cimport cuml.common.cuda

# This file is to test Spectral::fit_embedding
# which takes in a symmetric adjacency matrix

# then it will be turned into a normalized laplacian
# then the lanczos solver will find the eigenvalues/vectors

cdef extern from "cuml/cluster/spectral.hpp" namespace "ML::Spectral":
    
    cdef void fit_embedding(
        const handle_t& handle,
        int* rows,
        int* cols,
        float* vals,
        int nnz,
        int n,
        int n_components,
        float* out,
        unsigned long long seed) except +

    cdef void lanczos_solver(
        const handle_t& handle,
        int* rows,
        int* cols,
        float* vals,
        int nnz,
        int n,
        int n_components,
        float* eigenvectors,
        float* eigenvalues,
        int* eig_iters,
        unsigned long long seed,
        int maxiter,
        float tol,
        int conv_n_iters,
        float conv_eps,
        int restartiter) except +

    cdef void lanczos_solver(
        const handle_t& handle,
        int* rows,
        int* cols,
        double* vals,
        int nnz,
        int n,
        int n_components,
        double* eigenvectors,
        double* eigenvalues,
        int* eig_iters,
        unsigned long long seed,
        int maxiter,
        float tol,
        int conv_n_iters,
        float conv_eps,
        int restartiter) except +


@cuml.internals.api_return_array(get_output_type=True)
def lanczos(A, n_components, seed, handle):

    # rows = CumlArrayDescriptor()
    # cols = CumlArrayDescriptor()
    # vals = CumlArrayDescriptor()

    Acoo = A.tocoo()
    rows = Acoo.row
    cols = Acoo.col
    vals = Acoo.data

    rows, n, p, _ = \
            input_to_cuml_array(rows, order='C', check_dtype=np.int32,
                                convert_to_dtype=(np.int32))
        
    cols, n, p, _ = \
            input_to_cuml_array(cols, order='C', check_dtype=np.int32,
                                convert_to_dtype=(np.int32))

    vals, n, p, _ = \
            input_to_cuml_array(vals, order='C', check_dtype=np.float32,
                                convert_to_dtype=(np.float32))

    N = A.shape[0]
    
    embed = CumlArray.zeros(
            (N, n_components),
            order="F",
            dtype=np.float32)

    cdef uintptr_t embed_ptr = embed.ptr
    cdef uintptr_t rows_ptr = rows.ptr
    cdef uintptr_t cols_ptr = cols.ptr
    cdef uintptr_t vals_ptr = vals.ptr

    handle = Handle() if handle is None else handle
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

    fit_embedding(
        handle_[0],
        <int*> rows_ptr,
        <int*> cols_ptr,
        <float*> vals_ptr,
        <int> A.nnz,
        <int> N,
        <int> n_components,
        <float*> embed_ptr,
        <long long> seed,
    );

    return embed


# write solver only function
# no laplacian computed

# @cuml.internals.api_return_array(get_output_type=True)
def eig_lanczos(A, n_components, seed, dtype, maxiter=4000, tol=0.01, conv_n_iters = 5, conv_eps = 0.001, restartiter=15, handle=Handle()):

    # rows = CumlArrayDescriptor()
    # cols = CumlArrayDescriptor()
    # vals = CumlArrayDescriptor()

    Acoo = A.tocoo()
    rows = Acoo.row
    cols = Acoo.col
    vals = Acoo.data

    rows, n, p, _ = \
            input_to_cuml_array(rows, order='C', check_dtype=np.int32,
                                convert_to_dtype=(np.int32))
        
    cols, n, p, _ = \
            input_to_cuml_array(cols, order='C', check_dtype=np.int32,
                                convert_to_dtype=(np.int32))

    vals, n, p, _ = \
            input_to_cuml_array(vals, order='C', check_dtype=dtype,
                                convert_to_dtype=(dtype))

    N = A.shape[0]
    
    eigenvectors = CumlArray.zeros(
                    (N, n_components),
                    order="F",
                    dtype=dtype)
    
    eigenvalues = CumlArray.zeros(
                    (n_components),
                    order="F",
                    dtype=dtype)

    cdef int eig_iters = 0
    cdef int* eig_iters_ptr = &eig_iters

    cdef uintptr_t eigenvectors_ptr = eigenvectors.ptr
    cdef uintptr_t eigenvalues_ptr = eigenvalues.ptr
    cdef uintptr_t rows_ptr = rows.ptr
    cdef uintptr_t cols_ptr = cols.ptr
    cdef uintptr_t vals_ptr = vals.ptr

    handle = Handle() if handle is None else handle
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()
    

    if dtype == np.float32:
        lanczos_solver(
            handle_[0],
            <int*> rows_ptr,
            <int*> cols_ptr,
            <float*> vals_ptr,
            <int> A.nnz,
            <int> N,
            <int> n_components,
            <float*> eigenvectors_ptr,
            <float*> eigenvalues_ptr,
            <int*> eig_iters_ptr,
            <long long> seed,
            <int> maxiter,
            <float> tol,
            <int> conv_n_iters,
            <float> conv_eps,
            <int> restartiter,
        );
    elif dtype == np.float64:
        lanczos_solver(
            handle_[0],
            <int*> rows_ptr,
            <int*> cols_ptr,
            <double*> vals_ptr,
            <int> A.nnz,
            <int> N,
            <int> n_components,
            <double*> eigenvectors_ptr,
            <double*> eigenvalues_ptr,
            <int*> eig_iters_ptr,
            <long long> seed,
            <int> maxiter,
            <float> tol,
            <int> conv_n_iters,
            <float> conv_eps,
            <int> restartiter,
        );

    return cupy.asnumpy(eigenvalues), cupy.asnumpy(eigenvectors), eig_iters
