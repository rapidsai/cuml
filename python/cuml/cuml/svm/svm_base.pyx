# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
import cupyx.scipy.sparse
import numpy as np
import scipy.sparse

import cuml.internals
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.base import Base
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import FMajorInputTagMixin, SparseInputTagMixin

cimport cython
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

cimport cuml.svm.svm_headers as lib


@cython.no_gc_clear
cdef class _SVMModel:
    """A thin python wrapper for an SvmModel.

    Used to keep the memory around until all references to
    the underlying model are dropped."""
    cdef lib.SvmModel[float] *model_f
    cdef lib.SvmModel[double] *model_d
    cdef object handle

    @staticmethod
    cdef _SVMModel new(handle, bool is_float32):
        cdef _SVMModel self = _SVMModel.__new__(_SVMModel)
        self.handle = handle
        if is_float32:
            self.model_f = new lib.SvmModel[float]()
        else:
            self.model_d = new lib.SvmModel[double]()
        return self

    def __dealloc__(self):
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        if self.model_f != NULL:
            lib.svmFreeBuffers(handle_[0], self.model_f[0])
            del self.model_f
            self.model_f = NULL
        elif self.model_d != NULL:
            lib.svmFreeBuffers(handle_[0], self.model_d[0])
            del self.model_d
            self.model_d = NULL

    cdef _ptr_as_cupy(self, ptr, shape, dtype):
        """Expose a pointer managed by this model as a cupy array"""
        if ptr == 0:
            return cp.empty(shape=shape, dtype=dtype, order="F")
        dtype = np.dtype(dtype)
        mem = cp.cuda.UnownedMemory(
            ptr=ptr, size=np.prod(shape) * dtype.itemsize, owner=self,
        )
        mem_ptr = cp.cuda.memory.MemoryPointer(mem, 0)
        return cp.ndarray(shape=shape, dtype=dtype, order="F", memptr=mem_ptr)

    cdef unpack(self):
        """Unpack the model into its array components"""
        cdef double b
        cdef int n_support, n_cols, nnz
        cdef uintptr_t dual_coef_ptr, support_idx_ptr
        cdef uintptr_t data_ptr, indptr_ptr, indices_ptr

        if self.model_f != NULL:
            n_support = self.model_f.n_support
            n_cols = self.model_f.n_cols
            nnz = self.model_f.support_matrix.nnz
            b = self.model_f.b
            dual_coef_ptr = <uintptr_t>self.model_f.dual_coefs
            support_idx_ptr = <uintptr_t>self.model_f.support_idx
            data_ptr = <uintptr_t>self.model_f.support_matrix.data
            indptr_ptr = <uintptr_t>self.model_f.support_matrix.indptr
            indices_ptr = <uintptr_t>self.model_f.support_matrix.indices
            dtype = np.float32
        else:
            n_support = self.model_d.n_support
            n_cols = self.model_d.n_cols
            nnz = self.model_d.support_matrix.nnz
            b = self.model_d.b
            dual_coef_ptr = <uintptr_t>self.model_d.dual_coefs
            support_idx_ptr = <uintptr_t>self.model_d.support_idx
            data_ptr = <uintptr_t>self.model_d.support_matrix.data
            indptr_ptr = <uintptr_t>self.model_d.support_matrix.indptr
            indices_ptr = <uintptr_t>self.model_d.support_matrix.indices
            dtype = np.float64

        dual_coef = CumlArray(
            data=self._ptr_as_cupy(dual_coef_ptr, (1, n_support), dtype)
        )

        support = CumlArray(
            data=self._ptr_as_cupy(support_idx_ptr, (n_support,), np.int32)
        )

        intercept = CumlArray.full(1, b, dtype=dtype)

        if nnz == -1:
            support_vectors = CumlArray(
                data=self._ptr_as_cupy(data_ptr, (n_support, n_cols), dtype)
            )
        else:
            indptr = self._ptr_as_cupy(indptr_ptr, (n_support + 1,), np.int32)
            indices = self._ptr_as_cupy(indices_ptr, (nnz,), np.int32)
            data = self._ptr_as_cupy(data_ptr, (nnz,), dtype)
            support_vectors = SparseCumlArray(
                data=cupyx.scipy.sparse.csr_matrix(
                    (data, indices, indptr),
                    shape=(n_support, n_cols),
                )
            )

        return support, support_vectors, dual_coef, intercept


class SVMBase(Base,
              InteropMixin,
              FMajorInputTagMixin,
              SparseInputTagMixin):
    """Base class for Support Vector Machines"""

    support_ = CumlArrayDescriptor(order="F")
    support_vectors_ = CumlArrayDescriptor(order="F")
    dual_coef_ = CumlArrayDescriptor(order="F")
    intercept_ = CumlArrayDescriptor(order="F")

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "kernel",
            "degree",
            "gamma",
            "coef0",
            "tol",
            "C",
            "cache_size",
            "max_iter",
            "nochange_steps",
            "epsilon",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.kernel == "precomputed" or callable(model.kernel):
            raise UnsupportedOnGPU(f"`kernel={model.kernel!r}` is not supported")

        if (cache_size := model.cache_size) == 200:
            # XXX: the cache sizes differ between cuml and sklearn, for now we
            # just adjust when the value's match the defaults.
            cache_size = 1024.0

        return {
            "kernel": model.kernel,
            "degree": model.degree,
            "gamma": model.gamma,
            "coef0": model.coef0,
            "tol": model.tol,
            "C": model.C,
            "cache_size": cache_size,
            "max_iter": model.max_iter,
            "epsilon": model.epsilon,
        }

    def _params_to_cpu(self):
        if (cache_size := self.cache_size) == 1024:
            # XXX: the cache sizes differ between cuml and sklearn, for now we
            # just adjust when the value's match the defaults.
            cache_size = 200

        return {
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self.gamma,
            "coef0": self.coef0,
            "tol": self.tol,
            "C": self.C,
            "cache_size": cache_size,
            "max_iter": self.max_iter,
            "epsilon": self.epsilon,
        }

    def _attrs_from_cpu(self, model):
        # Computing n_support_ directly from support_vectors_ since
        # model.n_support_ is not always reliably computed.
        n_support, n_cols = model.support_vectors_.shape

        if model._sparse:
            # sklearn stores dual_coef_ and support_vectors_ as sparse
            # csr_matrix objects when fit on sparse data.
            # cuml always stores dual_coef_ as dense, and will optionally
            # store support_vectors_ as dense if it's smaller than 1 GiB.
            dual_coef_ = to_gpu(model.dual_coef_.toarray(), order="F")
            if (n_support * n_cols * 8) < (1 << 30):
                # support vectors are "small enough", store as dense
                support_vectors = to_gpu(
                    model.support_vectors_.toarray(), dtype=np.float64, order="F",
                )
            else:
                support_vectors = SparseCumlArray(
                    model.support_vectors_,
                    convert_to_dtype=np.float64,
                    convert_format=True
                )
        else:
            dual_coef_ = to_gpu(model.dual_coef_, order="F")
            support_vectors = to_gpu(
                model.support_vectors_, dtype=np.float64, order="F"
            )

        return {
            "dual_coef_": dual_coef_,
            "support_vectors_": support_vectors,
            "intercept_": to_gpu(model.intercept_, order="F"),
            "n_support_": int(n_support),
            "support_": to_gpu(model.support_, order="F"),
            "_gamma": float(model._gamma),
            "_sparse": model._sparse,
            "fit_status_": model.fit_status_,
            "shape_fit_": model.shape_fit_,
            "_probA": model._probA,
            "_probB": model._probB,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        intercept_ = to_cpu(self.intercept_, order="C", dtype=np.float64)

        if self._sparse:
            # sklearn stores dual_coef_ and support_vectors_ as sparse
            # csr_matrix objects when fit on sparse data.
            # cuml always stores dual_coef_ as dense, and will optionally
            # store support_vectors_ as dense if it's smaller than 1 GiB
            dual_coef_ = scipy.sparse.csr_matrix(
                self.dual_coef_.to_output("numpy", output_dtype=np.float64)
            )
            support_vectors_ = self.support_vectors_.to_output(
                "numpy", output_dtype=np.float64,
            )
            if not scipy.sparse.issparse(support_vectors_):
                support_vectors_ = scipy.sparse.csr_matrix(support_vectors_)
        else:
            dual_coef_ = to_cpu(self.dual_coef_, order="C", dtype=np.float64)
            support_vectors_ = to_cpu(self.support_vectors_, order="C", dtype=np.float64)

        return {
            "dual_coef_": dual_coef_,
            "_dual_coef_": dual_coef_,
            "fit_status_": self.fit_status_,
            "intercept_": intercept_,
            "_intercept_": intercept_,
            "shape_fit_": self.shape_fit_,
            "_n_support": np.array([self.n_support_, 0], dtype=np.int32),
            "support_": to_cpu(self.support_, order="C", dtype=np.int32),
            "support_vectors_": support_vectors_,
            "_gamma": self._gamma,
            "_probA": self._probA,
            "_probB": self._probB,
            "_sparse": self._sparse,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=1e-3,
        epsilon=0.1,
        cache_size=1024.0,
        max_iter=-1,
        nochange_steps=1000,
        handle=None,
        verbose=False,
        output_type=None,
    ):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.epsilon = epsilon
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.nochange_steps = nochange_steps

    @property
    @cuml.internals.api_base_return_array_skipall
    def coef_(self):
        if self.kernel != "linear":
            raise AttributeError("coef_ is only available for linear kernels")
        dual_coef = self.dual_coef_.to_output("cupy")
        support_vectors = self.support_vectors_.to_output("cupy")
        return CumlArray(data=dual_coef @ support_vectors)

    def _get_gamma(self, X):
        if isinstance(self.gamma, str):
            n_rows, n_cols = X.shape
            if self.gamma == "auto":
                return 1 / n_cols
            elif self.gamma == "scale":
                if isinstance(X, SparseCumlArray):
                    data = X.data.to_output("cupy")
                    n = n_cols * n_rows
                    x_mean = data.mean() * X.nnz / n
                    data = (data - x_mean)**2
                    x_var = ((data.sum() + (n - X.nnz) * x_mean**2) / n).item()
                else:
                    x_var = X.to_output("cupy").var().item()
                return 1 / (n_cols * x_var)
            else:
                raise ValueError(
                    f"`gamma` must be 'auto', 'scale', or a float, got {self.gamma!r}"
                )
        return float(self.gamma)

    def _get_kernel_type(self):
        kernel = {
            "linear": lib.KernelType.LINEAR,
            "poly": lib.KernelType.POLYNOMIAL,
            "rbf": lib.KernelType.RBF,
            "sigmoid": lib.KernelType.TANH
        }.get(self.kernel)
        if kernel is None:
            raise ValueError(
                f"Expected `kernel` to be in [`'linear', 'poly', 'rbf', 'sigmoid'], "
                f"got {kernel}"
            )
        return kernel

    def _fit(self, X, y, sample_weight=None):
        """Perform `fit`.

        Expects X, y, and sample_weight to already be validated and normalized to
        same dtype.
        """
        # Sanity assertions, user-facing errors should be raised earlier
        assert X.dtype == y.dtype
        assert sample_weight is None or X.dtype == sample_weight.dtype

        cdef bool is_classifier = self._estimator_type == "classifier"
        cdef bool is_sparse = isinstance(X, SparseCumlArray)
        cdef bool is_float32 = X.dtype == np.float32

        cdef double gamma = self._get_gamma(X)

        cdef lib.KernelParams kernel_params
        kernel_params.kernel = self._get_kernel_type()
        kernel_params.gamma = gamma
        kernel_params.coef0 = self.coef0
        kernel_params.degree = self.degree

        cdef lib.SvmParameter param
        param.C = self.C
        param.cache_size = self.cache_size
        param.max_iter = self.max_iter
        param.nochange_steps = self.nochange_steps
        param.tol = self.tol
        param.verbosity = self.verbose
        param.epsilon = self.epsilon
        param.svmType = lib.SvmType.C_SVC if is_classifier else lib.SvmType.EPSILON_SVR

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef int n_rows, n_cols, X_nnz
        n_rows, n_cols = X.shape
        cdef uintptr_t X_ptr, X_data_ptr, y_ptr, sample_weight_ptr
        cdef int *X_indptr
        cdef int *X_indices

        if is_sparse:
            X_nnz = X.nnz
            X_indptr = <int*><uintptr_t>X.indptr.ptr
            X_indices = <int*><uintptr_t>X.indices.ptr
            X_data_ptr = X.data.ptr
        else:
            X_ptr = X.ptr
        y_ptr = y.ptr
        sample_weight_ptr = 0 if sample_weight is None else sample_weight.ptr

        cdef _SVMModel internal = _SVMModel.new(self.handle, is_float32)

        with nogil:
            if is_sparse:
                if is_float32:
                    lib.svrFitSparse(
                        handle_[0],
                        X_indptr,
                        X_indices,
                        <float*>X_data_ptr,
                        n_rows,
                        n_cols,
                        X_nnz,
                        <float*>y_ptr,
                        param,
                        kernel_params,
                        internal.model_f[0],
                        <float*>sample_weight_ptr,
                    )
                else:
                    lib.svrFitSparse(
                        handle_[0],
                        X_indptr,
                        X_indices,
                        <double*>X_data_ptr,
                        n_rows,
                        n_cols,
                        X_nnz,
                        <double*>y_ptr,
                        param,
                        kernel_params,
                        internal.model_d[0],
                        <double*>sample_weight_ptr,
                    )
            else:
                if is_float32:
                    lib.svrFit(
                        handle_[0],
                        <float*> X_ptr,
                        n_rows,
                        n_cols,
                        <float*> y_ptr,
                        param,
                        kernel_params,
                        internal.model_f[0],
                        <float*>sample_weight_ptr,
                    )
                else:
                    lib.svrFit(
                        handle_[0],
                        <double*> X_ptr,
                        n_rows,
                        n_cols,
                        <double*> y_ptr,
                        param,
                        kernel_params,
                        internal.model_d[0],
                        <double*>sample_weight_ptr,
                    )
        self.handle.sync()

        support, support_vectors, dual_coef, intercept = internal.unpack()

        self.support_ = support
        self.support_vectors_ = support_vectors
        self.dual_coef_ = dual_coef
        self.intercept_ = intercept
        self.n_support_ = support.shape[0]
        self.fit_status_ = 0
        self.shape_fit_ = X.shape
        self._gamma = gamma
        self._sparse = is_sparse
        self._probA = np.empty(0, dtype=np.float64)
        self._probB = np.empty(0, dtype=np.float64)

    def _predict(self, X) -> CumlArray:
        """Perform `predict`.

        Expects X to already be validated and normalized to match fit dtype.
        """
        # Sanity checks, caller is responsible for casting dtype earlier
        assert self.support_vectors_.dtype == X.dtype
        assert self.dual_coef_.dtype == X.dtype
        assert self.intercept_.dtype == X.dtype
        assert X.shape[1] == self.support_vectors_.shape[1]

        support_vectors = self.support_vectors_

        cdef bool sparse_model = isinstance(support_vectors, SparseCumlArray)
        cdef bool is_sparse = isinstance(X, SparseCumlArray)
        cdef bool is_float32 = X.dtype == np.float32

        # Extract support_vectors_
        cdef int support_nnz = support_vectors.nnz if sparse_model else -1
        cdef int* support_indptr = <int*><uintptr_t>(
            support_vectors.indptr.ptr if sparse_model else 0
        )
        cdef int* support_indices = <int*><uintptr_t>(
            support_vectors.indices.ptr if sparse_model else 0
        )
        cdef uintptr_t support_data_ptr = (
            support_vectors.data.ptr if sparse_model else support_vectors.ptr
        )

        # Setup SvmModel of proper type
        cdef lib.SvmModel[float] model_f
        cdef lib.SvmModel[double] model_d
        if is_float32:
            model_f.n_support = self.support_.shape[0]
            model_f.n_cols = support_vectors.shape[1]
            model_f.b = self.intercept_.item()
            model_f.dual_coefs = <float*><uintptr_t>self.dual_coef_.ptr
            model_f.support_idx = <int*><uintptr_t>self.support_.ptr
            model_f.support_matrix.nnz = support_nnz
            model_f.support_matrix.indptr = support_indptr
            model_f.support_matrix.indices = support_indices
            model_f.support_matrix.data = <float*>support_data_ptr
        else:
            model_d.n_support = self.support_.shape[0]
            model_d.n_cols = support_vectors.shape[1]
            model_d.b = self.intercept_.item()
            model_d.dual_coefs = <double*><uintptr_t>self.dual_coef_.ptr
            model_d.support_idx = <int*><uintptr_t>self.support_.ptr
            model_d.support_matrix.nnz = support_nnz
            model_d.support_matrix.indptr = support_indptr
            model_d.support_matrix.indices = support_indices
            model_d.support_matrix.data = <double*>support_data_ptr

        # Setup KernelParams
        cdef lib.KernelParams kernel_params
        kernel_params.kernel = self._get_kernel_type()
        kernel_params.gamma = self._gamma
        kernel_params.coef0 = self.coef0
        kernel_params.degree = self.degree

        # Extract components from X
        cdef int *X_indptr
        cdef int *X_indices
        cdef uintptr_t X_data_ptr
        cdef int X_rows, X_cols, X_nnz
        X_rows, X_cols = X.shape
        if is_sparse:
            X_indptr = <int*><uintptr_t>X.indptr.ptr
            X_indices = <int*><uintptr_t>X.indices.ptr
            X_data_ptr = X.data.ptr
            X_nnz = X.nnz
        else:
            X_data_ptr = X.ptr

        # Allocate output array
        out = CumlArray.zeros(X_rows, dtype=X.dtype, index=X.index)
        cdef uintptr_t out_ptr = out.ptr

        cdef double cache_size = self.cache_size
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        # Call predict
        with nogil:
            if is_sparse:
                if is_float32:
                    lib.svcPredictSparse(
                        handle_[0],
                        X_indptr,
                        X_indices,
                        <float*>X_data_ptr,
                        X_rows,
                        X_cols,
                        X_nnz,
                        kernel_params,
                        model_f,
                        <float*>out_ptr,
                        <float>cache_size,
                        False,
                    )
                else:
                    lib.svcPredictSparse(
                        handle_[0],
                        X_indptr,
                        X_indices,
                        <double*>X_data_ptr,
                        X_rows,
                        X_cols,
                        X_nnz,
                        kernel_params,
                        model_d,
                        <double*>out_ptr,
                        <double>cache_size,
                        False,
                    )
            else:
                if is_float32:
                    lib.svcPredict(
                        handle_[0],
                        <float*>X_data_ptr,
                        X_rows,
                        X_cols,
                        kernel_params,
                        model_f,
                        <float*>out_ptr,
                        <float>cache_size,
                        False,
                    )
                else:
                    lib.svcPredict(
                        handle_[0],
                        <double*>X_data_ptr,
                        X_rows,
                        X_cols,
                        kernel_params,
                        model_d,
                        <double*>out_ptr,
                        <double>cache_size,
                        False,
                    )
        self.handle.sync()

        return out
