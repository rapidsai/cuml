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

import cuml
import cupy as cp
import numpy as np

from cudf import DataFrame as cu_df
from cuml.common.array import CumlArray
from cuml.common.input_utils import input_to_cuml_array
from cuml.common.logger import warn
from cuml.common.logger import info
from numba import cuda
from pandas import DataFrame as pd_df

from cuml.raft.common.handle cimport handle_t
from libcpp cimport bool
from libc.stdint cimport uintptr_t


try:
    from shap.utils import partition_tree_shuffle
    shap_not_found = False
except ImportError:
    shap_not_found = True


cdef extern from "cuml/datasets/make_permutation.hpp" namespace "ML":

    void make_permutation "ML::Datasets::make_permutation"(
        handle_t& handle,
        float* out,
        float* background,
        int n_rows,
        int n_cols,
        float* row,
        int* idx,
        bool rowMajor) except +

    void make_permutation "ML::Datasets::make_permutation"(
        handle_t& handle,
        double* out,
        double* background,
        int n_rows,
        int n_cols,
        double* row,
        int* idx,
        bool rowMajor) except +

    void single_entry_scatter "ML::Datasets::single_entry_scatter"(
        handle_t& handle,
        float* out,
        float* background,
        int n_rows,
        int n_cols,
        float* row,
        int* idx,
        bool rowMajor) except +

    void single_entry_scatter "ML::Datasets::single_entry_scatter"(
        handle_t& handle,
        double* out,
        double* background,
        int n_rows,
        int n_cols,
        double* row,
        int* idx,
        bool rowMajor) except +


def identity(x):
    return x


def _identity_inverse(x):
    return x


def logit(x):
    return cp.log(x / (1 - x))


def _logit_inverse(x):
    return 1 / (1 + cp.exp(-x))


identity.inverse = _identity_inverse
logit.inverse = _logit_inverse


# kernels to do a fast calculation of the values for each row
# depending on a shuffled index
_row_values_kernel_f32 = cp.RawKernel(r'''
    extern "C" __global__
    void _row_values(float* row_values, float* averaged_outs,
                     int* idx, int n) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if(tid < n){
            int pos = idx[tid];
            row_values[pos] +=
            2 * (averaged_outs[tid + 1] - averaged_outs[tid]);
        }
    }
    ''', '_row_values')

_row_values_kernel_f64 = cp.RawKernel(r'''
    extern "C" __global__
    void _row_values(double* row_values, double* averaged_outs,
                     int* idx, int n) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if(tid < n){
            int pos = idx[tid];
            row_values[pos] +=
            2 * (averaged_outs[tid + 1] - averaged_outs[tid]);
        }
    }
    ''', '_row_values')


class PermutationSHAP():
    """

    Initial experimental version of a GPU accelerated of SHAP's
    permutation explainer:
    https://github.com/slundberg/shap/blob/master/shap/explainers/_permutation.py

    This method approximates the Shapley values by iterating through
    permutations of the inputs. Quoting the SHAP library docs, it guarantees
    local accuracy (additivity) by iterating completely through  entire
    permutations of the features in both forward and reverse directions.

    Current limitations of the GPU version (support in progress):

    - Batched, both for supporting larger daasets as well as to accelerate
    smaller ones, is not implemented yet.
    - Only tabular masker is supported, via passing the background
    dataset explicitly. Since the new API of SHAP is still evolving, the main
    supported API for this version is the old one
    (i.e. explainer.shap_values())
    - Hierarchical clustering for Owen values are not GPU accelerated
    - Distributed version is in progress.
    - Sparse data support is in progress.
    - Some optimizations, like reducing number of model evaluations, are not
    part of the code yet.

    Parameters
    ----------
    model : function
        A callable python object that executes the model given a set of input
        data samples.
    masker : Dense matrix containing floats or doubles.
        cuML's permutation SHAP supports tabular data for now, so it expects
        a background dataset, as opposed to a shap.masker object. To respect
        a hierarchical structure of the data, use the (temporary) parameter
        'masker_type'
        Acceptable formats: CUDA array interface compliant objects like
        CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas
        DataFrame/Series.
    masker_type: {'independent', 'partition'} default = 'independent'
        If 'independent' is used, then this is equivalent to SHAP's
        independent masker and the algorithm is fully GPU accelerated.
        If 'partition' then it is equivalent to SHAP's Partition masker,
        which respects a hierarchical structure in the background data.
    link : function
        The link function used to map between the output units of the
        model and the SHAP value units.



    """

    def __init__(self,
                 model,
                 masker,
                 masker_type='independent',
                 link=identity,
                 output_names=None,
                 handle=None):

        self.handle = cuml.raft.common.handle.Handle() if handle is None \
            else handle

        self.model = model
        self.output_names = output_names

        # validate and save the link function
        if callable(link) and callable(getattr(link, "inverse", None)):
            self.link = link
        else:
            raise TypeError("`link` function is not valid.")

        self.masker, self.n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(masker, order='F')
        self.link = link

        if isinstance(masker, pd_df) or isinstance(masker, cu_df):
            self.feature_names = masker.columns.to_list()
        else:
            self.feature_names = [None for _ in range(len(masker))]

    def explain(self,
                *args,
                max_evals="auto",
                main_effects=False,
                error_bounds=False,
                batch_evals=False,
                verbose=False):

        warn("Support of the new API is experimetnal and depends upon "
             "changes in progress in the SHAP package. ")
        idx = 0
        X = cp.zeros((len(args), self.n_cols), dtype=self.dtype)
        for arg in args[0]:
            X[idx] = self.explain_row(row=arg,
                                      max_evals=max_evals,
                                      main_effects=main_effects,
                                      verbose=verbose)

        # todo: support shap.Explanation object (which is still experimental)
        # https://github.com/slundberg/shap/blob/3eee9448bcf86e76b14fe246bf518bcfa9a7bedc/shap/_explanation.py#L64  # noqa

        return X

    def explain_row(self,
                    row,
                    max_evals,
                    main_effects,
                    verbose,
                    order="F"):

        self.masker, *_ = input_to_cuml_array(self.masker, order=order)

        row = row.reshape(1, self.n_cols)
        row, n_rows, n_cols, dtype = \
            input_to_cuml_array(row, order=order)

        idx = cp.arange(n_cols, dtype=cp.int32)

        if max_evals == "auto":
            max_evals = 10 * 2 * self.n_cols

        npermutations = max_evals // (2 * len(idx) + 1)

        row_values = cp.zeros(n_cols, dtype=dtype)

        masked_inputs = cp.empty(
            shape=((2 * n_cols * self.n_rows + self.n_rows), n_cols),
            dtype=dtype,
            order=order
        )

        cdef handle_t* handle_ = \
            <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t row_ptr, bg_ptr, idx_ptr, masked_ptr

        for _ in range(npermutations):

            cp.random.shuffle(idx)

            # CuPy code for reference for generating permutations
            # Will be removed in 0.17 for final release of cuML/SHAP
            # masked_inputs = \
            #     cp.tile(masker.flatten(), 2 * n_cols + 1).reshape(
            #         ((2 * n_cols + 1) * len(masker), n_cols)
            #     )

            # for idx in range(len(idx)):
            #     masked_inputs[
            #         (idx + 1) * len(bg):(idx + 2) * (n_cols * len(bg)),
            #         idx[idx]] = row[idx[idx]]

            masked_ptr = masked_inputs.__cuda_array_interface__['data'][0]
            bg_ptr = self.masker.ptr
            row_ptr = row.ptr
            idx_ptr = idx.__cuda_array_interface__['data'][0]
            row_major = order == "C"

            if dtype == cp.float32:
                make_permutation(handle_[0],
                                 <float*> masked_ptr,
                                 <float*> bg_ptr,
                                 <int> self.n_rows,
                                 <int> n_cols,
                                 <float*> row_ptr,
                                 <int*> idx_ptr,
                                 <bool> row_major)
            else:
                make_permutation(handle_[0],
                                 <double*> masked_ptr,
                                 <double*> bg_ptr,
                                 <int> self.n_rows,
                                 <int> n_cols,
                                 <double*> row_ptr,
                                 <int*> idx_ptr,
                                 <bool> row_major)

            self.handle.sync()

            results = self.model(masked_inputs)

            results = results.reshape(2 * n_cols + 1, len(self.masker))

            averaged_outs = cp.mean(cp.asarray(self.link(results)), axis=1)

            tpb = 256
            blks = int((len(idx) + tpb - 1) / tpb)

            if dtype == cp.float32:
                _row_values_kernel_f32((blks,), (tpb,), (row_values,
                                                         averaged_outs,
                                                         idx,
                                                         len(idx)))
            else:
                _row_values_kernel_f64((blks,), (tpb,), (row_values,
                                                         averaged_outs,
                                                         idx,
                                                         len(idx)))

            cp.cuda.Stream.null.synchronize()

        row_values /= (2 * npermutations)
        self.expected_value = averaged_outs[0]
        self.obs_output = results[n_cols]

        if main_effects:
            del masked_inputs
            main_effect_values = self.main_effects(row)
        else:
            main_effect_values = None

        return row_values

    def main_effects(self, row, inds=None):
        if inds is None:
            inds = cp.arange(len(self.masker), dtype=np.float32)

        masked_inputs = cp.empty(
            shape=((self.n_rows * self.n_cols + self.n_rows), self.n_cols),
            dtype=self.dtype,
            order=self.masker.order
        )

        cdef handle_t* handle_ = \
            <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t row_ptr, bg_ptr, idx_ptr, masked_ptr

        masked_ptr = masked_inputs.__cuda_array_interface__['data'][0]
        bg_ptr = self.masker.ptr
        row_ptr = row.ptr
        idx_ptr = inds.__cuda_array_interface__['data'][0]
        row_major = self.masker.order == "C"

        if self.masker.order.dtype == cp.float32:
            single_entry_scatter(handle_[0],
                                 <float*> masked_ptr,
                                 <float*> bg_ptr,
                                 <int> self.n_rows,
                                 <int> self.n_cols,
                                 <float*> row_ptr,
                                 <int*> idx_ptr,
                                 <bool> row_major)
        else:
            single_entry_scatter(handle_[0],
                                 <double*> masked_ptr,
                                 <double*> bg_ptr,
                                 <int> self.n_rows,
                                 <int> self.n_cols,
                                 <double*> row_ptr,
                                 <int*> idx_ptr,
                                 <bool> row_major)

        self.handle.sync()

        outputs = self.model(masked_inputs)
        main_effects = outputs[1:] - outputs[0]
        return main_effects

    def shap_values(self,
                    X,
                    npermutations=10,
                    main_effects=False,
                    verbose=False):
        """
        Legacy interface to estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : Dense matrix containing floats or doubles.
            Acceptable formats: CUDA array interface compliant objects like
            CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas
            DataFrame/Series.
        npermutations : int
            Number of times to cycle through all the features, re-evaluating
            the model at each step. Each cycle evaluates the model function
            on a matrix of size
            (2 * n_features * n_background + n_background), n_features), where
            n_features is the number of rows of the background dataset.

        Returns
        -------
        array or list

        """

        results = self.explain(X, max_evals=npermutations * X.shape[1],
                               main_effects=main_effects)
        return results
