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
from cuml.experimental.explainer.common import get_dtype_from_model_func
from cuml.experimental.explainer.common import get_link_fn_from_str
from cuml.experimental.explainer.common import get_tag_from_model_func
from cuml.experimental.explainer.common import model_call
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


cdef extern from "cuml/explainer/permutation_shap.hpp" namespace "ML":

    void permutation_dataset "ML::Explainer::permutation_dataset"(
        handle_t& handle,
        float* out,
        float* background,
        int n_rows,
        int n_cols,
        float* row,
        int* idx,
        bool rowMajor) except +

    void permutation_dataset "ML::Explainer::permutation_dataset"(
        handle_t& handle,
        double* out,
        double* background,
        int n_rows,
        int n_cols,
        double* row,
        int* idx,
        bool rowMajor) except +

    void main_effect_dataset "ML::Explainer::main_effect_dataset"(
        handle_t& handle,
        float* out,
        float* background,
        int n_rows,
        int n_cols,
        float* row,
        int* idx,
        bool rowMajor) except +

    void main_effect_dataset "ML::Explainer::main_effect_dataset"(
        handle_t& handle,
        double* out,
        double* background,
        int n_rows,
        int n_cols,
        double* row,
        int* idx,
        bool rowMajor) except +


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
    - Sparse data support is not yet implemented.
    - Some optimizations are not yet implemented.

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
                 link='identity',
                 output_names=None,
                 handle=None,
                 gpu_model=None,
                 random_state=None):

        self.handle = cuml.raft.common.handle.Handle() if handle is None \
            else handle

        self.model = model
        self.output_names = output_names
        self.link = link
        self.link_fn = get_link_fn_from_str(link)

        self.random_state = random_state

        self.order = get_tag_from_model_func(func=model,
                                             tag='preferred_input_order',
                                             default='F')

        if gpu_model is None:
            # todo: when sparse support is added, use this tag to see if
            # model can accept sparse data
            self.model_gpu_based = \
                get_tag_from_model_func(func=model,
                                        tag='X_types_gpu',
                                        default=False) is not None
        else:
            self.model_gpu_based = gpu_model

        self.masker, self.n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(masker, order=self.order)
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

        idx = 0
        X = cp.zeros((len(args), self.n_cols), dtype=self.dtype)

        if main_effects:
            main_effects_res = cp.zeros((len(args), self.n_cols),
                                        dtype=self.dtype)

        for arg in args[0]:
            if main_effects:
                X[idx], main_effects[idx] = self.explain_row(
                    row=arg,
                    max_evals=max_evals,
                    main_effects=main_effects,
                    verbose=verbose
                )
            else:
                X[idx], _ = self.explain_row(row=arg,
                                             max_evals=max_evals,
                                             main_effects=main_effects,
                                             verbose=verbose)
            idx += 1

        return X, main_effects

    def explain_row(self,
                    row,
                    max_evals,
                    main_effects,
                    verbose):

        self.masker, *_ = input_to_cuml_array(self.masker, order=self.order)

        row = row.reshape(1, self.n_cols)
        row, n_rows, n_cols, dtype = \
            input_to_cuml_array(row, order=self.order)

        idx = cp.arange(n_cols, dtype=cp.int32)

        if max_evals == "auto":
            max_evals = 10 * 2 * self.n_cols

        npermutations = max_evals // (2 * len(idx) + 1)

        row_values = cp.zeros(n_cols, dtype=dtype)

        masked_inputs = cp.empty(
            shape=((2 * n_cols * self.n_rows + self.n_rows), n_cols),
            dtype=dtype,
            order=self.order
        )

        cdef handle_t* handle_ = \
            <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t row_ptr, bg_ptr, idx_ptr, masked_ptr

        if self.random_state is not None:
            cupy.random.seed(seed=self.random_state)

        for _ in range(npermutations):

            cp.random.shuffle(idx)

            masked_ptr = masked_inputs.__cuda_array_interface__['data'][0]
            bg_ptr = self.masker.ptr
            row_ptr = row.ptr
            idx_ptr = idx.__cuda_array_interface__['data'][0]
            row_major = self.order == "C"

            if dtype == cp.float32:
                permutation_dataset(handle_[0],
                                 <float*> masked_ptr,
                                 <float*> bg_ptr,
                                 <int> self.n_rows,
                                 <int> n_cols,
                                 <float*> row_ptr,
                                 <int*> idx_ptr,
                                 <bool> row_major)
            else:
                permutation_dataset(handle_[0],
                                 <double*> masked_ptr,
                                 <double*> bg_ptr,
                                 <int> self.n_rows,
                                 <int> n_cols,
                                 <double*> row_ptr,
                                 <int*> idx_ptr,
                                 <bool> row_major)

            self.handle.sync()

            results = model_call(X=masked_inputs,
                                 model=self.model,
                                 model_gpu_based=self.model_gpu_based)

            self.obs_output = results[len(results) / 2]

            results = results.reshape(2 * n_cols + 1, len(self.masker))

            averaged_outs = cp.mean(cp.asarray(self.link_fn(results)), axis=1)

            tpb = 512
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

        row_values /= (2 * npermutations)
        self.expected_value = averaged_outs[0]

        diff = cp.sum(row_values) - (self.obs_output - self.expected_value)

        if main_effects:
            del masked_inputs
            main_effect_values = self.main_effects(row)
        else:
            main_effect_values = None

        return row_values, main_effect_values

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
            main_effect_dataset(handle_[0],
                                 <float*> masked_ptr,
                                 <float*> bg_ptr,
                                 <int> self.n_rows,
                                 <int> self.n_cols,
                                 <float*> row_ptr,
                                 <int*> idx_ptr,
                                 <bool> row_major)
        else:
            main_effect_dataset(handle_[0],
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
        nrows = 1 if X.ndim == 1 else X.shape[1]

        results, main_effects = self.explain(
            X,
            max_evals=npermutations * nrows,
            main_effects=main_effects
        )

        # for compatibility with mainline SHAP api, we divide the return
        if main_effects:
            return resuls, main_effects
        else:
            return results

    def __call__(self,
                 X,
                 max_evals='auto',
                 main_effects=False):
        warn("SHAP's Explanation object is still experimental and depends "
             "on SHAP >= 0.36. The main API currently is "
             "'explainer.shap_values'.")
        res = self.explain(X,
                           l1_reg,
                           max_evals=max_evals)
        out = Explanation(
            values=res,
            base_values=self.expected_value,
            base_values=self.expected_value,
            data=self.background,
            feature_names=self.feature_names,
            main_effects=self.main_effects
        )
        return results
