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
from cuml.common.import_utils import has_shap
from cuml.common.input_utils import input_to_cupy_array
from cuml.common.logger import warn
from cuml.common.logger import info
from cuml.experimental.explainer.base import SHAPBase
from cuml.experimental.explainer.common import get_cai_ptr
from cuml.experimental.explainer.common import get_dtype_from_model_func
from cuml.experimental.explainer.common import get_tag_from_model_func
from cuml.experimental.explainer.common import model_func_call
from cuml.experimental.explainer.common import output_list_shap_values
from numba import cuda
from pandas import DataFrame as pd_df

from cuml.raft.common.handle cimport handle_t
from libcpp cimport bool
from libc.stdint cimport uintptr_t


cdef extern from "cuml/explainer/permutation_shap.hpp" namespace "ML":

    void permutation_shap_dataset "ML::Explainer::permutation_shap_dataset"(
        const handle_t& handle,
        float* out,
        const float* background,
        int n_rows,
        int n_cols,
        const float* row,
        int* idx,
        bool rowMajor) except +

    void permutation_shap_dataset "ML::Explainer::permutation_shap_dataset"(
        const handle_t& handle,
        double* out,
        const double* background,
        int n_rows,
        int n_cols,
        const double* row,
        int* idx,
        bool rowMajor) except +

    void shap_main_effect_dataset "ML::Explainer::shap_main_effect_dataset"(
        const handle_t& handle,
        float* out,
        const float* background,
        int n_rows,
        int n_cols,
        const float* row,
        int* idx,
        bool rowMajor) except +

    void update_perm_shap_values "ML::Explainer::update_perm_shap_values"(
        const handle_t& handle,
        float* shap_values,
        const float* y_hat,
        const int ncols,
        const int* idx) except +

    void update_perm_shap_values "ML::Explainer::update_perm_shap_values"(
        const handle_t& handle,
        double* shap_values,
        const double* y_hat,
        const int ncols,
        const int* idx) except +


class PermutationExplainer(SHAPBase):
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
    dataset explicitly. Since the new API of SHAP is still evolving, the
    supported API for this version is the old one
    (i.e. explainer.shap_values()). The new one, and the new SHAP Explanation
    object will be supported in the next version.
    - Hierarchical clustering for Owen values is not GPU accelerated
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
                 handle=None,
                 gpu_model=None,
                 random_state=None,
                 dtype=None,
                 output_type=None,
                 verbose=False,):
        super(PermutationExplainer, self).__init__(
            order='C',
            model=model,
            background=masker,
            link=link,
            verbose=verbose,
            random_state=random_state,
            gpu_model=gpu_model,
            handle=handle,
            dtype=dtype,
            output_type=output_type
        )

        self._synth_data = None

    def shap_values(self,
                    X,
                    npermutations=10,
                    main_effects=False):
        """
        Interface to estimate the SHAP values for a set of samples.
        Corresponds to the SHAP package's legacy interface, and is our main
        API currently.

        Parameters
        ----------
        X : Dense matrix containing floats or doubles.
            Acceptable formats: CUDA array interface compliant objects like
            CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas
            DataFrame/Series.
        npermutations : int (default = 10)
            The l1 regularization to use for feature selection.

        Returns
        -------
        array or list

        """
        return self._explain(X,
                             npermutations=npermutations,
                             main_effects=main_effects)

    def _explain(self,
                 X,
                 npermutations=None,
                 main_effects=False,
                 testing=False):

        X = input_to_cupy_array(X, order=self.order,
                                convert_to_dtype=self.dtype)[0]

        if X.ndim == 1:
            X = X.reshape((1, self.M))

        shap_values = []
        for i in range(self.D):
            shap_values.append(cp.zeros(X.shape, dtype=self.dtype))

        # Allocate synthetic dataset array once for multiple explanations
        if self._synth_data is None:
            self._synth_data = cp.zeros(
                shape=((2 * self.M * self.N + self.N), self.M),
                dtype=self.dtype,
                order=self.order
            )

        idx = 0
        for x in X:
            # use mutability of lists and cupy arrays to get all shap values
            self._explain_single_observation(
                shap_values,
                x.reshape(1, self.M),
                main_effects=main_effects,
                npermutations=npermutations,
                idx=idx,
                testing=testing
            )
            idx = idx + 1

        return output_list_shap_values(shap_values, self.D, self.output_type)

    def _explain_single_observation(self,
                                    shap_values,
                                    row,
                                    main_effects,
                                    npermutations,
                                    idx,
                                    testing):

        inds = cp.arange(self.M, dtype=cp.int32)

        cdef handle_t* handle_ = \
            <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t row_ptr, bg_ptr, idx_ptr, ds_ptr, shap_ptr, y_hat_ptr

        if self.random_state is not None:
            cp.random.seed(seed=self.random_state)

        for _ in range(npermutations):

            if not testing:
                cp.random.shuffle(inds)
            # inds = cp.asarray(inds)
            # inds = cp.arange(self.M - 1, -1, -1).astype(cp.int32)
            ds_ptr = get_cai_ptr(self._synth_data)
            bg_ptr = get_cai_ptr(self.background)
            row_ptr = get_cai_ptr(row)
            idx_ptr = get_cai_ptr(inds)
            row_major = self.order == "C"

            if self.dtype == cp.float32:
                permutation_shap_dataset(handle_[0],
                                         <float*> ds_ptr,
                                         <float*> bg_ptr,
                                         <int> self.N,
                                         <int> self.M,
                                         <float*> row_ptr,
                                         <int*> idx_ptr,
                                         <bool> row_major)
            else:
                permutation_shap_dataset(handle_[0],
                                         <double*> ds_ptr,
                                         <double*> bg_ptr,
                                         <int> self.N,
                                         <int> self.M,
                                         <double*> row_ptr,
                                         <int*> idx_ptr,
                                         <bool> row_major)

            self.handle.sync()

            # evaluate model on combinations
            y = model_func_call(X=self._synth_data,
                                model_func=self.model,
                                gpu_model=self.gpu_model)

            for i in range(self.D):
                # reshape the results to coincide with each entry of the
                # permutation
                if self.D == 1:
                    y_hat = y.reshape(2 * self.M + 1, len(self.background))


                else:
                    y_hat = y[:, i].reshape(2 * self.M + 1,
                                            len(self.background))

                # we get the average of each entry
                y_hat = cp.mean(cp.asarray(self.link_fn(y_hat)),
                                axis=1).astype(self.dtype)

                shap_ptr = get_cai_ptr(shap_values[i][idx])
                y_hat_ptr = get_cai_ptr(y_hat)

                # aggregation of results calculation matches mainline SHAP
                if self.dtype == cp.float32:
                    update_perm_shap_values(handle_[0],
                                            <float*> shap_ptr,
                                            <float*> y_hat_ptr,
                                            <int> self.M,
                                            <int*> idx_ptr)
                else:
                    update_perm_shap_values(handle_[0],
                                            <double*> shap_ptr,
                                            <double*> y_hat_ptr,
                                            <int> self.M,
                                            <int*> idx_ptr)

                self.handle.sync()

        shap_values[0][idx] = shap_values[0][idx] / (2 * npermutations)
