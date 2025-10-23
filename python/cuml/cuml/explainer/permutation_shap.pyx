#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
import time

import cupy as cp
import numpy as np

from cuml.explainer.base import SHAPBase
from cuml.explainer.common import get_cai_ptr, model_func_call

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/explainer/permutation_shap.hpp" namespace "ML" nogil:

    void permutation_shap_dataset "ML::Explainer::permutation_shap_dataset"(
        const handle_t& handle,
        float* dataset,
        const float* background,
        int n_rows,
        int n_cols,
        const float* row,
        int* idx,
        bool rowMajor) except +

    void permutation_shap_dataset "ML::Explainer::permutation_shap_dataset"(
        const handle_t& handle,
        double* dataset,
        const double* background,
        int n_rows,
        int n_cols,
        const double* row,
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
    GPU accelerated version of SHAP's PermutationExplainer

    cuML's SHAP based explainers accelerate the algorithmic part of SHAP.
    They are optimized to be used with fast GPU based models, like those in
    cuML. By creating the datasets and internal calculations,
    alongside minimizing data copies and transfers, they can accelerate
    explanations significantly. But they can also be used with
    CPU based models, where speedups can still be achieved, but those can be
    capped by factors like data transfers and the speed of the models.

    PermutationExplainer is algorithmically similar and based on the Python
    SHAP package kernel explainer:
    https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py

    This method approximates the Shapley values by iterating through
    permutations of the inputs. From the SHAP library docs: it guarantees
    local accuracy (additivity) by iterating completely through  entire
    permutations of the features in both forward and reverse directions.

    Current characteristics of the GPU version:

     * Only tabular data is supported for now, via passing the background
       dataset explicitly.
     * Hierarchical clustering for Owen values are planned for the near
       future.
     * Sparse data support is planned for the near future.

    **Setting the random seed**:

    This explainer uses CuPy to generate the permutations that are used, so
    to have reproducible results use `CuPy's seeding mechanism
    <https://docs.cupy.dev/en/stable/reference/generated/cupy.random.seed.html>`_.

    Parameters
    ----------
    model : function
        A callable python object that executes the model given a set of input
        data samples.
    masker : Dense matrix containing floats or doubles.
        cuML's permutation SHAP supports tabular data for now, so it expects
        a background dataset, as opposed to a shap.masker object. To respect
        a hierarchical structure of the data, use the (temporary) parameter
        `masker_type`
        Acceptable formats: CUDA array interface compliant objects like
        CuPy, cuDF DataFrame/Series, NumPy ndarray and Pandas
        DataFrame/Series.
    masker_type: {'independent', 'partition'} default = 'independent'
        If 'independent' is used, then this is equivalent to SHAP's
        independent masker and the algorithm is fully GPU accelerated.
        If 'partition' then it is equivalent to SHAP's Partition masker,
        which respects a hierarchical structure in the background data.
    link : function or str (default = 'identity')
        The link function used to map between the output units of the
        model and the SHAP value units. From the SHAP package: The link
        function used to map between the output units of the model and the
        SHAP value units. By default it is identity, but logit can be useful
        so that expectations are computed in probability units while
        explanations remain in the (more naturally additive) log-odds units.
        For more details on how link functions work see any overview of link
        functions for generalized linear models.
    gpu_model : bool or None (default = None)
        If None Explainer will try to infer whether `model` can take GPU data
        (as CuPy arrays), otherwise it will use NumPy arrays to call `model`.
        Set to True to force the explainer to use GPU data,  set to False to
        force the Explainer to use NumPy data.
    handle : pylibraft.common.handle (default = None)
        Specifies the handle that holds internal CUDA state for
        computations in this model, a new one is created if it is None.
        Most importantly, this specifies the CUDA stream that will be used for
        the model's computations, so users can run different models
        concurrently in different streams by creating handles in several
        streams.
    dtype : np.float32 or np.float64 (default = np.float32)
        Parameter to specify the precision of data to generate to call the
        model.
    output_type : 'cupy' or 'numpy' (default = 'numpy')
        Parameter to specify the type of data to output.
        If not specified, the explainer will default to 'numpy' for the time
        being to improve compatibility.

    Examples
    --------

    .. code-block:: python

        >>> from cuml import SVR
        >>> from cuml import make_regression
        >>> from cuml import train_test_split

        >>> from cuml.explainer import PermutationExplainer

        >>> X, y = make_regression(
        ...     n_samples=102,
        ...     n_features=10,
        ...     noise=0.1,
        ...     random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X,
        ...     y,
        ...     test_size=2,
        ...     random_state=42)
        >>> model = SVR().fit(X_train, y_train)

        >>> cu_explainer = PermutationExplainer(
        ...     model=model.predict,
        ...     data=X_train,
        ...     random_state=42)

        >>> cu_shap_values = cu_explainer.shap_values(X_test)
        >>> cu_shap_values  # doctest: +SKIP
        array([[ 0.16611198, 0.74156773, 0.05906528,  0.30015892, 2.5425286 ,
                0.0970122 , 0.12258395, 2.1998262 , -0.02968234, -0.8669155 ],
            [-0.10587756,  0.77705824, -0.08259875, -0.71874434,  1.781551  ,
                -0.05454511, 0.11826539, -1.1734306 , -0.09629871, 0.4571011]],
            dtype=float32)

    """

    def __init__(self,
                 *,
                 model,
                 data,
                 masker_type='independent',
                 link='identity',
                 handle=None,
                 is_gpu_model=None,
                 random_state=None,
                 dtype=np.float32,
                 output_type=None,
                 verbose=False,):
        super().__init__(
            order='C',
            model=model,
            background=data,
            link=link,
            verbose=verbose,
            is_gpu_model=is_gpu_model,
            handle=handle,
            dtype=dtype,
            output_type=output_type
        )

    def shap_values(self,
                    X,
                    npermutations=10,
                    as_list=True,
                    **kwargs):
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
            Number of times to cycle through all the features, re-evaluating
            the model at each step. Each cycle evaluates the model function
            2 * (# features + 1) times on a data matrix of (# background
            data samples) rows. An exception to this is when
            PermutationExplainer can avoid evaluating the model because a
            feature's value is the same in X and the background dataset
            (which is common for example with sparse features).
        as_list : bool (default = True)
            Set to True to return a list of arrays for multi-dimensional
            models (like predict_proba functions) to match the SHAP package
            shap_values API behavior.
            Set to False to return them as an array of arrays.

        Returns
        -------
        shap_values : array or list

        """
        return self._explain(X,
                             synth_data_shape=(
                                 (2 * self.ncols * self.nrows + self.nrows),
                                 self.ncols
                             ),
                             npermutations=npermutations,
                             return_as_list=as_list,
                             **kwargs)

    def _explain_single_observation(self,
                                    shap_values,
                                    row,
                                    idx,
                                    npermutations=10,
                                    testing=False):
        total_timer = time.time()
        inds = cp.arange(self.ncols, dtype=cp.int32)

        cdef handle_t* handle_ = \
            <handle_t*><size_t>self.handle.getHandle()
        cdef uintptr_t row_ptr, bg_ptr, idx_ptr, ds_ptr, shap_ptr, y_hat_ptr

        if self.random_state is not None:
            cp.random.seed(seed=self.random_state)

        for _ in range(npermutations):

            if not testing:
                cp.random.shuffle(inds)

            ds_ptr = get_cai_ptr(self._synth_data)
            bg_ptr = get_cai_ptr(self.background)
            row_ptr = get_cai_ptr(row)
            idx_ptr = get_cai_ptr(inds)
            row_major = self.order == "C"

            if self.dtype == cp.float32:
                permutation_shap_dataset(handle_[0],
                                         <float*> ds_ptr,
                                         <float*> bg_ptr,
                                         <int> self.nrows,
                                         <int> self.ncols,
                                         <float*> row_ptr,
                                         <int*> idx_ptr,
                                         <bool> row_major)
            else:
                permutation_shap_dataset(handle_[0],
                                         <double*> ds_ptr,
                                         <double*> bg_ptr,
                                         <int> self.nrows,
                                         <int> self.ncols,
                                         <double*> row_ptr,
                                         <int*> idx_ptr,
                                         <bool> row_major)

            self.handle.sync()

            # evaluate model on combinations
            model_timer = time.time()
            y = model_func_call(X=self._synth_data,
                                model_func=self.model,
                                gpu_model=self.is_gpu_model)
            self.model_call_time = \
                self.model_call_time + (time.time() - model_timer)

            for i in range(self.model_dimensions):
                # reshape the results to coincide with each entry of the
                # permutation
                if self.model_dimensions == 1:
                    y_hat = y.reshape(2 * self.ncols + 1, len(self.background))

                else:
                    y_hat = y[:, i].reshape(2 * self.ncols + 1,
                                            len(self.background))

                # we get the average of each entry
                y_hat = cp.mean(cp.asarray(self.link_fn(y_hat)),
                                axis=1).astype(self.dtype)

                shap_ptr = get_cai_ptr(shap_values[i][idx])
                y_hat_ptr = get_cai_ptr(y_hat)

                if self.dtype == cp.float32:
                    update_perm_shap_values(handle_[0],
                                            <float*> shap_ptr,
                                            <float*> y_hat_ptr,
                                            <int> self.ncols,
                                            <int*> idx_ptr)
                else:
                    update_perm_shap_values(handle_[0],
                                            <double*> shap_ptr,
                                            <double*> y_hat_ptr,
                                            <int> self.ncols,
                                            <int*> idx_ptr)

                self.handle.sync()

        for i in range(self.model_dimensions):
            shap_values[i][idx] = shap_values[i][idx] / (2 * npermutations)

        self.total_time = self.total_time + (time.time() - total_timer)
