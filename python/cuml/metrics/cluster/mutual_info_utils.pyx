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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numpy as np
import cupy as cp

from cuml.utils import with_cupy_rmm, input_to_cuml_array


@with_cupy_rmm
def prepare_data(labels_true, labels_pred):
    """Helper function to avoid code duplication for homogeneity score, mutual
    info score and completeness score.
    """
    preds_m, n_rows, _, _ = input_to_cuml_array(
        labels_pred,
        check_dtype=np.int32,
        check_cols=1
    )

    ground_truth_m, _, _, _ = input_to_cuml_array(
        labels_true,
        check_dtype=np.int32,
        check_rows=n_rows,
        check_cols=1
    )

    cp_ground_truth_m = ground_truth_m.to_output(output_type='cupy')
    cp_preds_m = preds_m.to_output(output_type='cupy')

    lower_class_range = min(cp.min(cp_ground_truth_m),
                            cp.min(cp_preds_m))
    upper_class_range = max(cp.max(cp_ground_truth_m),
                            cp.max(cp_preds_m))

    return (ground_truth_m, preds_m,
            n_rows,
            lower_class_range, upper_class_range)
