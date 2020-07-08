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

import cupy as cp
import numpy as np
from cuml.common.memory_utils import with_cupy_rmm
from cuml.common import input_to_cuml_array
import math

@with_cupy_rmm
def precision_recall_curve(y_true, y_score):

    y_true, n_rows, n_cols, ytype = \
        input_to_cuml_array(y_true, check_dtype=[np.int32, np.int64,
                                                 np.float32, np.float64])

    y_score, _, _, _ = \
        input_to_cuml_array(y_score, check_dtype=[np.int32, np.int64,
                                                  np.float32, np.float64],
                            check_rows=n_rows, check_cols=n_cols)

    y_true = y_true.to_output('cupy')
    y_score = y_score.to_output('cupy')

    if cp.min(y_true) == 0 and cp.max(y_true) == 0:
        raise ValueError("precision_recall_curve cannot be used when "
                         "y_true is all zero.")

    if y_true.dtype.kind == 'f' and np.any(y_true != y_true.astype(int)):
        raise ValueError("Continuous format of y_true  "
                         "is not supported by roc_auc_score")

    ids = cp.argsort(-y_score) 
    sorted_score = y_score[ids]
    
    tps = y_true[ids].astype('float32') # for calculating true positives 
    pps = cp.ones_like(y_true).astype('float32') # for calculating predicted positives
    
    # calculate groups
    group = _group_same_scores(sorted_score)    
    num = int(group[-1])

    r = cp.zeros(num, dtype='float32') 
    p = cp.zeros(num, dtype='float32') 

    r = _addup_x_in_group(group, tps, r)    
    p = _addup_x_in_group(group, pps, p)  

    sum_one = cp.sum(y_true)
    r = cp.cumsum(r)
    
    precision = cp.flip(r/cp.cumsum(p), axis=0)
    recall = cp.flip(r/sum_one,axis=0)
    n = (recall==1).sum()
    thresholds = cp.unique(y_score)

    if n>1:
        precision = precision[n-1:]
        recall = recall[n-1:]
        thresholds = thresholds[n-1:]
    precision = cp.concatenate([precision,cp.ones(1)])
    recall = cp.concatenate([recall,cp.zeros(1)])
    
    return precision,recall,thresholds

@with_cupy_rmm
def roc_auc_score(y_true, y_score):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    from prediction scores.
    Note: this implementation can only be used with binary classification.
    Parameters
    ----------
        y_true : array-like of shape (n_samples,)
        True labels. The binary cases
        expect labels with shape (n_samples,)

    y_score : array-like of shape (n_samples,)
        Target scores. In the binary cases, these can be either
        probability estimates or non-thresholded decision values (as returned
        by `decision_function` on some classifiers). The binary
        case expects a shape (n_samples,), and the scores must be the scores of
        the class with the greater label.
    Returns
    -------
        auc : float

    Examples
    --------
    >>> import numpy as np
    >>> from cuml.metrics import roc_auc_score
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> roc_auc_score(y_true, y_scores)
    0.75
    """

    y_true, n_rows, n_cols, ytype = \
        input_to_cuml_array(y_true, check_dtype=[np.int32, np.int64,
                                                 np.float32, np.float64])

    y_score, _, _, _ = \
        input_to_cuml_array(y_score, check_dtype=[np.int32, np.int64,
                                                  np.float32, np.float64],
                            check_rows=n_rows, check_cols=n_cols)

    return _binary_roc_auc_score(y_true, y_score)


def _binary_roc_auc_score(y_true, y_score):
    """Compute binary roc_auc_score using cupy"""
    y_true = y_true.to_output('cupy')
    y_score = y_score.to_output('cupy')

    if cp.unique(y_true).shape[0] == 1:
        raise ValueError("roc_auc_score cannot be used when "
                         "only one class present in y_true. ROC AUC score "
                         "is not defined in that case.")

    if y_true.dtype.kind == 'f' and np.any(y_true != y_true.astype(int)):
        raise ValueError("Continuous format of y_true  "
                         "is not supported by roc_auc_score")

    if cp.unique(y_score).shape[0] == 1:
        return 0.5

    y_true = y_true.astype('float32')
    ids = cp.argsort(-y_score)  # we want descedning order

    sorted_score = y_score[ids]
    ones = y_true[ids]
    zeros = 1 - ones

    group = _group_same_scores(sorted_score)

    sum_ones = cp.sum(ones)
    sum_zeros = cp.sum(zeros)

    num = int(group[-1])

    tps = cp.zeros(num, dtype='float32')  # true positives
    fps = cp.zeros(num, dtype='float32')  # false positives

    tps = _addup_x_in_group(group, ones, tps)
    fps = _addup_x_in_group(group, zeros, fps)

    tpr = cp.cumsum(tps)/sum_ones
    fpr = cp.cumsum(fps)/sum_zeros

    return _calculate_area_under_curve(fpr, tpr).item()

def _addup_x_in_group(group, x, result):
    addup_x_in_group_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void addup_x_in_group(const int* group, const float* x,
            float* result, int N)
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid<N){
                atomicAdd(result + group[tid] - 1, x[tid]);
            }
        }
    ''', 'addup_x_in_group')

    N = x.shape[0]
    tpb = 256
    bpg = math.ceil(N/tpb)
    addup_x_in_group_kernel((bpg,), (tpb,),
                          (group, x, result, N))  # grid, block and arguments
    return result

def _group_same_scores(sorted_score):
    mask = cp.empty(sorted_score.shape, dtype=cp.bool_)
    mask[0] = True
    mask[1:] = sorted_score[1:] != sorted_score[:-1]
    group = cp.cumsum(mask, dtype=cp.int32)
    return group

def _calculate_area_under_curve(fpr, tpr):
    """helper function to calculate area under curve given fpr & tpr arrays"""
    return cp.sum((fpr[1:]-fpr[:-1])*(tpr[1:]+tpr[:-1]))/2 + tpr[0]*fpr[0]/2
