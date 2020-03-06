import cupy as cp
import numpy as np
from cuml.utils.memory_utils import with_cupy_rmm
from cuml.utils import input_to_cuml_array
import math

@with_cupy_rmm
def roc_auc_score(y_true, y_score):
    
    y_true, n_rows, n_cols, ytype = \
        input_to_cuml_array(y_true, check_dtype=[np.int32, np.int64, np.float32, np.float64])

    y_score, _, _, _ = \
        input_to_cuml_array(y_score, check_dtype=[np.int32, np.int64, np.float32, np.float64],
                            check_rows=n_rows, check_cols=n_cols)
    
    return _binary_roc_score(y_true, y_score)

def _binary_roc_score(y_true, y_score):    
    
    y_true = y_true.to_output()
    y_score = y_score.to_output()
    
    if cp.unique(y_true).shape[0]==1:
        raise ValueError("roc_auc_score cannot be used when "
               "only one class present in y_true. ROC AUC score "
                              "is not defined in that case.") 
        
    if y_true.dtype.kind == 'f' and np.any(y_true != y_true.astype(int)):
        raise ValueError("Continuous format of y_true  "
               "is not supported by roc_auc_score")
        
    if cp.unique(y_score).shape[0]==1:
        return 0.5
  
    y_true = y_true.astype('float32')
    ids = cp.argsort(-y_score) # we want descedning order
  
    
    sorted_score = y_score[ids]
    ones = y_true[ids]
    zeros = 1 - ones

    mask = cp.empty(sorted_score.shape, dtype=cp.bool_)
    mask[0] = True
    mask[1:] = sorted_score[1:] != sorted_score[:-1]
  
    mask = mask.astype('int32')
    group = cp.cumsum(mask,dtype=cp.int32)
  
    sum_ones = cp.sum(ones)
    sum_zeros = cp.sum(zeros)
  
    num = int(group[-1])
  
    tps = cp.zeros(num).astype('float32') # true positives
    fps = cp.zeros(num).astype('float32') # false positives
  
    update_counter_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void update_counter(const int* group, const float* truth, float* counter, int N) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if(tid<N){
                atomicAdd(counter + group[tid] - 1, truth[tid]);
            }
        }
    ''', 'update_counter')
    
    N = ones.shape[0]
    tpb = 128 
    bpg = math.ceil(N/tpb)
    update_counter_kernel((bpg,), (tpb,), (group, ones, tps, N))  # grid, block and arguments
    update_counter_kernel((bpg,), (tpb,), (group, zeros, fps, N))
    
    tpr = cp.cumsum(tps)/sum_ones
    fpr = cp.cumsum(fps)/sum_zeros
    
    return _calculate_area_under_curve(fpr,tpr)

def _calculate_area_under_curve(fpr,tpr):
    """helper function to calculate area under curve given fpr & tpr arrays"""
    return cp.sum((fpr[1:]-fpr[:-1])*(tpr[1:]+tpr[:-1]))/2 + tpr[0]*fpr[0]/2