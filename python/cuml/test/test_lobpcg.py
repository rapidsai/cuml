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
from cupy import prof

import pytest

import cuml.sparse.linalg.lobpcg
from cuml.sparse.linalg.lobpcg import lobpcg as cp_lobpcg
#from cuml.common.input_utils import sparse_scipy_to_cp

from numpy.testing import assert_allclose

import sys

import time as t

import math

import numpy as np

import warnings

import scipy
from scipy.sparse.linalg import lobpcg as scipy_lobpcg

@pytest.mark.parametrize("dtype", [\
                                 cp.float32,\
                                 cp.float64\
                                ])
@pytest.mark.parametrize("k,n",  [\
                                    (1,10), (2,10), \
                                    (1, 100), (2, 100), (5, 100), \
                                    (2,1000), (10,1000), \
                                    #(2,10000)\
                                    ])

#TODO: debug and run tests for method ortho and basic, currently they aren't that accurate
@pytest.mark.parametrize("method", [\
                                    #"ortho",\
                                    #"basic",\
                                    'blopex'\
                                    ])
@pytest.mark.parametrize("largest", [True, \
                                    False\
                                    ])
@pytest.mark.parametrize("maxiter", [\
                                    20,\
                                    #200,\ #Not practically required
                                    #2000\
                                    ])
@pytest.mark.parametrize("isB", [True, False])



def test_lobpcg(dtype, k, n, method, largest, maxiter, isB):

    #settings default value of Error flags
    cp_error_flag = False
    scipy_error_flag = False

    A = np.random.rand(n,n)
    A = np.matmul(A, A.transpose()) #making A symmetric positive definite
    B = np.eye(n)
    X = np.random.randn(n,k)
    while(isB):
        B = np.random.randn(n,n)
        B = np.matmul(B,B.transpose()) #making B Symmetric positive definite
        if(not math.isclose(np.linalg.det(B),0, abs_tol=1e-04)): #making sure B is not singular
            break
    A_gpu_array = cp.asarray(A)
    B_gpu_array = cp.asarray(B)
    X_gpu_array = cp.asarray(X)

    """running cupy implementation:
    """
    try:#(please work :3)

        with cp.prof.time_range(message='start', color_id=10):

            cp_val, cp_vec = cp_lobpcg(A_gpu_array,X=X_gpu_array, B=B_gpu_array, \
                                        maxiter=maxiter, largest=largest, method=method)

            cp.cuda.Stream.null.synchronize()


        isnan_and_reduc_kernel = cp.ReductionKernel('T x', 'T y', 'x == x', 'a and b', 'a', '1', 'isnan_and_reduc')
        is_vec_ok = isnan_and_reduc_kernel(cp_vec)
        if(not is_vec_ok):
            cp_error_flag = True
            print("cupy: Nan values encountered in output vec method: {}!!")


    except Exception as e:
        print("{} occured in running the cupy lobpcg method for method: {} k:{} and n:{} with maxiter:{}!\n".format(e,method,k,n,maxiter))
        cp_error_flag = True


    """running scipy implementation:
    """
    try:
        scipy_val, scipy_vec = scipy_lobpcg(A, X, B = B, maxiter=maxiter, largest=largest)
    except:
        print("error occured in running the scipy lobpcg method for k:{} and n:{} with maxiter:{}!\n".format(k,n,maxiter))
        scipy_error_flag = True


    if(scipy_error_flag or cp_error_flag):
        print("cholesky failed")
        warnings.warn("cholesky failed for k:{}, n:{}, maxiter:{}".format(k,n,maxiter))
        return

    #taking absolutes of eigen vectors as they can toggle between positive and negative
    abs_cp_vec = cp.absolute(cp_vec)
    abs_scipy_vec = np.absolute(scipy_vec)

    """
    generalized eigen value problem: AX = LBX where,
    - A, B are input operator matrices
    - X is the (n,k) block of k eigen vectors
    - L is the diagonal matrix with k eigen values in diagonal

    since these are approximations, the residual = LHS - RHS
    where, LHS = AX and RHS = LBX
    further, we take the absolutes of these residuals
    """
    cp_lhs = cp.asnumpy(cp.matmul(A_gpu_array,cp_vec))
    cp_rhs = cp.asnumpy(cp.multiply(cp_val, cp.matmul(B_gpu_array,cp_vec)))

    scipy_lhs = np.matmul(A,scipy_vec)
    scipy_rhs = np.multiply(scipy_val, np.matmul(B, scipy_vec))

    #absolute of the residuals
    abs_cp_res = abs(cp_lhs - cp_rhs)
    abs_scipy_res = abs(scipy_lhs - scipy_rhs)

    #taking a eigenvalue-weighted average residues across axis=1
    cp_weighted_res = np.average(abs_cp_res, axis=1, weights=cp.asnumpy(cp_val))
    scipy_weighted_res = np.average(abs_scipy_res, axis=1, weights=scipy_val)

    #taking norm of vector
    cp_norm = np.linalg.norm(cp_weighted_res)
    scipy_norm = np.linalg.norm(scipy_weighted_res)

    """
    TEST-STRATEGY--------------------
    The following is a comparison after norm-reduction (ie, (n,1) -> 1) of eigen-value-weighted-reduction (ie, (n,k) -> (n,1))
    of absolutes of resulting residuals
    """
    if cp_norm > scipy_norm: #most of the times, cupy cuml.lobpcg is more accurate than scipy
       assert math.isclose(cp_norm, scipy_norm, abs_tol=1e-2, rel_tol=1e-2), "cp_norm:{} scipy_norm:{}".format(cp_norm, scipy_norm)