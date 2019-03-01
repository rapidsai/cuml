import pytest
from mlprims import Likelihood
import cudf
import numpy as np
from numba import cuda
from numpy.random import randn
from math import sqrt

from scipy.stats import multivariate_normal


def gen_matrix(m, n, dt):
    return np.random.rand((m, n)).astype(dt)


def gen_input(nDim, nCl, nPts, dt):
    X = gen_matrix(nDim, nPts, dt)
    mus = gen_matrix(nDim, nCl, dt)
    sigmas = gen_matrix(nDim * nDim, nCl, dt)
    rhos = gen_matrix(nDim, 1, dt)
    return {"X": X, "mus": mus, "sigmas": sigmas, "rhos": rhos}


def true_likelihood(x, mu, sigma):
    nPts = x.shape[1]
    nCl = mu.shape[1]
    dim = mu.shape[0]
    return np.array(
        [[
            multivariate_normal.logpdf(
                x[i], mean=mu[k], cov=sigma[k].reshape((dim, dim)))
            for i in range(nPts)]
            for k in range(nCl)])


def get_mse(A, B):
    return np.linalg.norm(A - B) / (A.shape[0] * A.shape[1])


@pytest.mark.parametrize('precision', ['single', 'double'])
def test_likelihood(precision):

    L = Likelihood()

    if precision == 'single':
        dt = np.float32
    else:
        dt = np.float64

    inputs = gen_input(3, 2, 100, dt)
    L.compute(inputs)

    error = 0
    assert error < 1e-5
