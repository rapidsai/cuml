import numpy as np


def zero_out(A, m, n, ldda):
    A[m:ldda, :] = 0.
    return A


def sample_matrix(m, n, lddA, isSymPos):
    A = np.random.rand(lddA, m)
    if isSymPos:
        assert m == n
        A[0:m, :] = A[0:m, :] + A[0:m, :].T
        for idx in range(m):
            A[idx, idx] += m
    return zero_out(A, m, n, lddA)


def sample_mus(nDim, nCl, lddmu):
    return sample_matrix(nDim, nCl, lddmu, False)


def sample_sigmas(nDim, nCl, lddsigma):
    s = [sample_matrix(nDim, nDim, lddsigma, True).T.reshape(
        lddsigma * nDim) for _ in range(nCl)]
    return np.array(s)


def sample_pis(nCl):
    pis = np.random.rand(nCl)
    pis /= np.sum(pis)
    return pis


def sample_llhd(nCl, nObs, ldLlhd):
    eps = 1e-6
    llhd = sample_matrix(nCl, nObs, ldLlhd, False)
    llhd /= (np.sum(llhd, axis=1) + eps * np.ones(ldLlhd))[:, None]
    return llhd


def sample_mixture(mus, sigmas, pis, nCl, nDim, nObs, lddsigma, dt):
    def _sample_mixture():
        idx = np.random.multinomial(1, pis).argmax()
        cov = sigmas[idx].reshape((nDim, lddsigma))[:nDim, :nDim]
        return np.random.multivariate_normal(mus[idx], cov, 1)
    return np.array([_sample_mixture() for _ in range(nObs)], dtype=dt)
