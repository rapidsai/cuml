import numpy as np


def zero_out(A, m, n, ldda):
    A = A.reshape((ldda, n), order='F')
    A[m:ldda, :] = 0.
    return A.flatten(order='F')


def complete_zeros(A, m, n, ldda):
    B = np.zeros((ldda, n))
    B[:m,:] = A
    return B

def remove_zeros(A, m, n, ldda):
    A = A.reshape((ldda, n), order='F')
    return A[:m, :]

def sample_matrix(m, n, lddA, dt, isSymPos=False, isNorm=False):
    A = np.random.rand(lddA, n)

    if isSymPos:
        assert m == n
        A[0:m, :] = A[0:m, :] + A[0:m, :].T
        for idx in range(m):
            A[idx, idx] += m

    if isNorm :
        A = A.flatten(order='F')
        zero_out(A, m, n, lddA)
        A = A.reshape((lddA, n), order='F')
        A /= np.sum(A, axis=0)[None, :]
        return A.flatten(order='F')

    A = A.flatten(order='F')
    A_al = zero_out(A, m, n, lddA)
    A = A.reshape((m, n), order="F")
    return A, A_al

def print_matrix(m, n, A_al, lddA) :
    A = A_al.reshape((lddA, n), order='F')
    print(A)

def sample_mus(nDim, nCl, lddmu, dt):
    return sample_matrix(nDim, nCl, lddmu, False)

def sample_sigmas(nDim, nCl, lddsigma, dt):
    sigmas_samples = [sample_matrix(nDim, nDim, lddsigma, True) for _ in range(nCl)]
    sigmas_al = np.concatenate([sigma[1] for sigma in sigmas_samples])
    sigmas = np.concatenate([sigma[0] for sigma in sigmas_samples])
    return sigmas, sigmas_al


def sample_pis(nCl, lddpis, dt):
    pis, pis_al = sample_matrix(nCl, 1, lddpis, False, False)
    pis /= np.sum(pis)
    pis_al /= np.sum(pis_al)
    return pis, pis_al


def sample_llhd(nCl, nObs, ldLlhd):
    return sample_matrix(nCl, nObs, ldLlhd, isNorm=True)


def sample_data(mus, sigmas, pis, nObs, nDim, lddx, dt):
    def _sample_mixture():
        idx = np.random.multinomial(1, pis).argmax()
        x = np.random.multivariate_normal(mus[idx], sigmas[idx], 1)[0]
        return x
    X = np.array([_sample_mixture() for _ in range(nObs)], dtype=dt).T
    X_al = complete_zeros(X, nDim, nObs, lddx)
    X_al.reshape((lddx, nObs),  order='F')
    return X, X_al

def sample_parameters(nDim, nCl, lddmu, lddsigmas, lddpis, dt):
    return {"mus" : sample_mus(nDim=nDim, nCl=nCl, lddmu=lddmu, dt=dt),
            "sigmas" : sample_sigmas(nDim=nDim, nCl=nCl, lddsigma=lddsigmas, dt=dt),
            "pis" : sample_pis(nCl=nCl, lddpis=lddpis, dt=dt)
            }

