import numpy as np
from sklearn.utils.validation import check_random_state
from numba import cuda

def roundup(x, ref):
    return (int)(ref * np.ceil(x / ref))


def align(A, ldda):
    m = A.shape[0]
    n = A.shape[1]

    B = np.zeros((ldda, n))
    B[:m, :] = A
    return B


def deallign(A, m, n, ldda):
    A = A.reshape((ldda, n), order='F')
    return A[:m, :]


def sample_matrix(m, n, random_state, isSymPos=False, isRowNorm=False, isColNorm=False,
                  epsilon=1e-06):

    random_state = check_random_state(random_state)
    A = random_state.rand(m, n).astype(np.float64)

    if isSymPos:
        assert m == n
        A = np.matmul(A, A.T) + epsilon * np.eye(n)

    if isColNorm:
        A /= np.sum(A, axis=0)[None, :]

    if isRowNorm:
        A /= np.sum(A, axis=1)[:, None]

    return A


def sample_data(nObs, params):
    nCl = params["pis"][0].shape[0]
    counts = np.random.multinomial(nObs, params["pis"][0])
    samples = [np.random.multivariate_normal(params["mus"][idx],
                                             params["sigmas"][idx],
                                             counts[idx])
               for idx in range(nCl)]
    X = np.concatenate(samples)
    return X


def sample_parameters(nDim, nCl):
    mus = sample_matrix(nCl, nDim, None, isSymPos=False)

    sigmas = [sample_matrix(nDim, nDim, None, isSymPos=True)
              for _ in range(nCl)]
    sigmas = np.array(sigmas, dtype=np.float64)

    pis = sample_matrix(1, nCl, None, False)
    pis /= np.sum(pis)

    return {"mus": mus,
            "sigmas": sigmas,
            "pis": pis
            }


def align_parameters(params, ldd):
    aligned = dict()
    for key in params.keys():
        if key in ["llhd", "x"]:
            aligned[key] = align(params[key], ldd[key])
        else:
            aligned[key] = params[key]
    return aligned


def flatten_parameters(params):
    for key in params.keys():
        params[key] = params[key].flatten(order="F")
    return params


def cast_parameters(params, dtype):
    for key in params.keys():
        params[key] = params[key].astype(dtype)
    return params

def process_parameter(A, ldda, dtype):
    A = align(A, ldda)
    A = A.flatten(order='F')
    A =  A.astype(dtype=dtype)
    return cuda.to_device(A)