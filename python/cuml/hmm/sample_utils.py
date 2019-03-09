import numpy as np

def roundup(x, ref):
    return  (int) (ref * np.ceil(x / ref))

def align(A, ldda):
    m = A.shape[0]
    n = A.shape[1]

    B = np.zeros((ldda, n))
    B[:m,:] = A
    return B

def deallign(A, m, n, ldda):
    A = A.reshape((ldda, n), order='F')
    return A[:m, :]

def sample_matrix(m, n, isSymPos=False, isRowNorm=False, isColNorm=False,
                  coef=10., epsilon=1e-06):
    A = np.random.rand(m, n).astype(np.float64)
    A *= coef

    if isSymPos:
        assert m == n
        A = np.matmul(A, A.T) + epsilon * np.eye(n)

    if isColNorm :
        A /= np.sum(A, axis=0)[None, :]

    if isRowNorm:
        A /= np.sum(A, axis=1)[:, None]

    return A

# def sample_data(nObs, params, dt):
#     def _sample_mixture():
#         idx = np.random.multinomial(1, params["pis"][0]).argmax()
#         mu = params["mus"][idx]
#         cov =params["sigmas"][idx]
#
#         # print(cov)
#         # print(np.linalg.det(cov))
#         # TODO : Check warning positive definite
#         x = [0]
#         # print(x)
#         # print('\n')
#         return x
#     print(np.linalg.det(params["sigmas"]))
#     X = np.array([_sample_mixture() for _ in range(nObs)], dtype=dt)
#     return X

def sample_data(nObs, params):
    nCl = params["pis"][0].shape[0]
    counts = np.random.multinomial(nObs, params["pis"][0])
    samples = [np.random.multivariate_normal(params["mus"][idx],
                                             params["sigmas"][idx],
                                             counts[idx])
               for idx in range(nCl)]
    print(params["sigmas"][0].dtype)
    print(np.linalg.eig(params["sigmas"][0]))
    X = np.concatenate(samples)
    return X

def sample_parameters(nDim, nCl):
    mus = sample_matrix(nCl, nDim, isSymPos=False)

    sigmas = [sample_matrix(nDim, nDim, isSymPos=True)
              for _ in range(nCl)]
    sigmas = np.array(sigmas, dtype=np.float64)

    pis = sample_matrix(1, nCl, False)
    pis /= np.sum(pis)

    return {"mus" : mus,
            "sigmas" : sigmas,
            "pis" : pis
            }

def align_parameters(params, ldd):
    aligned = dict()
    for key in params.keys():
        if key is 'sigmas' :
            sigmas = np.hstack(params["sigmas"])
            aligned [key] = align(sigmas, ldd[key])
        elif key in ["llhd", "x"] :
            aligned[key] = align(params[key], ldd[key])
        else :
            A = params[key].T
            aligned[key] = align(A, ldd[key])
    return aligned

def flatten_parameters(params):
    for key in params.keys():
        params[key] = params[key].flatten(order="F")
    return  params

def cast_parameters(params, dtype):
    for key in params.keys():
        params[key] = params[key].astype(dtype)
    return  params