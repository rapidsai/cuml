
import numpy as np
from scipy.linalg.lapack import sgemm, ssyrk
from numba import njit, prange


@njit(nogil = True, fastmath = True, cache = True)
def furthest(D, row_norm):
    n = len(D)
    max_d = 0
    max_i = 0
    max_j = 0
    
    for i in range(n):
        norm = row_norm[i]
        for j in range(i):
            d = D[i, j] + row_norm[j] + norm
            if d > max_d:
                max_d = d
                max_i = i
                max_j = j
    return [max_i, max_j]



def furthest_points(X, row_norm):
    D = ssyrk(a = X.T, alpha = -2, trans = 1) if X.flags.c_contiguous else ssyrk(a = X, alpha = -2, trans = 0)
    return furthest(D.T, row_norm)



def BinaryTree(X, row_norm, indices, n, size, centres = None):
    if n <= size:
        return ( (indices, n), None, None )
    else:
        candidates = indices[np.random.choice(n, size = size, replace = False)]
        centres = candidates[ furthest_points(X[candidates], row_norm[candidates]) ]
        
        D = sgemm(a = X[indices].T, b = X[centres].T, alpha = -2, trans_a = 1) + row_norm[centres] + row_norm[indices][:,np.newaxis]
        s = D.argmin(1)==0
        left, right = indices[s], indices[~s]
        
        return (
            (centres, n),
            BinaryTree(X, row_norm, left, len(left), size),
            BinaryTree(X, row_norm, right, len(right), size)
        )



class RandomKDTree:
    def __init__(self, k = 20):
        self.k = k

    def fit(self, X):
        row_norm = np.einsum('ij,ij->i', X, X)
        self.Tree = BinaryTree(X, row_norm, np.arange(n, dtype = np.uint32), len(X), k)
        return

    def query(self, newX, k = 10):
        Levels = []

        for x in newX:
            level = Tree[0]
            size = level[1]
            while size >= 10:
                centres = level[0]
                D = -2 * (x @ X[centres].T) + row_norm[centres] + np.einsum('ij,ij->i', x, x)[:,np.newaxis]
                D = D.ravel()
                if D[0] <= D[1]:
                    if Tree[1] != None:
                        Tree = Tree[1]
                        level = Tree[0]
                    else:
                        break
                else:
                    if Tree[2] != None:
                        Tree = Tree[2]
                        level = Tree[0]
                    else:
                        break
            Levels.append(level)
            
        return Levels

