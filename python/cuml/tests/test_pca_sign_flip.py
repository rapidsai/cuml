import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as skPCA
from cuml.decomposition import PCA as cuPCA

def test_pca_sign_flip():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    sk_pca = skPCA(n_components=10, random_state=13)
    sk_pca_res = sk_pca.fit_transform(X)
    sk_pca_comp = sk_pca.components_
    cu_pca = cuPCA(n_components=10)
    cu_pca_res = cu_pca.fit_transform(X).to_numpy()
    cu_pca_comp = cu_pca.components_.to_numpy()
    assert np.allclose(sk_pca_res, cu_pca_res, rtol=1e-3), "Transform results differ!"
    assert np.allclose(sk_pca_comp, cu_pca_comp, rtol=1e-3), "Components differ!"
