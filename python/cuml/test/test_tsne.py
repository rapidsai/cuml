

from cuml.manifold import TSNE

from sklearn.manifold.t_sne import trustworthiness
from sklearn import datasets
import pandas as pd
import numpy as np
import cudf
import pytest


def unit_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.unit)


def quality_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.quality)


def stress_param(*args, **kwargs):
    return pytest.param(*args, **kwargs, marks=pytest.mark.stress)


@pytest.mark.parametrize('nrows', [unit_param(30), quality_param(5000),
                         stress_param(500000)])
@pytest.mark.parametrize('n_feats', [unit_param(10), quality_param(100),
                         stress_param(1000)])
def test_blobs_cluster(nrows, n_feats):
    X, _ = datasets.make_blobs(
        n_samples=nrows, n_features=n_feats, centers=5, random_state=0)
    Y = TSNE(verbose=False).fit_transform(X)

    trust = trustworthiness(X, Y)
    print("Trust = ", trust)
    if trust < 0.97:
        if trust < 0.95:
            assert trust > 0.9
    del X, Y


def test_tsne():
    """
    This tests how TSNE handles a lot of input data across time.
    (1) cuDF DataFrames are passed input
    (2) Numpy arrays are passed in
    (3) Params are changed in the TSNE class
    (4) The class gets re-used across time
    (5) Trustworthiness is checked
    (6) Tests NAN in TSNE output for learning rate explosions
    (7) Tests verbosity
    """
    from sklearn.datasets import load_digits as data1
    from sklearn.datasets import load_boston as data2
    from sklearn.datasets import load_iris as data3
    from sklearn.datasets import load_breast_cancer as data4
    from sklearn.datasets import load_diabetes as data5

    X1 = data1().data
    X1_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X1))

    X2 = data2().data

    X3 = data3().data
    X3_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X3))

    X4 = data4().data

    X5 = data5().data
    X5_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X5))

    for i in range(3):
        print("-------------------------------------")
        print("iteration = ", i)

        tsne = TSNE(2, random_state=i, verbose=0, learning_rate=2+i)

        Y = tsne.fit_transform(X1_cudf).to_pandas().values
        nans = np.sum(np.isnan(Y))
        trust = trustworthiness(X1, Y)
        print("Trust = ", trust)
        if trust < 0.97:
            if trust < 0.95:
                assert trust > 0.9
        del Y

        # Reuse
        Y = tsne.fit_transform(X2)
        nans = np.sum(np.isnan(Y))
        trust = trustworthiness(X2, Y)
        print("Trust = ", trust)
        if trust < 0.97:
            if trust < 0.95:
                assert trust > 0.9
        del Y

        # Again
        tsne = TSNE(2, random_state=i+2, verbose=1, learning_rate=2+i+2)

        Y = tsne.fit_transform(X3_cudf).to_pandas().values
        nans = np.sum(np.isnan(Y))
        trust = trustworthiness(X3, Y)
        print("Trust = ", trust)
        if trust < 0.97:
            if trust < 0.95:
                assert trust > 0.9
        del Y

        # Reuse
        Y = tsne.fit_transform(X4)
        nans = np.sum(np.isnan(Y))
        trust = trustworthiness(X4, Y)
        print("Trust = ", trust)
        if trust < 0.97:
            if trust < 0.95:
                assert trust > 0.9
        del Y

        # Again
        tsne = TSNE(2, random_state=i+4, verbose=0, learning_rate=2+i+4)

        Y = tsne.fit_transform(X5_cudf).to_pandas().values
        nans = np.sum(np.isnan(Y))
        trust = trustworthiness(X5, Y)
        assert nans == 0
        print("Trust = ", trust)
        if trust < 0.97:
            if trust < 0.95:
                assert trust > 0.9
        del Y
