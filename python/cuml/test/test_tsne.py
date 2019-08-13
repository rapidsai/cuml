
from cuml.manifold import TSNE

from sklearn.manifold.t_sne import trustworthiness
from sklearn import datasets
import pandas as pd
import numpy as np
import cudf
import pytest

dataset_names = ['digits', 'boston', 'iris', 'breast_cancer',
                 'diabetes', 'wine']


@pytest.mark.parametrize('name', dataset_names)
def test_tsne(name):
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
    datasets
    X = eval("datasets.load_{}".format(name))().data
    X_cudf = cudf.DataFrame.from_pandas(pd.DataFrame(X))

    total_trust = 0
    trials = 0

    for i in range(200):
        print("iteration = ", i)

        tsne = TSNE(2, random_state=i, verbose=0, learning_rate=2+i)

        tsne.fit(X_cudf)

        # Reuse
        Y = tsne.fit_transform(X)
        nans = np.sum(np.isnan(Y))
        assert nans == 0

        trials +=1

    print("Total Trust %s" % (total_trust / trials))
    assert (total_trust / trials) >= 0.97
