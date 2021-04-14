import numpy as np
import scipy
from cuml.neighbors import NearestNeighbors as cuKNN


# from tsnecuda import TSNE
# from tsnecuda import TSNE
# from cuml.test.utils import stress_param
# from cuml.neighbors import NearestNeighbors as cuKNN

from cuml.manifold import TSNE
from sklearn.datasets import make_blobs
from sklearn.manifold.t_sne import trustworthiness
# from cuml.metrics import trustworthiness
from sklearn import datasets


DEFAULT_N_NEIGHBORS = 150

test_datasets = {

                    "digits": datasets.load_digits(),
                 "boston": datasets.load_boston(),
                 "cancer": datasets.load_breast_cancer(),
                 "diabetes": datasets.load_diabetes()}


def validate_embedding(X, Y, score=0.76, n_neighbors=DEFAULT_N_NEIGHBORS):
    """Compares TSNE embedding trustworthiness, NAN and verbosity"""
    nans = np.sum(np.isnan(Y))
    trust = trustworthiness(X, Y, n_neighbors=n_neighbors)

    print("Trust=%s" % trust)
    assert trust > score
    assert nans == 0

def test_tsne_knn_graph_used():


    for i in range(25):
        for dataset in test_datasets.values():
            X = dataset.data

            neigh = cuKNN(n_neighbors=DEFAULT_N_NEIGHBORS).fit(X)
            knn_graph = neigh.kneighbors_graph(X, mode="distance").astype('float32')

            # if type_knn_graph == 'cuml':
            #     knn_graph = cupyx.scipy.sparse.csr_matrix(knn_graph)

            tsne = TSNE(random_state=1,
                        n_neighbors=DEFAULT_N_NEIGHBORS,
                        method="fft",
                        perplexity=50,
                        learning_rate_method='none')



            # Perform tsne with normal knn_graph
            Y = tsne.fit_transform(X, True, knn_graph)

            print("Embedding: %s, mean: %s, min: %s, max: %s" % (Y, np.mean(Y, axis=0), np.min(Y), np.max(Y)))

            print("Y=" + str(hex(id(Y))))
            trust_normal = trustworthiness(X, Y, n_neighbors=DEFAULT_N_NEIGHBORS)

            X_garbage = np.ones(X.shape)
            knn_graph_garbage = neigh.kneighbors_graph(
                X_garbage, mode="distance").astype('float32')

            # if type_knn_graph == 'cuml':
            #     knn_graph_garbage = cupyx.scipy.sparse.csr_matrix(knn_graph_garbage)

            tsne = TSNE(random_state=1,
                        n_neighbors=DEFAULT_N_NEIGHBORS,
                        method="fft",
                        perplexity=50,
                        learning_rate_method='none')

            # Perform tsne with garbage knn_graph
            Y = tsne.fit_transform(X, True, knn_graph_garbage)
            print("Y=" + str(hex(id(Y))))

            trust_garbage = trustworthiness(X, Y, n_neighbors=DEFAULT_N_NEIGHBORS)
            # print (trust_normal - trust_garbage)
            assert (trust_normal - trust_garbage) > 0.15


            print("calling delete2")
            del tsne
            del Y


def test_tsne():
    """
    This tests how TSNE handles a lot of input data across time.
    (1) Numpy arrays are passed in
    (2) Params are changed in the TSNE class
    (3) The class gets re-used across time
    (4) Trustworthiness is checked
    (5) Tests NAN in TSNE output for learning rate explosions
    (6) Tests verbosity
    """

    for i in range(25):
        for dataset in test_datasets.values():
            X = dataset.data
            tsne = TSNE(random_state=1,
                        n_neighbors=DEFAULT_N_NEIGHBORS,
                        method="fft",
                        perplexity=50,
                        learning_rate_method='none')

            Y = tsne.fit_transform(X)
            validate_embedding(X, Y)

            # Again
            tsne = TSNE(random_state=1,
                        n_neighbors=DEFAULT_N_NEIGHBORS,
                        method="fft",
                        perplexity=50,
                        learning_rate_method='none')

            Y = tsne.fit_transform(X)
            validate_embedding(X, Y)

# test_tsne_knn_graph_used()
test_tsne()