import cuml

import sklearn.cluster, sklearn.neighbors
import umap

class AlgorithmPair():
    """
    Wraps a cuML algorithm and (optionally) a cpu-based algorithm
    (typically scikit-learn, but does not need to be). Provides
    mechanisms to each version with default arguments.
    If no CPU-based version of the algorithm is availab.e, pass None for the
    cpu_class when instantiating
    """
    def __init__(self,
                 cpu_class,
                 cuml_class,
                 shared_args,
                 cuml_args={},
                 sklearn_args={},
                 name=None,
                 accepts_labels=True):
        """
        Parameters
        ----------
        cpu_class : class
           Class for CPU version of algorithm. Set to None if not available.
        cuml_class : class
           Class for cuML algorithm
        shared_args : dict
           Arguments passed to both implementations
        ....
        accepts_labels : boolean
           If True, the fit methods expects both X and y
           inputs. Otherwise, it expects only an X input.
        """
        if name:
            self.name = name
        else:
            self.name = cuml_class.__name__
        self.accepts_labels = accepts_labels
        self.cpu_class = cpu_class
        self.cuml_class = cuml_class
        self.shared_args = shared_args
        self.cuml_args = cuml_args
        self.sklearn_args = sklearn_args

    def __str__(self):
        return "AlgoPair:%s" % (self.name)

    def run_cpu(self, data, **override_args):
        """Runs the cpu-based algorithm's fit method on specified data"""
        if self.cpu_class is None:
            raise ValueError("No CPU implementation for %s" % self.name)
        all_args = {**self.shared_args, **self.sklearn_args}
        all_args = {**all_args, **override_args}

        cpu_obj = self.cpu_class(**all_args)
        if self.accepts_labels:
            cpu_obj.fit(data[0], data[1])
        else:
            cpu_obj.fit(data[0])

    def run_cuml(self, data, **override_args):
        """Runs the cuml-based algorithm's fit method on specified data"""
        all_args = {**self.shared_args, **self.cuml_args}
        all_args = {**all_args, **override_args}

        cuml_obj = self.cuml_class(**all_args)
        if self.accepts_labels:
            cuml_obj.fit(data[0], data[1])
        else:
            cuml_obj.fit(data[0])

def all_algorithms():
    return [
        AlgorithmPair(
            sklearn.cluster.KMeans,
            cuml.cluster.KMeans,
            shared_args=dict(init='random',
                             n_clusters=8,
                             max_iter=300),
            name='KMeans',
            accepts_labels=False),
        AlgorithmPair(
            sklearn.decomposition.PCA,
            cuml.PCA,
            shared_args=dict(n_components=10),
            name='PCA',
            accepts_labels=False),
        AlgorithmPair(
            sklearn.neighbors.NearestNeighbors,
            cuml.neighbors.NearestNeighbors,
            shared_args=dict(n_neighbors=1024),
            sklearn_args=dict(algorithm='brute'),
            cuml_args=dict(n_gpus=1),
            name='NearestNeighbors',
            accepts_labels=False),
        AlgorithmPair(
            sklearn.cluster.DBSCAN,
            cuml.DBSCAN,
            shared_args=dict(eps=3, min_samples=2),
            sklearn_args=dict(algorithm='brute'),
            name='DBSCAN',
            accepts_labels=False
        ),
        AlgorithmPair(
            sklearn.linear_model.LinearRegression,
            cuml.linear_model.LinearRegression,
            shared_args={},
            name='LinearRegression',
            accepts_labels=True),
        AlgorithmPair(
            umap.UMAP,
            cuml.manifold.UMAP,
            shared_args=dict(n_neighbors=5, n_epochs=500),
            name='UMAP',
            accepts_labels=False),
    ]


def algorithm_by_name(name):
    """Returns the algorithm pair with the name 'name' (case-insensitive)"""
    algos = all_algorithms()
    return next((a for a in algos if a.name.lower() == name.lower()),
                None)
