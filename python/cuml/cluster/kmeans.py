from cuml.cluster.sg.kmeans import KMeans as sgKmeans
from cuml.dask.cluster.kmeans import KMeans as mgKmeans

class KMeans():
    def __init__(self, handle=None, n_clusters=8, max_iter=300, tol=1e-4,
                verbose=0, random_state=1, precompute_distances='auto',
                init='scalable-k-means++', n_init=1, algorithm='auto'):
    super(KMeans, self).__init__(handle, verbose)
    self.n_clusters = n_clusters
    self.verbose = verbose
    self.random_state = random_state
    self.precompute_distances = precompute_distances
    self.init = init
    self.n_init = n_init
    self.copy_x = None
    self.n_jobs = None
    self.algorithm = algorithm
    self.max_iter = max_iter
    self.tol = tol
    self.labels_ = None
    self.cluster_centers_ = None
    self.n_gpu = 1
    self.kmeans_obj = None
    self.execution_type = 'SG'

    def fit(self, X):
        pass

    def fit_predict(self, X, y):
        pass

    def predict(self, X):
        pass

    def transform(self, X):
        pass

    def score(self, X):
        pass

    def fit_transform(self, X):
        pass

    def get_params(self):
        pass

    def set_params(self):
        pass