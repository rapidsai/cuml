from cuml.cluster.sg.kmeans import KMeans as sgKmeans
from cuml.dask.cluster.kmeans import KMeans as mgKmeans
from cuml.utils.input_utils import input_to_multi_gpu

class KMeans():
    def __init__(self, handle=None, n_clusters=8, max_iter=300, tol=1e-4,
                verbose=0, random_state=1, precompute_distances='auto',
                init='scalable-k-means++', n_init=1, algorithm='auto', client=None):
        self.handle = handle
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
        self.execution_type = None
        self.client = client

    def kmeans_init(self, execution_type):
        if self.execution_type == 'SG':
            self.kmeans_obj = sgKmeans(handle=self.handle,
                                    n_clusters=self.n_clusters,
                                    verbose=self.verbose,
                                    random_state=self.random_state,
                                    max_iter=self.max_iter, init=self.init,
                                    tol=self.tol, n_init=self.n_init,
                                    precompute_distances=
                                    self.precompute_distances,
                                    algorithm=self.algorithm,
                                    n_gpu=self.n_gpu)
        else:
            self.kmeans_obj = mgKmeans(n_clusters=self.n_clusters,
                                    init_method=self.init, 
                                    verbose=self.verbose,
                                    client=self.client)

    def fit(self, X):
        X, self.execution_type, self.client = input_to_multi_gpu(X, client=self.client)
        print(self.execution_type)
        self.kmeans_init(self.execution_type)
        self.kmeans_obj.fit(X)

    def fit_predict(self, X, y):
        X, self.execution_type, self.client = input_to_multi_gpu(X, client=self.client)
        y, _, self.client = input_to_multi_gpu(y, execution_type=self.execution_type, client=self.client)
        self.kmeans_init(self.execution_type)
        self.kmeans_obj.fit(X)  

    def predict(self, X):
        X, _, self.client = input_to_multi_gpu(X, execution_type=self.execution_type, client=self.client)
        self.kmeans_obj.fit(X)

    def transform(self, X):
        X, self.execution_type, self.client = input_to_multi_gpu(X, client=self.client)
        self.kmeans_init(self.execution_type)
        self.kmeans_obj.fit(X)

    def score(self, X):
        X, _, self.client = input_to_multi_gpu(X, execution_type=self.execution_type, client=self.client)
        self.kmeans_obj.fit(X)

    def fit_transform(self, X):
        X, self.execution_type, self.client = input_to_multi_gpu(X, client=self.client)
        self.kmeans_init(self.execution_type)
        self.kmeans_obj.fit(X)

    def get_params(self, deep=True):
        self.kmeans_obj.get_params(deep)

    def set_params(self, **params):
        self.kmeans_obj.set_params(**params)