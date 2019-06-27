# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from cuml.dask.common import CommsBase, extract_ddf_partitions, to_dask_cudf
from cuml.cluster import KMeans as cumlKMeans

from dask.distributed import wait, get_worker

import random


class KMeans(CommsBase):

    def __init__(self, n_clusters=8, init_method="random", verbose=0):
        super(KMeans, self).__init__(comms_coll=True, comms_p2p=True)
        self.init_(n_clusters=n_clusters, init_method=init_method, verbose=verbose)

    def init_(self, n_clusters, init_method, verbose=0):
        """
        Creates local kmeans instance on each worker
        """
        self.init()

        self.kmeans = [(w, self.client.submit(KMeans.func_build_kmeans_,
                                    a, n_clusters, init_method, verbose, i,
                                    workers=[w])) for i, w, a in self.handles]
        wait(self.kmeans)

    @staticmethod
    def func_build_kmeans_(handle, n_clusters, init_method, verbose, r):
        """
        Create local KMeans instance on worker
        """
        return cumlKMeans(handle=handle, init=init_method, n_clusters=n_clusters, verbose=verbose)

    @staticmethod
    def func_fit(model, df, r): return model.fit(df)

    @staticmethod
    def func_predict(model, df, r): return model.predict(df)

    def run_model_func_on_dask_cudf(self, func, X):
        gpu_futures = self.client.sync(extract_ddf_partitions, X)

        worker_model_map = dict(map(lambda x: (x[0], x[1]), self.kmeans))

        f = [self.client.submit(func,  # Function to run on worker
                      worker_model_map[w],  # Model instance
                      f,  # Input DataFrame partition
                      random.random())  # Worker ID
             for w, f in gpu_futures]
        wait(f)
        return f

    def fit(self, X):
        self.run_model_func_on_dask_cudf(KMeans.func_fit, X)
        return self

    def predict(self, X):
        f = self.run_model_func_on_dask_cudf(KMeans.func_predict, X)
        return to_dask_cudf(f)

    def fit_predict(self, X):
        return self.fit(X).predict(X)
