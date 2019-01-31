# Copyright (c) 2018, NVIDIA CORPORATION.
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

import numpy as np
import pandas as pd
import cudf


class KNNparams:
    def __init__(self, n_gpus):
        self.n_gpus = n_gpus


class KNN:
    """

    Create a DataFrame, fill it with data, and compute KNN:

    .. code-block:: python

      import cudf
      from cuml import KNN
      import numpy as np

      np_float = np.array([
        [1,2,3], # Point 1
        [1,2,4], # Point 2
        [2,2,4]  # Point 3
      ]).astype('float32')

      gdf_float = cudf.DataFrame()
      gdf_float['dim_0'] = np.ascontiguousarray(np_float[:,0])
      gdf_float['dim_1'] = np.ascontiguousarray(np_float[:,1])
      gdf_float['dim_2'] = np.ascontiguousarray(np_float[:,2])

      print('n_samples = 3, n_dims = 3')
      print(gdf_float)

      knn_float = KNN(n_gpus=1)
      knn_float.fit(gdf_float)
      Distance,Index = knn_float.query(gdf_float,k=3) #get 3 nearest neighbors

      print(Index)
      print(Distance)

    Output:

    .. code-block:: python

      n_samples = 3, n_dims = 3

      dim_0 dim_1 dim_2

      0   1.0   2.0   3.0
      1   1.0   2.0   4.0
      2   2.0   2.0   4.0

      # Index:

               index_neighbor_0 index_neighbor_1 index_neighbor_2
      0                0                1                2
      1                1                0                2
      2                2                1                0
      # Distance:

               distance_neighbor_0 distance_neighbor_1 distance_neighbor_2
      0                 0.0                 1.0                 2.0
      1                 0.0                 1.0                 1.0
      2                 0.0                 1.0                 2.0

    For an additional example see `the KNN notebook <https://github.com/rapidsai/cuml/blob/master/python/notebooks/knn_demo.ipynb>`_. For additional docs, see `scikitlearn's KDtree <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree>`_.

    """
    def __init__(self, n_gpus=-1):
        # -1 means using all gpus

        # import faiss
        try:
            import faiss
        except ImportError:
            msg = "KNN not supported without faiss"
            raise ImportError(msg)


        self.params = KNNparams(n_gpus)

    def fit(self, X):

        try:
            import faiss
        except ImportError:
            msg = "KNN not supported without faiss"
            raise ImportError(msg)

        if (isinstance(X, cudf.DataFrame)):
            X = self.to_nparray(X)
        assert len(X.shape) == 2, 'data should be two dimensional'
        n_dims = X.shape[1]
        cpu_index = faiss.IndexFlatL2(n_dims)
        # build a flat (CPU) index
        if self.params.n_gpus == 1:
            res = faiss.StandardGpuResources()
            # use a single GPU
            # make it a flat GPU index
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index,
                                                    ngpu=self.params.n_gpus)
        gpu_index.add(X)
        self.gpu_index = gpu_index

    def query(self, X, k):
        X = self.to_nparray(X)
        D, I = self.gpu_index.search(X, k)
        D = self.to_cudf(D, col='distance')
        I = self.to_cudf(I, col='index')
        return D, I

    def to_nparray(self, x):
        if isinstance(x, cudf.DataFrame):
            x = x.to_pandas()
        return np.ascontiguousarray(x)

    def to_cudf(self, df, col=''):
        # convert pandas dataframe to cudf dataframe
        if isinstance(df,np.ndarray):
            df = pd.DataFrame({'%s_neighbor_%d'%(col, i): df[:, i] for i in range(df.shape[1])})
        pdf = cudf.DataFrame.from_pandas(df)
        return pdf
