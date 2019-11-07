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

from cuml.dask.common import extract_ddf_partitions, to_dask_cudf, \
    raise_exception_from_futures, workers_to_parts
from dask.distributed import default_client
from cuml.dask.common.comms import worker_state, CommsContext
from dask.distributed import wait

from cuml.dask.neighbors import NearestNeighbors

from collections import OrderedDict

from functools import reduce

from uuid import uuid1


def _raise_import_exception():
    raise Exception("cuML has not been built with multiGPU support "
                    "enabled. Build with the --multigpu flag to"
                    " enable multiGPU support.")


class KNeighborsClassifier(NearestNeighbors):
    """
    Multi-node Multi-GPU NearestNeighbors Model.
    """
    def __init__(self, client=None, **kwargs):
        super(KNeighborsClassifier, self).__init__(client, **kwargs)
        self._y = None

    def fit(self, X, y):
        """
        Fit a multi-node multi-GPU Nearest Neighbors index
        :param X : dask_cudf.Dataframe
        :return : NearestNeighbors model
        """
        super(KNeighborsClassifier, self).fit(X)
        self._y = self.client.sync(extract_ddf_partitions, y)

        if len(self.X) != len(self._y) or X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows and "
                             "partitions")

        return self

    @staticmethod
    def _create_classifier_model(**kwargs):

    @staticmethod
    def _predict(model, X, y, **kwargs):

        if X.shape[0] != y.shape[0]:
            raise ValueError("X (%d) and y (%d) partition sizes unequal" %
                             X.shape[0], y.shape[0])

        from cuml.neighbors import KNeighborsClassifier as cumlKNC
        return cumlKNC(**kwargs).fit(X, y).predict(X)


    def predict(self, X, convert_dtype=True):
        """
        Query the distributed nearest neighbors index
        :param X : dask_cudf.Dataframe Vectors to query. If not
                   provided, neighbors of each indexed point are returned.
        :param n_neighbors : Number of neighbors to query for each row in
                             X. If not provided, the n_neighbors on the
                             model are used.
        :param return_distance : If false, only indices are returned
        :return : dask_cudf.DataFrame containing distances
        :return : dask_cudf.DataFrame containing indices
        """
        nn_fit, out_i_futures = \
            super(KNeighborsClassifier, self).kneighbors(X, None, False)

        # Co-locate X and y partitions
        y_parts = self.client.sync(extract_ddf_partitions, self.y)








