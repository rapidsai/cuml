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

from cuml.dask.common import extract_ddf_partitions, to_dask_cudf
from dask.distributed import default_client
from cuml.dask.common.comms import worker_state, CommsContext
from dask.distributed import wait
import numpy as np

from uuid import uuid1

import cudf

class NearestNeighbors(object):
    """
    Multi-node Multi-GPU NearestNeighbors Model.
    """
    def __init__(self, client=None, **kwargs):
        self.client = default_client() if client is None else client
        self.kwargs = kwargs

    def fit(self, X, replicate=False):
        """
        Fit a multi-node multi-GPU Nearest Neighbors index
        :param X : dask_cudf.Dataframe
        :param replicate : bool, string, or int. If X is small enough,
                it can be replicated onto the workers, which will
                enable embarrassingly parallel prediction. Setting
                replicate to True or False explicitly turns it on
                or off. Setting it to a string specifies the threshold,
                using a format like "2G", for determining whether X
                should be replicated. If this value is an int, the
                number of elements is used as a threshold.
        :return : NearestNeighbors model
        """
        return self

    def kneighbors(self, X, k=None, replicate=False):
        """
        Query the NearestNeighbors index
        :param X : dask_cudf.Dataframe list of vectors to query
        :param k : Number of neighbors to query for each row in X
        :param replicate : bool, string, or int. If X is small enough, it
                can be replicated onto the workers containing the indices.
                If the indices are replicated, this means only a single
                worker needs to perform the predict, otherwise, the results
                of the query are able to be reduced to a single partition.
                Setting replicate to True or False explicitly turns it
                on or off. Setting it to a string, specifies the threshold,
                using a format like "2GB", for determining whether X should
                be replicated. If this value is an int, the number of
                elements is used as a threshold.
        :return : dask_cudf.Dataframe containing the results
        """

