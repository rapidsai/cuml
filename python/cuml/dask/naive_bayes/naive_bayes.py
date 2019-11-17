#
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
from cuml.naive_bayes import MultinomialNB as MNB

from cuml.dask.preprocessing import LabelBinarizer

from cuml.dask.common import extract_ddf_partitions, worker_to_parts

from dask.distributed import default_client

class MultinomialNB(object):

    def __init__(self, client=None):

        self.client_ = client if client is not None else default_client()
        pass

    @staticmethod
    def _fit(X, y):

    def fit(self, X, y):


        self.label_binarizer_ = LabelBinarizer()
        Y = self.label_binarizer_.fit_transform(y)

        x_futures = self.client_.sync(extract_ddf_partitions, X)
        y_futures = self.client_.sync(extract_ddf_partitions, Y)
        x_worker_parts = worker_to_parts(x_futures)
        y_worker_parts = worker_to_parts(y_futures)

        models = [self.client_.submit(
            MultinomialNB._fit,
        ) for w, p in x_worker_parts]




        """
        Distributed Naive Bayes works as follows:
        1. Distributed label binarization
        2. Each worker calls fit(_labels_binarized=True) on SG model
        3. Each worker pulls prior and feature counts from local SG model
        4. Prior & feature counts from all workers collected to client & summed
        5. model.class_count_ and model.feature_count_ are set back on the client's
           local model, which will be broadcast to workers for embarrassingly parallel
           prediction.
        :param X:
        :param y:
        :return:
        """
        pass

    def predict(self, X):
        pass




