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

import cupy as cp


from uuid import uuid1

from cuml.naive_bayes import MultinomialNB as MNB

import dask

from cuml.dask.common import to_dask_df, extract_arr_partitions, \
    extract_ddf_partitions, workers_to_parts, parts_to_ranks, \
    flatten_grouped_results

from dask.distributed import default_client


class MultinomialNB(object):

    def __init__(self, client=None, **kwargs):

        self.client_ = client if client is not None else default_client()
        self.model_ = None
        self.kwargs = kwargs

    @staticmethod
    def _fit(X, y, classes, kwargs):

        model = MNB(**kwargs)

        for x, y in zip(X, y):
            model.partial_fit(x, y, classes=classes)
        return model.class_count_, model.feature_count_

    @staticmethod
    def _predict(model, X):
        return [model.predict(x) for x in X]

    def fit(self, X, y, classes=None):

        # Only Dask.Array supported for now
        if not isinstance(X, dask.array.core.Array):
            raise ValueError("Only dask.Array is supported for X")

        if len(X.chunks[1]) != 1:
            raise ValueError("X must be chunked by row only. "
                             "Multi-dimensional chunking is not supported")

        x_worker_parts = self.client_.sync(extract_arr_partitions, X)
        y_worker_parts = self.client_.sync(extract_arr_partitions, y)

        x_worker_parts = workers_to_parts(x_worker_parts)
        y_worker_parts = workers_to_parts(y_worker_parts)

        n_features = X.shape[1]

        classes = cp.unique(y.map_blocks(cp.unique).compute()) \
            if classes is None else classes

        n_classes = len(classes)

        counts = self.client_.compute([self.client_.submit(
            MultinomialNB._fit,
            p,
            y_worker_parts[w],
            classes,
            self.kwargs,
        ) for w, p in x_worker_parts.items()], sync=True)

        self.model_ = MNB(**self.kwargs)
        self.model_.classes_ = classes
        self.model_.n_classes = n_classes
        self.model_.n_features = X.shape[1]

        self.model_.class_count_ = cp.zeros(n_classes, order="F",
                                            dtype=cp.float32)
        self.model_.feature_count_ = cp.zeros((n_classes, n_features),
                                      order="F", dtype=cp.float32)

        for class_count_, feature_count_ in counts:

            print("class_count_=%s, feature_count_=%s" % (class_count_, feature_count_))
            self.model_.class_count_ += class_count_
            self.model_.feature_count_ += feature_count_

        self.model_.update_log_probs()

    @staticmethod
    def _get_part(parts, idx):
        return parts[idx]

    @staticmethod
    def _get_size(arrs):
        return arrs.shape[0]

    def predict(self, X):

        gpu_futures = self.client_.sync(extract_arr_partitions, X)
        x_worker_parts = workers_to_parts(gpu_futures)

        key = uuid1()

        futures = [(wf[0],
                    self.client_.submit(MultinomialNB._get_size,
                                  wf[1],
                                  workers=[wf[0]],
                                  key="%s-%s" % (key, idx)))
                   for idx, wf in enumerate(gpu_futures)]

        sizes = self.client_.compute(list(map(lambda x: x[1],
                                              futures)), sync=True)

        preds = dict([(w, self.client_.submit(
            MultinomialNB._predict,
            self.model_,
            p
        )) for w, p in x_worker_parts.items()])

        final_parts = {}
        to_concat = []
        for wp, size in zip(gpu_futures, sizes):
            w, p = wp
            if w not in final_parts:
                final_parts[w] = 0

            to_concat.append(
                dask.array.from_delayed(
                    dask.delayed(self.client_.submit(MultinomialNB._get_part,
                                        preds[w],
                                        final_parts[w])),
                    dtype=cp.int32, shape=(size,)))

            final_parts[w] += 1

        return dask.array.concatenate(to_concat)
