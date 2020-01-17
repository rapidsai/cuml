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

from cuml.dask.common import extract_arr_partitions, \
    workers_to_parts

from dask.distributed import default_client


class MultinomialNB(object):

    """
    Distributed Naive Bayes classifier for multinomial models

    The multinomial Naive Bayes classifier is suitable for classification
    with discrete features (e.g., word counts for text classification).

    The multinomial distribution normally requires integer feature counts.
    However, in practice, fractional counts such as tf-idf may also work.
    """
    def __init__(self, client=None, **kwargs):

        """
        Create new multinomial distributed Naive Bayes classifier instance

        Parameters
        -----------

        client : dask.distributed.Client optional Dask client to use
        """

        self.client_ = client if client is not None else default_client()
        self.model_ = None
        self.kwargs = kwargs

    @staticmethod
    def _fit(Xy, classes, kwargs):

        model = MNB(**kwargs)

        for x, y in Xy:
            model.partial_fit(x, y, classes=classes)
        return model.class_count_, model.feature_count_

    @staticmethod
    def _predict(model, X):
        return [model.predict(x) for x in X]

    def fit(self, X, y, classes=None):

        """
        Fit distributed Naive Bayes classifier model

        Parameters
        ----------

        X : dask.Array with blocks containing dense or sparse cupy arrays
        y : dask.Array with blocks containing cupy.ndarray
        classes : array-like containing unique class labels

        Returns
        -------

        cuml.dask.naive_bayes.MultinomialNB current model instance
        """

        # Only Dask.Array supported for now
        if not isinstance(X, dask.array.core.Array):
            raise ValueError("Only dask.Array is supported for X")

        if not isinstance(y, dask.array.core.Array):
            raise ValueError("Only dask.Array is supported for y")

        if len(X.chunks[1]) != 1:
            raise ValueError("X must be chunked by row only. "
                             "Multi-dimensional chunking is not supported")

        worker_parts = self.client_.sync(extract_arr_partitions,
                                         [X, y])

        worker_parts = workers_to_parts(worker_parts)

        n_features = X.shape[1]

        classes = cp.unique(y.map_blocks(cp.unique).compute()) \
            if classes is None else classes

        n_classes = len(classes)

        counts = self.client_.compute([self.client_.submit(
            MultinomialNB._fit,
            p,
            classes,
            self.kwargs,
            workers=[w]
        ) for w, p in worker_parts.items()], sync=True)

        self.model_ = MNB(**self.kwargs)
        self.model_.classes_ = classes
        self.model_.n_classes = n_classes
        self.model_.n_features = X.shape[1]

        self.model_.class_count_ = cp.zeros(n_classes, order="F",
                                            dtype=cp.float32)
        self.model_.feature_count_ = cp.zeros((n_classes, n_features),
                                              order="F", dtype=cp.float32)

        for class_count_, feature_count_ in counts:
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

        """
        Use distributed Naive Bayes model to predict the classes for a
        given set of data samples.

        Parameters
        ----------

        X : dask.Array with blocks containing dense or sparse cupy arrays


        Returns
        -------

        dask.Array containing predicted classes

        """

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

        models = dict([(w, self.client_.scatter(self.model_,
                                                broadcast=True,
                                                workers=[w]))
                       for w, p in x_worker_parts.items()])

        preds = dict([(w, self.client_.submit(
            MultinomialNB._predict,
            models[w],
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
