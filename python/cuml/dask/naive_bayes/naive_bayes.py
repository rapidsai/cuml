#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

from cuml.utils import rmm_cupy_ary

from dask.distributed import default_client

from cuml.dask.common.utils import run_cupy_sparse_patch_on_workers


class MultinomialNB(object):

    """
    Distributed Naive Bayes classifier for multinomial models

    Examples
    --------

    Load the 20 newsgroups dataset from Scikit-learn and train a
    Naive Bayes classifier.

    .. code-block:: python

    import cupy as cp

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer

    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client

    from cuml.dask.common import to_sp_dask_array

    from cuml.dask.naive_bayes import MultinomialNB

    # Create a local CUDA cluster

    cluster = LocalCUDACluster()
    client = Client(cluster)

    # Load corpus

    twenty_train = fetch_20newsgroups(subset='train',
                              shuffle=True, random_state=42)

    cv = CountVectorizer()
    xformed = cv.fit_transform(twenty_train.data).astype(cp.float32)

    X = to_sp_dask_array(xformed, client)
    y = dask.array.from_array(twenty_train.target, asarray=False,
                          fancy=False).astype(cp.int32)

    # Train model

    model = MultinomialNB()
    model.fit(X, y)

    # Compute accuracy on training set

    model.score(X, y)


    Output:

    .. code-block:: python

    0.9244298934936523

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

        run_cupy_sparse_patch_on_workers(self.client_)

    @staticmethod
    def _fit(Xy, classes, kwargs):

        model = MNB(**kwargs)

        for x, y in Xy:
            model.partial_fit(x, y, classes=classes)

        return model.class_count_, model.feature_count_

    @staticmethod
    def _predict(model, X):
        return [model.predict(x) for x in X]

    @staticmethod
    def _unique(x):
        return rmm_cupy_ary(cp.unique, x)

    @staticmethod
    def _get_class_counts(x):
        return x[0]

    @staticmethod
    def _get_feature_counts(x):
        return x[1]

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

        classes = MultinomialNB._unique(y.map_blocks(
            MultinomialNB._unique).compute()) if classes is None else classes

        n_classes = len(classes)

        counts = [self.client_.submit(
            MultinomialNB._fit,
            p,
            classes,
            self.kwargs,
            workers=[w]
        ) for w, p in worker_parts.items()]

        class_counts = self.client_.compute(
            [self.client_.submit(MultinomialNB._get_class_counts, c)
             for c in counts], sync=True)
        feature_counts = self.client_.compute(
            [self.client_.submit(MultinomialNB._get_feature_counts, c)
             for c in counts], sync=True)

        self.local_model = MNB(**self.kwargs)
        self.local_model.classes_ = classes
        self.local_model.n_classes = n_classes
        self.local_model.n_features = X.shape[1]

        self.local_model.class_count_ = rmm_cupy_ary(cp.zeros,
                                                     n_classes,
                                                     order="F",
                                                     dtype=cp.float32)
        self.local_model.feature_count_ = rmm_cupy_ary(cp.zeros,
                                                       (n_classes, n_features),
                                                       order="F",
                                                       dtype=cp.float32)

        for class_count_ in class_counts:
            self.local_model.class_count_ += class_count_
        for feature_count_ in feature_counts:
            self.local_model.feature_count_ += feature_count_

        self.local_model.update_log_probs()

    @staticmethod
    def _get_part(parts, idx):
        return parts[idx]

    @staticmethod
    def _get_size(arrs):
        return arrs.shape[0]

    def predict(self, X):
        # TODO: Once cupy sparse arrays are fully supported
        #  underneath Dask arrays,
        # this can extend DelayedPredictionMixin.
        """
        Predict classes for distributed Naive Bayes classifier model

        Parameters
        ----------

        X : dask.Array with blocks containing dense or sparse cupy arrays

        Returns
        -------

        dask.Array containing class predictions
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

        sizes = self.client_.compute([x[1] for x in futures], sync=True)

        models = self.client_.scatter(self.local_model,
                                      broadcast=True,
                                      direct=True,
                                      hash=False,
                                      workers=list(x_worker_parts.keys()))

        preds = dict([(w, self.client_.submit(
            MultinomialNB._predict,
            models,
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

    def score(self, X, y):
        """
        Compute accuracy score

        Parameters
        ----------

        X : Dask.Array with features to predict
        y : Dask.Array with labels to use for computing accuracy

        Returns
        -------
        score : float the resulting accuracy score
        """

        y_hat = self.predict(X)
        gpu_futures = self.client_.sync(extract_arr_partitions, [y_hat, y])

        def _count_accurate_predictions(y_hat_y):
            y_hat, y = y_hat_y
            y_hat = rmm_cupy_ary(cp.asarray, y_hat, dtype=y_hat.dtype)
            y = rmm_cupy_ary(cp.asarray, y, dtype=y.dtype)
            return y.shape[0] - cp.count_nonzero(y-y_hat)

        key = uuid1()

        futures = [self.client_.submit(_count_accurate_predictions,
                                       wf[1],
                                       workers=[wf[0]],
                                       key="%s-%s" % (key, idx)).result()
                   for idx, wf in enumerate(gpu_futures)]

        return sum(futures) / X.shape[0]
