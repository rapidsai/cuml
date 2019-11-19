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

from cuml.naive_bayes import MultinomialNB as MNB

from cuml.dask.preprocessing import LabelBinarizer

from cuml.dask.common import extract_ddf_partitions, \
    to_dask_df, sparse_df_to_cp

from dask.distributed import default_client


class MultinomialNB(object):

    def __init__(self, client=None):

        self.client_ = client if client is not None else default_client()
        self.model_ = None

    @staticmethod
    def _fit(X, y, label_binarizer, n_classes, n_features):

        # X is a 3-column cudf
        model = MNB()
        for x, y in zip(X, y):

            x_cp = sparse_df_to_cp(x, x.shape[0], n_features)
            y_cp = sparse_df_to_cp(y, y.shape[0], n_classes)

            print("X=" + str(x_cp.shape))

            model.partial_fit(x_cp,
                              y_cp,
                              classes=label_binarizer.classes_,
                              _sparse_labels=True)

        return model.class_count_, model.feature_count_

    @staticmethod
    def _predict(model, X):
        return [model.predict(sparse_df_to_cp(x)) for x in X]

    def fit(self, X, y, n_features):

        self.label_binarizer_ = LabelBinarizer(client=self.client_, sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)

        x_worker_parts = self.client_.sync(extract_ddf_partitions, X)
        y_worker_parts = self.client_.sync(extract_ddf_partitions, Y)

        print("Counting!")

        n_classes = len(self.label_binarizer_.classes_)

        counts = self.client_.compute([self.client_.submit(
            MultinomialNB._fit,
            p,
            y_worker_parts[w],
            self.label_binarizer_.model,
            n_classes,
            n_features
        ) for w, p in x_worker_parts.items()], sync=True)

        print("Done!")

        self.model_ = MNB()
        self.model_.classes_ = self.label_binarizer_.classes_
        self.model_.n_classes = self.label_binarizer_.classes_.shape[0]
        self.model_.n_features = X.shape[1]

        self.model_.class_count_ = cp.zeros(n_classes, dtype=cp.float32)
        self.model_.feature_count_ = cp.zeros((n_classes, n_features),
                                       dtype=cp.float32)

        print("COUNTS: "+ str(counts))

        for class_count_, feature_count_ in counts:
            self.model_.class_count_ += class_count_
            self.model_.feature_count_ += feature_count_

    def predict(self, X):
        x_worker_parts = self.client_.sync(extract_ddf_partitions, X)

        preds = [self.client_.submit(
            MultinomialNB._predict,
            self.model_,
            p
        ) for w, p in x_worker_parts.items()]

        return to_dask_df(preds)
