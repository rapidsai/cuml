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
import cuml

import sklearn.cluster, sklearn.neighbors, sklearn.ensemble, sklearn
from sklearn import metrics
import cuml.metrics
import umap
import numpy as np

class AlgorithmPair():
    """
    Wraps a cuML algorithm and (optionally) a cpu-based algorithm
    (typically scikit-learn, but does not need to be as long as it offers
    `fit` and `predict` or `transform` methods).
    Provides mechanisms to run each version with default arguments.
    If no CPU-based version of the algorithm is available, pass None for the
    cpu_class when instantiating
    """
    def __init__(self,
                 cpu_class,
                 cuml_class,
                 shared_args,
                 cuml_args={},
                 cpu_args={},
                 name=None,
                 accepts_labels=True,
                 data_prep_hook=None,
                 accuracy_function=None):
        """
        Parameters
        ----------
        cpu_class : class
           Class for CPU version of algorithm. Set to None if not available.
        cuml_class : class
           Class for cuML algorithm
        shared_args : dict
           Arguments passed to both implementations's initializer
        cuml_args : dict
           Arguments *only* passed to cuml's initializer
        cpu_args dict
           Arguments *only* passed to sklearn's initializer
        accepts_labels : boolean
           If True, the fit methods expects both X and y
           inputs. Otherwise, it expects only an X input.
        data_prep_hook : function (data -> data)
           Optional function to run on input data before passing to fit
        accuracy_function : function (y_test, y_pred)
           Function that returns a scalar representing accuracy
        """
        if name:
            self.name = name
        else:
            self.name = cuml_class.__name__
        self.accepts_labels = accepts_labels
        self.cpu_class = cpu_class
        self.cuml_class = cuml_class
        self.shared_args = shared_args
        self.cuml_args = cuml_args
        self.cpu_args = cpu_args
        self.data_prep_hook = data_prep_hook
        self.accuracy_function = accuracy_function

    def __str__(self):
        return "AlgoPair:%s" % (self.name)

    def run_cpu(self, data, **override_args):
        """Runs the cpu-based algorithm's fit method on specified data"""
        if self.cpu_class is None:
            raise ValueError("No CPU implementation for %s" % self.name)
        all_args = {**self.shared_args, **self.cpu_args}
        all_args = {**all_args, **override_args}

        cpu_obj = self.cpu_class(**all_args)
        if self.data_prep_hook:
            data = self.data_prep_hook(data)
        if self.accepts_labels:
            cpu_obj.fit(data[0], data[1])
        else:
            cpu_obj.fit(data[0])

        return cpu_obj

    def run_cuml(self, data, **override_args):
        """Runs the cuml-based algorithm's fit method on specified data"""
        all_args = {**self.shared_args, **self.cuml_args}
        all_args = {**all_args, **override_args}

        cuml_obj = self.cuml_class(**all_args)
        if self.data_prep_hook:
            data = self.data_prep_hook(data)
        if self.accepts_labels:
            cuml_obj.fit(data[0], data[1])
        else:
            cuml_obj.fit(data[0])

        return cuml_obj

def _labels_to_int_hook(data):
    """Helper function converting labels to int32"""
    return (data[0], data[1].astype(np.int32))

def all_algorithms():
    """Returns all defined AlgorithmPair objects"""
    return [
        AlgorithmPair(
            sklearn.cluster.KMeans,
            cuml.cluster.KMeans,
            shared_args=dict(init='random',
                             n_clusters=8,
                             max_iter=300),
            name='KMeans',
            accepts_labels=False,
            accuracy_function=metrics.homogeneity_score),
        AlgorithmPair(
            sklearn.decomposition.PCA,
            cuml.PCA,
            shared_args=dict(n_components=10),
            name='PCA',
            accepts_labels=False),
        AlgorithmPair(
            sklearn.neighbors.NearestNeighbors,
            cuml.neighbors.NearestNeighbors,
            shared_args=dict(n_neighbors=1024),
            cpu_args=dict(algorithm='brute'),
            cuml_args=dict(n_gpus=1),
            name='NearestNeighbors',
            accepts_labels=False),
        AlgorithmPair(
            sklearn.cluster.DBSCAN,
            cuml.DBSCAN,
            shared_args=dict(eps=3, min_samples=2),
            cpu_args=dict(algorithm='brute'),
            name='DBSCAN',
            accepts_labels=False
        ),
        AlgorithmPair(
            sklearn.linear_model.LinearRegression,
            cuml.linear_model.LinearRegression,
            shared_args={},
            name='LinearRegression',
            accepts_labels=True,
            accuracy_function=metrics.r2_score),
        AlgorithmPair(
            sklearn.linear_model.LogisticRegression,
            cuml.linear_model.LogisticRegression,
            shared_args={},
            name='LogisticRegression',
            accepts_labels=True,
            accuracy_function=metrics.accuracy_score),
        AlgorithmPair(
            sklearn.ensemble.RandomForestClassifier,
            cuml.ensemble.RandomForestClassifier,
            shared_args={'max_features': 1.0, 'n_estimators': 100},
            name='RandomForestClassifier',
            accepts_labels=True,
            data_prep_hook=_labels_to_int_hook,
            accuracy_function=metrics.accuracy_score),
        AlgorithmPair(
            sklearn.ensemble.RandomForestRegressor,
            cuml.ensemble.RandomForestRegressor,
            shared_args={'max_features': 1.0, 'n_estimators': 100},
            name='RandomForestRegressor',
            accepts_labels=True,
            accuracy_function=metrics.r2_score),
        AlgorithmPair(
            umap.UMAP,
            cuml.manifold.UMAP,
            shared_args=dict(n_neighbors=5, n_epochs=500),
            name='UMAP',
            accepts_labels=False,
            accuracy_function=cuml.metrics.trustworthiness),
        AlgorithmPair(
            None,
            cuml.linear_model.MBSGDClassifier,
            shared_args={},
            name='MBSGDClassifier',
            accepts_labels=True,
            accuracy_function=cuml.metrics.accuracy_score),
    ]


def algorithm_by_name(name):
    """Returns the algorithm pair with the name 'name' (case-insensitive)"""
    algos = all_algorithms()
    return next((a for a in algos if a.name.lower() == name.lower()),
                None)
