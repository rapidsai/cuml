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

import sklearn
import sklearn.cluster
import sklearn.neighbors
import sklearn.ensemble
import sklearn.random_projection
from sklearn import metrics
import cuml.metrics
import cuml.decomposition
import umap
import numpy as np
import os

from cuml.utils.import_utils import has_xgboost, has_treelite
if has_xgboost():
    import xgboost as xgb
if has_treelite():
    import treelite as tl
    import treelite.runtime

class AlgorithmPair:
    """
    Wraps a cuML algorithm and (optionally) a cpu-based algorithm
    (typically scikit-learn, but does not need to be as long as it offers
    `fit` and `predict` or `transform` methods).
    Provides mechanisms to run each version with default arguments.
    If no CPU-based version of the algorithm is available, pass None for the
    cpu_class when instantiating
    """

    def __init__(
        self,
        cpu_class,
        cuml_class,
        shared_args,
        cuml_args={},
        cpu_args={},
        name=None,
        accepts_labels=True,
        data_prep_hook=None,
        accuracy_function=None,
    ):
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


class AlgorithmFIL:
    """
    Wraps a cuML algorithm and other tree inference related algorithms 
    (scikit-learn, xgboost-cpu, xgboost-gpu, treelite).
    Provides mechanisms to run each version with default arguments.
    If no CPU-based version of the algorithm is available, pass None for the
    cpu_class when instantiating
    """

    def __init__(
        self,
        xgb_class,
        skl_class,
        tl_class, 
        cuml_class,
        shared_args,
        cuml_args={},
        cpu_args={},
        name=None,
        data_prep_hook=None,
        accuracy_function=None,
    ):
        """
        Parameters
        ----------
        xgb_class : class
           Class for XGBoost version of algorithm. Set to None if not available.
        skl_class : class
           Class for sklearn version of algorithm. Set to None if not available.
        tl_class : class
           Class for treelite version of algorithm. Set to None if not available.           
        cuml_class : class
           Class for cuML algorithm
        shared_args : dict
           Arguments passed to both implementations's initializer
        cuml_args : dict
           Arguments *only* passed to cuml's initializer
        cpu_args dict
           Arguments *only* passed to sklearn's initializer
        data_prep_hook : function (data -> data)
           Optional function to run on input data before passing to fit
        accuracy_function : function (y_test, y_pred)
           Function that returns a scalar representing accuracy
        """

        XGOBOOST_MODEL_PATH = './'
        if name:
            self.name = name
        else:
            self.name = cuml_class.__name__
        self.xgb_class = xgb_class 
        self.skl_class = skl_class
        self.tl_class = tl_class
        self.cuml_class = cuml_class
        self.shared_args = shared_args
        self.cpu_args = cpu_args
        self.cuml_args = cuml_args
        self.data_prep_hook = data_prep_hook
        self.accuracy_function = accuracy_function
        self.model_path = os.path.join(XGOBOOST_MODEL_PATH, "xgb.model")
        self.tl_path = os.path.join(XGOBOOST_MODEL_PATH, "treelite_model.so")

    def __str__(self):
        return "AlgoPair:%s" % (self.name)

    def prepare_xgboost(self, data, **override_args):
        """Prepares the xgboost algorithm for FIL benchmarking"""
        if self.xgb_class is None:
            raise ValueError("No xgboost implementation for %s" % self.name)
        
        if self.data_prep_hook:
            data = self.data_prep_hook(data)
        xgb_train_args = {"silent": 1, "eval_metric": "error", "objective": "binary:logistic"}
        all_args = {**self.shared_args, **xgb_train_args}
        all_args = {**all_args, **override_args}
        dtrain = xgb.DMatrix(data[2], label=data[3])
        self.xgb_tree = self.xgb_class.train(all_args, dtrain, all_args["n_estimators"])
        self.xgb_tree.save_model(self.model_path)
        self.dvalidation = xgb.DMatrix(data[0], label=data[1])
        
    def prepare_sklearn(self, data, **override_args):
        """Prepares the sklearn algorithm for FIL benchmarking"""
        if self.skl_class is None:
            raise ValueError("No sklearn implementation for %s" % self.name)        
        
        if self.data_prep_hook:
            data = self.data_prep_hook(data)
        all_args = {**self.shared_args, **self.cpu_args}
        all_args = {**all_args, **override_args}
        self.skl_tree = self.skl_class(**all_args)
        self.skl_tree.fit(data[2], data[3])
    
    def prepare_treelite(self, data, **override_args):
        """Prepares the treelite algorithm for FIL benchmarking"""
        """This function depends on prepare_xgboost function"""
        if self.tl_class is None:
            raise ValueError("No treelite implementation for %s" % self.name)          
        
        model = self.tl_class.Model.from_xgboost(self.xgb_tree)
        if os.path.exists(self.tl_path):
            os.remove(self.tl_path)
        model.export_lib(toolchain="gcc", libpath=self.tl_path, params={"parallel_comp": 0}, verbose=False)
        self.tl_tree = None # this is needed, as predictor cannot be overwrite automatically 
        self.tl_tree = treelite.runtime.Predictor(self.tl_path, verbose=False)
        self.tl_batch = treelite.runtime.Batch.from_npy2d(data[0])

    def prepare_cuml(self, data, **override_args):
        """Prepares the cuml algorithm for FIL benchmarking"""
        """This function depends on prepare_xgboost function"""
        if self.cuml_class is None:
            raise ValueError("No cuml implementation for %s" % self.name)                  
        all_args = {**self.shared_args, **self.cuml_args}
        all_args = {**all_args, **override_args}
        self.cuml_tree = cuml.ForestInference.load(self.model_path, all_args)

    def run_sklearn(self, data):
        """Runs the sklearn algorithm's predict method on specified data"""
        if self.skl_class is None:
            raise ValueError("No sklearn implementation for %s" % self.name)

        if self.data_prep_hook:
            data = self.data_prep_hook(data)
        skl_pred = self.skl_tree.predict(data[0])

        return skl_pred

    def run_xgboost_cpu(self, data):
        """Runs the xgboost_cpu algorithm's predict method on specified data"""
        if self.xgb_class is None:
            raise ValueError("No xgboost implementation for %s" % self.name)
        base_args = {"predictor": "cpu_predictor", "n_gpus": 0}

        self.xgb_tree.set_param(base_args)
        if self.data_prep_hook:
            data = self.data_prep_hook(data)
        xgb_cpu_pred = self.xgb_tree.predict(self.dvalidation)

        return xgb_cpu_pred

    def run_xgboost_gpu(self, data):
        """Runs the xgboost_gpu algorithm's predict method on specified data"""
        if self.xgb_class is None:
            raise ValueError("No xgboost implementation for %s" % self.name)
        base_args = {"predictor": "gpu_predictor", "n_gpus": 1, "gpu_id": 0}

        self.xgb_tree.set_param(base_args)
        if self.data_prep_hook:
            data = self.data_prep_hook(data)
        xgb_gpu_pred = self.xgb_tree.predict(self.dvalidation)

        return xgb_gpu_pred

    def run_treelite(self, data):      
        """Runs the treelite algorithm's predict method on specified data"""
        if self.tl_class is None:
            raise ValueError("No treelite implementation for %s" % self.name)

        if self.data_prep_hook:
            data = self.data_prep_hook(data)
        tl_pred = self.tl_tree.predict(self.tl_batch)

        return tl_pred

    def run_cuml(self, data):
        """Runs the cuml algorithm's predict method on specified data"""
        if self.cuml_class is None:
            raise ValueError("No cuml implementation for %s" % self.name)

        if self.data_prep_hook:
            data = self.data_prep_hook(data)
        cuml_pred = np.asarray(self.cuml_tree.predict(data[0]))

        return cuml_pred


def _labels_to_int_hook(data):
    """Helper function converting labels to int32"""
    return (data[0], data[1].astype(np.int32))


def all_algorithms():
    """Returns all defined AlgorithmPair objects"""
    return [
        AlgorithmPair(
            sklearn.cluster.KMeans,
            cuml.cluster.KMeans,
            shared_args=dict(init="random", n_clusters=8, max_iter=300),
            name="KMeans",
            accepts_labels=False,
            accuracy_function=metrics.homogeneity_score,
        ),
        AlgorithmPair(
            sklearn.decomposition.PCA,
            cuml.PCA,
            shared_args=dict(n_components=10),
            name="PCA",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.decomposition.truncated_svd.TruncatedSVD,
            cuml.decomposition.tsvd.TruncatedSVD,
            shared_args=dict(n_components=10),
            name="tSVD",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.random_projection.GaussianRandomProjection,
            cuml.random_projection.GaussianRandomProjection,
            shared_args=dict(n_components=10),
            name="GaussianRandomProjection",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.neighbors.NearestNeighbors,
            cuml.neighbors.NearestNeighbors,
            shared_args=dict(n_neighbors=1024),
            cpu_args=dict(algorithm="brute"),
            cuml_args={},
            name="NearestNeighbors",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.cluster.DBSCAN,
            cuml.DBSCAN,
            shared_args=dict(eps=3, min_samples=2),
            cpu_args=dict(algorithm="brute"),
            name="DBSCAN",
            accepts_labels=False,
        ),
        AlgorithmPair(
            sklearn.linear_model.LinearRegression,
            cuml.linear_model.LinearRegression,
            shared_args={},
            name="LinearRegression",
            accepts_labels=True,
            accuracy_function=metrics.r2_score,
        ),
        AlgorithmPair(
            sklearn.linear_model.ElasticNet,
            cuml.linear_model.ElasticNet,
            shared_args={"alpha": 0.1, "l1_ratio": 0.5},
            name="ElasticNet",
            accepts_labels=True,
            accuracy_function=metrics.r2_score,
        ),
        AlgorithmPair(
            sklearn.linear_model.Lasso,
            cuml.linear_model.Lasso,
            shared_args={},
            name="Lasso",
            accepts_labels=True,
            accuracy_function=metrics.r2_score,
        ),
        AlgorithmPair(
            sklearn.linear_model.LogisticRegression,
            cuml.linear_model.LogisticRegression,
            shared_args={},
            name="LogisticRegression",
            accepts_labels=True,
            accuracy_function=metrics.accuracy_score,
        ),
        AlgorithmPair(
            sklearn.ensemble.RandomForestClassifier,
            cuml.ensemble.RandomForestClassifier,
            shared_args={"max_features": 1.0, "n_estimators": 10},
            name="RandomForestClassifier",
            accepts_labels=True,
            data_prep_hook=_labels_to_int_hook,
            accuracy_function=metrics.accuracy_score,
        ),
        AlgorithmPair(
            sklearn.ensemble.RandomForestRegressor,
            cuml.ensemble.RandomForestRegressor,
            shared_args={"max_features": 1.0, "n_estimators": 10},
            name="RandomForestRegressor",
            accepts_labels=True,
            accuracy_function=metrics.r2_score,
        ),
        AlgorithmPair(
            sklearn.manifold.TSNE,
            cuml.manifold.TSNE,
            shared_args=dict(),
            name="TSNE",
            accepts_labels=False,
        ),
        AlgorithmPair(
            umap.UMAP,
            cuml.manifold.UMAP,
            shared_args=dict(n_neighbors=5, n_epochs=500),
            name="UMAP",
            accepts_labels=False,
            accuracy_function=cuml.metrics.trustworthiness,
        ),
        AlgorithmPair(
            None,
            cuml.linear_model.MBSGDClassifier,
            shared_args={},
            cuml_args=dict(eta0=0.005, epochs=100),
            name="MBSGDClassifier",
            accepts_labels=True,
            accuracy_function=cuml.metrics.accuracy_score,
        ),
        AlgorithmFIL(
            xgb, 
            sklearn.ensemble.RandomForestClassifier,
            tl,
            cuml.ForestInference,
            shared_args={"max_depth": 10, "n_estimators": 10},
            cuml_args=dict(algo="BATCH_TREE_REORG", output_class=True, threshold=0.50),
            name="FIL",
            accuracy_function=metrics.accuracy_score,
        ),
    ]


def algorithm_by_name(name):
    """Returns the algorithm pair with the name 'name' (case-insensitive)"""
    algos = all_algorithms()
    return next((a for a in algos if a.name.lower() == name.lower()), None)
