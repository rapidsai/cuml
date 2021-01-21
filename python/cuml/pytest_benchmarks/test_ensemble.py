# Copyright (c) 2021, NVIDIA CORPORATION.
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

"""Benchmark integrations described in blog:
   https://medium.com/rapids-ai/100x-faster-machine-learning-model-ensembling-with-rapids-cuml-and-scikit-learn-meta-estimators-d869788ee6b1
"""

from cuml.test.utils import unit_param, stress_param
from cuml.benchmark.algorithms import AlgorithmPair
from cuml.common.import_utils import has_pytest_benchmark

import pytest

from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import AdaBoostRegressor, VotingClassifier, StackingClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, SVC

import functools

import cuml

#
# Testing utilities
#


@functools.lru_cache(maxsize=8)
def gendata(datafunc, n_samples, n_features, **data_kwargs):
    return datafunc(n_samples=n_samples, n_features=n_features, **data_kwargs)


def _ensemble_benchmark_algo(gpubenchmark,
                             algorithm_pair,
                             dataset_func=make_regression,
                             n_samples=10000,
                             n_features=100,
                             data_kwargs={},
                             algo_type='gpu'):
    """Execute the benchmark for cpu and gpu, return the Xspeedup as cpu/gpu"""

    data = gendata(dataset_func, n_samples, n_features, **data_kwargs)

    if algo_type == 'cpu':
        gpubenchmark(algorithm_pair.run_cpu, (data, None))
    else:
        gpubenchmark(algorithm_pair.run_cuml, (data, None))


def fit_and_score(clf, data, *argc, **kwargs):
    clf.fit(data[0], data[1])
    return clf.score(data[-2], data[-1])


def fit_and_predict(clf, data):
    clf.fit(data[0], data[1])
    return clf.predict(data[0])

#
# Voting Classifier
#


votingclassifier = AlgorithmPair(
    cpu_class=VotingClassifier,
    cuml_class=VotingClassifier,
    cuml_args={'estimators':
               [('lr', cuml.linear_model.LogisticRegression(C=1)),
                ('svc', cuml.svm.SVC())]},
    cpu_args={'estimators':
              [('lr', LogisticRegression(C=1, n_jobs=-1)),
               ('svc', SVC())],
              'n_jobs': -1},
    shared_args={'voting': 'hard',
                 'weights': [0.2, 0.8]},
    name="VotingClassifier",
    accepts_labels=False,
    cpu_data_prep_hook=None,
    cuml_data_prep_hook=None,
    accuracy_function=None,
    bench_func=fit_and_score,
    setup_cpu_func=None,
    setup_cuml_func=None)


@pytest.mark.integration
@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_samples', [unit_param(500), stress_param(50000)])
@pytest.mark.parametrize('n_features', [5, 50])
@pytest.mark.parametrize('algotype', ['gpu', 'cpu'])
def test_ensemble_voting_classifier(gpubenchmark, n_samples,
                                    n_features, algotype):
    _ensemble_benchmark_algo(gpubenchmark, votingclassifier,
                             dataset_func=make_classification,
                             n_samples=n_samples, n_features=n_features,
                             data_kwargs={'n_redundant': 0,
                                          'class_sep': 0.75,
                                          'n_classes': 2},
                             algo_type=algotype)

#
# Stacking Classifier
#


stackingclassifier = AlgorithmPair(
    cpu_class=StackingClassifier,
    cuml_class=StackingClassifier,
    cuml_args={'estimators':
               [('rf', cuml.linear_model.LogisticRegression(C=1)),
                ('svc', cuml.svm.SVC())]},
    cpu_args={'estimators':
              [('rf', LogisticRegression(C=1, n_jobs=-1)),
               ('svc', SVC())],
              'n_jobs': -1},
    shared_args={'stack_method': "predict"},
    name="StackingClassifier",
    accepts_labels=False,
    cpu_data_prep_hook=None,
    cuml_data_prep_hook=None,
    accuracy_function=None,
    bench_func=fit_and_score,
    setup_cpu_func=None,
    setup_cuml_func=None)


@pytest.mark.integration
@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_samples', [unit_param(500), stress_param(50000)])
@pytest.mark.parametrize('n_features', [5, 50])
@pytest.mark.parametrize('algotype', ['gpu', 'cpu'])
def test_ensemble_stacking_classifier(gpubenchmark, n_samples,
                                      n_features, algotype):
    _ensemble_benchmark_algo(gpubenchmark, stackingclassifier,
                             dataset_func=make_classification,
                             n_samples=n_samples, n_features=n_features,
                             data_kwargs={'n_redundant': 0,
                                          'class_sep': 0.75,
                                          'n_classes': 2},
                             algo_type=algotype)


#
# Bagging Regressor
#
baggingregressor = AlgorithmPair(
    cpu_class=BaggingRegressor,
    cuml_class=BaggingRegressor,
    cuml_args={'base_estimator':
               cuml.neighbors.KNeighborsRegressor()},
    cpu_args={'base_estimator':
              KNeighborsRegressor(n_jobs=-1),
              'verbose': 2,
              'n_jobs': -1},
    shared_args={'n_estimators': 20,
                 'random_state': 12},
    name="BaggingRegressor",
    accepts_labels=False,
    accuracy_function=None,
    bench_func=fit_and_predict,
    setup_cpu_func=None,
    setup_cuml_func=None)


@pytest.mark.integration
@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_samples', [unit_param(500), stress_param(250000)])
@pytest.mark.parametrize('n_features', [5, 20])
@pytest.mark.parametrize('algotype', ['gpu', 'cpu'])
def test_ensemble_bagging_regressor(gpubenchmark, n_samples,
                                    n_features, algotype):
    _ensemble_benchmark_algo(gpubenchmark, baggingregressor,
                             dataset_func=make_regression,
                             n_samples=n_samples, n_features=n_features,
                             data_kwargs={'random_state': 12, 'noise': 200},
                             algo_type=algotype)


#
# Boosted Regression
#

boostedregressor = AlgorithmPair(
    cpu_class=AdaBoostRegressor,
    cuml_class=AdaBoostRegressor,
    cuml_args={'base_estimator': cuml.svm.SVR()},
    cpu_args={'base_estimator': SVR()},
    shared_args={'n_estimators': 10,
                 'random_state': 12},
    name="BoostedRegressor",
    accepts_labels=False,
    accuracy_function=None,
    bench_func=fit_and_score,
    setup_cpu_func=None,
    setup_cuml_func=None)


@pytest.mark.integration
@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_samples', [unit_param(50), stress_param(20000)])
@pytest.mark.parametrize('n_features', [1, 10])
@pytest.mark.parametrize('algotype', ['gpu', 'cpu'])
def test_ensemble_boosted_regressor(gpubenchmark, n_samples,
                                    n_features, algotype):
    _ensemble_benchmark_algo(gpubenchmark, boostedregressor,
                             dataset_func=make_regression,
                             n_samples=n_samples, n_features=n_features,
                             data_kwargs={'random_state': 12, 'noise': 200},
                             algo_type=algotype)
