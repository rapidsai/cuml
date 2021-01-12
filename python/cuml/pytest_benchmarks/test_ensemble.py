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

"""Benchmark integrations described in blog:

using pytest benchmarks and rapids-pytest-benchmark extension
Requires pytest-benchmark and rapids-pytest-benchmark which are not currently installed by default.
"""

from cuml.benchmark.algorithms import AlgorithmPair
from cuml.common.import_utils import has_pytest_benchmark

import pytest

from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import AdaBoostRegressor, VotingClassifier, StackingClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score

import cuml

#
# Testing utilities
#
def _ensemble_benchmark_algo( benchmark,
                              algorithm_pair,
                              datagen=make_regression,
                              n_samples=10000,
                              n_features=100,
                              n_estimators=10,
                              input_type='numpy',
                              data_kwargs={},
                              algo_args={}):
    """Execute the benchmark for cpu and gpu, return the Xspeedup as cpu/gpu"""
    algo = algorithm_pair
    #pack data in an extra tuple because AlgorithmPair wants to pop specific members from data
    #TODO: figure out how to make the abstraction more flexible so that this hack is unnecessary
    #      search here for date[0][n] subscripts those are referencing this nested element
    data = (datagen( n_samples=n_samples, n_features=n_features, n_informative=n_features, 
                      **data_kwargs ), None)
    
    benchmark(algo.run_cuml, data, **algo_args)
    #benchmark(algo.run_cpu, data, **algo_args)


def fit_and_score( clf, data, *argc, **kwargs ):
    print( f"Data has length {len(data)}")
    clf.fit(data[0],data[1])
    return clf.score(data[-2],data[-1])


def fit_and_predict( clf, data ):
    print( f"Data has length {len(data)}")
    clf.fit(data[0],data[1])
    return clf.predict(data[0])


def dg_train_and_test( data ):
    nudata = train_test_split( data[0], data[1], random_state=42 )
    print( f"Train and test got data of length {len(data)} and produced data of length {len(nudata)}")
    return nudata

#
# ADABoostRegressor
#

adaboostregressor = AlgorithmPair( cpu_class=SVR, 
                                   cuml_class=AdaBoostRegressor,
                                   cuml_args={ 'base_estimator':SVR(),
                                               'n_estimators':10,
                                               'random_state':12},
                                   cpu_args={},
                                   shared_args={},
                                   name="BoostedSVR",
                                   accepts_labels=False,
                                   cpu_data_prep_hook=dg_train_and_test,
                                   cuml_data_prep_hook=dg_train_and_test,
                                   accuracy_function=None,
                                   bench_func=fit_and_score,
                                   setup_cpu_func=None,
                                   setup_cuml_func=None )


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_samples', [500, 50000])
@pytest.mark.parametrize('n_features', [5, 50])
def test_ensemble_boost_svr(benchmark, n_samples, n_features):
    _ensemble_benchmark_algo(benchmark, adaboostregressor, 
                             datagen=make_regression,
                             n_samples=n_samples, n_features=n_features,
                             data_kwargs={'random_state':12, 'noise':100})


#
# Voting Classifier
#

votingclassifier = AlgorithmPair( cpu_class=VotingClassifier,
                                  cuml_class=VotingClassifier,
                                  cuml_args={ 'estimators':[ ('lr', cuml.linear_model.LogisticRegression(C=1)), 
                                                             ('svc', cuml.svm.SVC())]},
                                  cpu_args={ 'estimators':[('lr', LogisticRegression(C=1, n_jobs=-1)), 
                                                           ('svc', SVC())],
                                             'n_jobs':-1},
                                  shared_args={'voting':'hard',
                                               'weights':[0.2, 0.8]},
                                  name="VotingClassifier",
                                  accepts_labels=False,
                                  cpu_data_prep_hook=None,
                                  cuml_data_prep_hook=None,
                                  accuracy_function=None,
                                  bench_func=fit_and_score,
                                  setup_cpu_func=None,
                                  setup_cuml_func=None )


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_samples', [500, 50000])
@pytest.mark.parametrize('n_features', [5, 50])
def test_ensemble_voting_classifier(benchmark, n_samples, n_features):
    _ensemble_benchmark_algo(benchmark, votingclassifier,
                             datagen=make_classification,
                             n_samples=n_samples, n_features=n_features,
                             data_kwargs={'n_redundant':0, 'class_sep':0.75, 'n_classes':2})

#
# Stacking Classifier
#

stackingclassifier = AlgorithmPair( cpu_class=StackingClassifier,
                                    cuml_class=StackingClassifier,
                                    cuml_args={ 'estimators':[ ('rf', cuml.linear_model.LogisticRegression(C=1)), 
                                                               ('svc', cuml.svm.SVC())]},
                                    cpu_args={ 'estimators':[('rf', LogisticRegression(C=1, n_jobs=-1)), 
                                                             ('svc', SVC())],
                                               'n_jobs':-1},
                                    shared_args={'stack_method':"predict"},
                                    name="StackingClassifier",
                                    accepts_labels=False,
                                    cpu_data_prep_hook=None,
                                    cuml_data_prep_hook=None,
                                    accuracy_function=None,
                                    bench_func=fit_and_score,
                                    setup_cpu_func=None,
                                    setup_cuml_func=None )


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_samples', [500, 50000])
@pytest.mark.parametrize('n_features', [5, 50])
def test_ensemble_stacking_classifier(benchmark, n_samples, n_features):
    _ensemble_benchmark_algo(benchmark, stackingclassifier,
                             datagen=make_classification,
                             n_samples=n_samples, n_features=n_features,
                             data_kwargs={'n_redundant':0, 'class_sep':0.75, 'n_classes':2})


#
# Bagging Regressor
#
baggingregressor =  AlgorithmPair( cpu_class=BaggingRegressor, 
                                   cuml_class=BaggingRegressor,
                                   cuml_args={ 'base_estimator':cuml.neighbors.KNeighborsRegressor()},
                                   cpu_args={ 'base_estimator':KNeighborsRegressor(n_jobs=-1),
                                             'verbose':2,
                                             'n_jobs':-1},
                                   shared_args={'n_estimators':20,
                                                'random_state':12},
                                   name="BaggingRegressor",
                                   accepts_labels=False,
                                   accuracy_function=None,
                                   bench_func=fit_and_predict,
                                   setup_cpu_func=None,
                                   setup_cuml_func=None )


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_samples', [500, 250000])
@pytest.mark.parametrize('n_features', [5, 20])
def test_ensemble_bagging_regressor(benchmark, n_samples, n_features):
    _ensemble_benchmark_algo(benchmark, baggingregressor,
                             datagen=make_regression,
                             n_samples=n_samples, n_features=n_features,
                             data_kwargs={'random_state':12, 'noise':200})


#
# Boosted Regression
#

boostedregressor =  AlgorithmPair( cpu_class=AdaBoostRegressor, 
                                   cuml_class=AdaBoostRegressor,
                                   cuml_args={ 'base_estimator':cuml.svm.SVR()},
                                   cpu_args={ 'base_estimator':SVR()},
                                   shared_args={'n_estimators':10,
                                                'random_state':12},
                                   name="BoostedRegressor",
                                   accepts_labels=False,
                                   accuracy_function=None,
                                   bench_func=fit_and_score,
                                   setup_cpu_func=None,
                                   setup_cuml_func=None )


@pytest.mark.skipif(not has_pytest_benchmark(),
                    reason='pytest-benchmark missing')
@pytest.mark.parametrize('n_samples', [50, 20000])
@pytest.mark.parametrize('n_features', [1, 10])
def test_ensemble_boosted_regressor(benchmark, n_samples, n_features):
    _ensemble_benchmark_algo(benchmark, boostedregressor,
                             datagen=make_regression,
                             n_samples=n_samples, n_features=n_features,
                             data_kwargs={'random_state':12, 'noise':200})

