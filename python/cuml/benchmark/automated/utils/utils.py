#
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

try:
    from rapids_pytest_benchmark import setFixtureParamNames
except ImportError:
    print("\n\nWARNING: rapids_pytest_benchmark is not installed, "
          "falling back to pytest_benchmark fixtures.\n")

    # if rapids_pytest_benchmark is not available, just perfrom time-only
    # benchmarking and replace the util functions with nops
    import pytest_benchmark
    gpubenchmark = pytest_benchmark.plugin.benchmark

    def setFixtureParamNames(*args, **kwargs):
        pass

import os
import json
import time
import itertools as it
import numpy as np
import cupy as cp
import cudf

import pytest
from cuml.benchmark import datagen, algorithms
from cuml.benchmark.nvtx_benchmark import Profiler
import dask.array as da
import dask.dataframe as df
from copy import copy

from cuml.benchmark.bench_helper_funcs import pass_func, fit, predict, \
                                              transform, kneighbors


def distribute(client, data):
    if data is not None:
        n_rows = data.shape[0]
        n_workers = len(client.scheduler_info()['workers'])
        if isinstance(data, (np.ndarray, cp.ndarray)):
            dask_array = da.from_array(x=data,
                                       chunks={0: n_rows // n_workers, 1: -1})
            return dask_array
        elif isinstance(data, (cudf.DataFrame, cudf.Series)):
            dask_df = df.from_pandas(data,
                                     chunksize=n_rows // n_workers)
            return dask_df
        else:
            raise ValueError('Could not distribute data')


def nvtx_profiling(algo_name, data_kwargs, setup_kwargs,
                   training_kwargs, inference_kwargs):
    dataset_type = data_kwargs['dataset_type']
    n_samples = data_kwargs['n_samples']
    n_features = data_kwargs['n_features']
    dataset_format = (data_kwargs['dataset_format'] if 'dataset_format'
                      in data_kwargs else 'cupy')

    data_kwargs_edited = copy(data_kwargs)
    for param in ['dataset_type', 'n_samples', 'n_features',
                  'dataset_format']:
        data_kwargs_edited.pop(param, None)

    path = os.path.dirname(os.path.realpath(__file__))
    command = """
    python {path}/auto_nvtx_bench.py
        --algo_name {algo_name}
        --dataset_type {dataset_type}
        --n_samples {n_samples}
        --n_features {n_features}
        --dataset_format {dataset_format}
        --data_kwargs {data_kwargs}
        --setup_kwargs {setup_kwargs}
        --training_kwargs {training_kwargs}
        --inference_kwargs {inference_kwargs}
    """.format(path=path,
               algo_name=algo_name,
               dataset_type=dataset_type,
               n_samples=n_samples,
               n_features=n_features,
               dataset_format=dataset_format,
               data_kwargs=json.dumps(data_kwargs_edited,
                                      separators=(',', ':')),
               setup_kwargs=json.dumps(setup_kwargs,
                                       separators=(',', ':')),
               training_kwargs=json.dumps(training_kwargs,
                                          separators=(',', ':')),
               inference_kwargs=json.dumps(inference_kwargs,
                                           separators=(',', ':')))
    command = command.replace('\n', '').replace('\t', ' ')
    command = ' '.join(command.split())

    print('\n\n' + '\033[96m' + '=x'*48)
    print('=x'*20 + ' NVTX BENCHMARK ' + '=x'*20)

    profiler = Profiler()
    profiler.profile(command)

    print('=x'*48)
    print('=x'*48 + '\033[0m' + '\n')


def cpu_bench(algo, bench_step, dataset, inference_args, cpu_setup):
    if algo.cpu_class is None:
        return

    t = time.process_time()
    if bench_step == 'training':
        algo.run_cpu(dataset, **cpu_setup)
    elif bench_step == 'inference':
        algo.run_cpu(dataset, **inference_args, **cpu_setup)
    elapsed_time = time.process_time() - t

    print('\n' + '\033[33m' + '=x'*20 + '  CPU BENCHMARK ' + '=x'*20)
    print(algo.name + ' : ' + str(algo.cpu_class))
    print('\tbench_function: ' + str(algo.bench_func))
    print('\truntime: ' + str(elapsed_time))
    print('=x'*48 + '\033[0m' + '\n')


def setup_bench(platform, algo, bench_step, dataset,
                setup_kwargs, training_kwargs):
    if platform == 'cuml':
        setup = algo.setup_cuml(dataset, **setup_kwargs)
    elif platform == 'cpu':
        setup = algo.setup_cpu(dataset, **setup_kwargs)

    if bench_step == 'training':
        if hasattr(algo.cuml_class, 'fit'):
            algo.bench_func = fit
        elif algo.setup_cuml_func:
            algo.bench_func = pass_func
        else:
            raise ValueError('Training function not found')
    elif bench_step == 'inference':
        if hasattr(algo.cuml_class, 'fit'):
            algo.bench_func = fit
        elif algo.setup_cuml_func:
            algo.bench_func = pass_func
        else:
            raise ValueError('Training function not found')

        if platform == 'cuml':
            setup['cuml_setup_result'] = \
                algo.run_cuml(dataset, **training_kwargs, **setup)
        elif platform == 'cpu':
            setup['cpu_setup_result'] = \
                algo.run_cpu(dataset, **training_kwargs, **setup)

        if hasattr(algo.cuml_class, 'predict'):
            algo.bench_func = predict
        elif hasattr(algo.cuml_class, 'transform'):
            algo.bench_func = transform
        elif hasattr(algo.cuml_class, 'kneighbors'):
            algo.bench_func = kneighbors
        else:
            pytest.xfail('Inference function not found')
    else:
        raise ValueError('bench_func should be either training or inference')
    return setup


def _benchmark_algo(
    benchmarker,
    algo_name,
    bench_step,
    dataset,
    setup_kwargs={},
    training_kwargs={},
    inference_kwargs={},
    client=None
):
    """Simplest benchmark wrapper to time algorithm 'name' on dataset"""

    dataset, data_kwargs = dataset

    MNMG_mode = client is not None
    if MNMG_mode:  # if MNMG => scatter data
        setup_kwargs['client'] = client
        if algo_name != 'MNMG.DBSCAN':  # exception with MNMG DBSCAN
            dataset = [distribute(client, d) for d in dataset]

    algo = algorithms.algorithm_by_name(algo_name)
    cuml_setup = setup_bench('cuml', algo, bench_step, dataset,
                             setup_kwargs, training_kwargs)

    if bench_step == 'training':
        benchmarker(algo.run_cuml, dataset, **training_kwargs, **cuml_setup)
    elif bench_step == 'inference':
        benchmarker(algo.run_cuml, dataset, **inference_kwargs, **cuml_setup)

    if not MNMG_mode:  # if SG => run NVTX benchmark and CPU bench
        if algo.cpu_class:
            cpu_dataset = datagen._convert_to_numpy(dataset)
            cpu_setup = setup_bench('cpu', algo, bench_step, cpu_dataset,
                                    setup_kwargs, training_kwargs)
            cpu_bench(algo, bench_step, cpu_dataset, inference_kwargs,
                      cpu_setup)

        if bench_step == 'inference':
            nvtx_profiling(algo_name, data_kwargs, setup_kwargs,
                           training_kwargs, inference_kwargs)


def fixture_generation_helper(params):
    param_names = sorted(params)
    param_combis = list(it.product(*(params[param_name]
                                     for param_name in param_names)))
    ids = ['-'.join(map(str, param_combi)) for param_combi in param_combis]
    param_combis = [dict(zip(param_names, param_combi))
                    for param_combi in param_combis]
    return {
        'scope': 'session',
        'params': param_combis,
        'ids': ids
    }


@pytest.fixture(scope='session',
                params=['training', 'inference'],
                ids=['training', 'inference'])
def bench_step(request):
    return request.param
