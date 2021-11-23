#
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
from cuml.benchmark import datagen, algorithms
from cuml.benchmark.nvtx_benchmark import Profiler
import dask.array as da

from sklearnex import patch_sklearn
patch_sklearn()


def generate_dataset(dataset_name, n_samples, n_features,
                     input_type, data_kwargs):
    data = datagen.gen_data(
        dataset_name,
        input_type,
        n_samples=n_samples,
        n_features=n_features,
        **data_kwargs
    )
    return data


def distribute(client, array):
    if array is not None:
        n_rows = array.shape[0]
        n_workers = len(client.scheduler_info()['workers'])
        dask_array = da.from_array(array, chunks=n_rows // n_workers)
        return dask_array


def nvtx_profiling(name, dataset_name, n_samples, n_features,
                   input_type, data_kwargs, algo_args):
    path = os.path.dirname(os.path.realpath(__file__))
    command = """
    python {path}/bench_nvtx.py
        --algo {algo}
        --dataset_type {dataset_type}
        --n_samples {n_samples}
        --n_features {n_features}
        --input_type {input_type}
        --data_kwargs {data_kwargs}
        --algo_args {algo_args}
    """.format(path=path,
               algo=name,
               dataset_type=dataset_name,
               n_samples=n_samples,
               n_features=n_features,
               input_type=input_type,
               data_kwargs=json.dumps(data_kwargs, separators=(',', ':')),
               algo_args=json.dumps(algo_args, separators=(',', ':')))
    command = command.replace('\n', '').replace('\t', ' ')
    command = ' '.join(command.split())

    print('\n\n' + '\033[96m' + '=x'*48)
    print('=x'*20 + ' NVTX BENCHMARK ' + '=x'*20)

    profiler = Profiler()
    profiler.profile(command)

    print('=x'*48)
    print('=x'*48 + '\033[0m' + '\n')


def sklearn_intelex_bench(algo, data, algo_args):
    setup_overrides = algo.setup_cpu(data, **algo_args)

    t = time.process_time()
    algo.run_cpu(data, **algo_args, **setup_overrides)
    elapsed_time = time.process_time() - t

    print('\n' + '\033[33m' + '=x'*20 + ' SKLEARN-INTELEX ' + '=x'*20)
    print(algo.name + ' : ' + str(algo.cpu_class))
    print('\tbench_function: ' + str(algo.bench_func))
    print('\truntime: ' + str(elapsed_time))
    print('=x'*49 + '\033[0m' + '\n')


def _benchmark_algo(
    benchmark,
    name,
    dataset_name,
    n_samples=10000,
    n_features=100,
    input_type='cupy',
    data_kwargs={},
    algo_args={},
    client=None
):
    """Simplest benchmark wrapper to time algorithm 'name' on dataset
    'dataset_name'"""
    algo = algorithms.algorithm_by_name(name)
    data = generate_dataset(dataset_name, n_samples, n_features,
                            input_type, data_kwargs)

    setup_overrides = algo.setup_cuml(data, client=client, **algo_args)

    multi_node = client is not None
    if multi_node and name != 'MNMG.DBSCAN':  # if MNMG => scatter data
        data = [distribute(client, d) for d in data]

    def _benchmark_inner():
        algo.run_cuml(data, **algo_args, **setup_overrides)

    benchmark(_benchmark_inner)

    if not multi_node:  # if SG => run NVTX benchmark
        nvtx_profiling(name, dataset_name, n_samples, n_features,
                       input_type, data_kwargs, algo_args)

        sklearn_intelex_bench(algo, data, algo_args)
