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
from cuml.benchmark import datagen, algorithms
from cuml.benchmark.nvtx_benchmark import Profiler
import dask.array as da


def to_dask_array(np_array, client):
    if np_array is not None:
        n_rows = np_array.shape[0]
        n_workers = len(client.scheduler_info()['workers'])
        dask_array = da.from_array(np_array, chunks=n_rows // n_workers)
        return dask_array


def _benchmark_algo(
    benchmark,
    name,
    dataset_name,
    n_samples=10000,
    n_features=100,
    input_type='numpy',
    data_kwargs={},
    algo_args={},
    client=None
):
    """Simplest benchmark wrapper to time algorithm 'name' on dataset
    'dataset_name'"""
    algo = algorithms.algorithm_by_name(name)
    data = datagen.gen_data(
        dataset_name,
        input_type,
        n_samples=n_samples,
        n_features=n_features,
        **data_kwargs
    )

    if client:  # if MNMG => scatter data
        algo_args['client'] = client
        data = [to_dask_array(d, client) for d in data]

    def _benchmark_inner():
        algo.run_cuml(data, **algo_args)

    benchmark(_benchmark_inner)

    if client is None:  # if SG => run NVTX benchmark
        path = os.path.dirname(os.path.realpath(__file__))
        command = """
        python {path}/run_benchmarks.py
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
