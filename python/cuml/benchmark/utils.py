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

import numpy as np
import pandas as pd
import cudf
import os
import time
import pickle
import sklearn.datasets

def numpy_convert(data):
    if isinstance(data, tuple):
        return tuple([numpy_convert(d) for d in data])
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    else:
        raise Exception("Unsupported type %s" % str(type(data)))

def cudf_convert(data):
    if isinstance(data, tuple):
        return tuple([pandas_convert(d) for d in data])
    elif isinstance(data, pd.DataFrame):
        return cudf.DataFrame.from_pandas(data)
    elif isinstance(data, pd.Series):
        return cudf.Series.from_pandas(data)
    else:
        raise Exception("Unsupported type %s" % str(type(data)))

class SpeedupBenchmark(object):
    def __init__(self, converter=numpy_convert, name="speedup"):
        self.name = name
        self.converter = converter

    def __str__(self):
        return "Speedup"

    def run(self, algo, rows, dims, data):
        data2 = self.converter(data)
        cu_start = time.time()
        algo.cuml(data2)
        cu_elapsed = time.time() - cu_start

        sk_start = time.time()
        algo.sk(data)
        sk_elapsed = time.time() - float(sk_start)

        # Needs to return the calculation and the name given to it.
        return dict(cu_time=cu_elapsed,
                    sk_time=sk_elapsed,
                    speedup=sk_elapsed / float(cu_elapsed))

class CuMLOnlyBenchmark(object):
    def __init__(self, converter = numpy_convert):
        self.name = "speedup"
        self.converter = converter

    def __str__(self):
        return "CuMLOnly"

    def run(self, algo, rows, dims, data):
        data2 = self.converter(data)
        cu_start = time.time()
        algo.cuml(data2)
        cu_elapsed = time.time() - cu_start

        # Needs to return the calculation and the name given to it.
        return dict(cu_time=cu_elapsed,
                    sk_time=np.nan,
                    speedup=np.nan)


class BenchmarkRunner(object):
    def __init__(self,
                 benchmarks = [SpeedupBenchmark()],
                 out_filename = "benchmark.pickle",
                 rerun = False,
                 n_runs = 3,
                 bench_rows = [2**x for x in range(13, 20)],
                 bench_dims = [64, 128, 256, 512],
                 continue_on_fail = False,
                 verbose = True,
                 persist_results = False):

        self.benchmarks = benchmarks
        # XXX: maybe provide an option to never save/load data but
        # just keep in memory?
        self.rerun = rerun
        self.n_runs = n_runs
        self.bench_rows = bench_rows
        self.bench_dims = bench_dims
        self.out_filename = out_filename
        self.verbose = verbose
        self.all_results = {}
        self.persist_results = persist_results
        if persist_results:
            self.all_results = self.load_results()

    def _log(self, s):
        if self.verbose:
            print(s)

    def load_results(self):
        if os.path.exists(self.out_filename):
            self._log("Loading previous benchmark results from %s" % (self.out_filename))
            with open(self.out_filename, 'rb') as f:
                return pickle.load(f)
        else:
            return {}

    def store_results(self, final_results):
        with open(self.out_filename, 'wb') as f:
            pickle.dump(final_results, f)

    def results_df(self):
        # results = self.load_results()
        results = self.all_results
        all_keys = []
        all_values = []
        for algorithm, algo_results in results.items():
            for (n_samples, n_features, benchmark), values in algo_results.items():
                all_keys.append({
                    'algorithm': algorithm,
                    'benchmark': benchmark,
                    'n_samples': n_samples,
                    'n_features': n_features})
                all_values.append(values)

        return pd.concat((
            pd.DataFrame.from_records(all_keys),
            pd.DataFrame.from_records(all_values)),
            axis=1)

    def _run_single_algo(self, benchmark, algo, n_rows, n_dims):
        data = algo.load_data(n_rows, n_dims)
        runs_list = [benchmark.run(algo, n_rows, n_dims, data) for i in range(self.n_runs)]
        runs_df = pd.DataFrame.from_records(runs_list)
        cur_result = runs_df.mean(0) # XXX decide whether we really want mean or min, recalc speedup?
        return cur_result

    def run(self, algo):
        # final_results = self.load_results()
        final_results = self.all_results
        for benchmark in self.benchmarks:
            if algo.name in final_results:
                results = final_results[algo.name]
            else:
                results = {}
                final_results[algo.name] = results

            for n_rows in self.bench_rows:
                for n_dims in self.bench_dims:
                    if (n_rows, n_dims, benchmark.name) not in results or self.rerun:
                        self._log("Running %s. (nrows=%d, n_dims=%d)" % (str(algo), n_rows, n_dims))
                        cur_result = self._run_single_algo(benchmark,
                                                           algo,
                                                           n_rows,
                                                           n_dims)

                        results[(n_rows, n_dims, benchmark.name)] = cur_result
                        self._log("Benchmark for %30s = %8.3f cu, %8.3f sk, %8.3f speedup" % (
                            str((n_rows, n_dims, benchmark.name)),
                            cur_result['cu_time'],
                            cur_result['sk_time'],
                            cur_result['speedup']))

                        if self.persist_results:
                            self.store_results(final_results)

    def chart(self, algo, title = "cuML vs SKLearn"):
        import matplotlib.pyplot as plt

        for benchmark in self.benchmarks:
            results = self.all_results[algo.name]

            final = {}

            plts = []
            for dim in self.bench_dims:
                data = {k: v for (k, v) in results.items() if dim == k[1]}

                if len(data) > 0:
                    data = [(k[0], v) for k, v in data.items()]
                    data.sort(key = lambda x: x[0])

                    final[dim] = list(map(lambda x: x[1], data))

                    keys = list(map(lambda x: np.log2(x[0]), data))
                line = plt.plot(keys, final[dim], label = str(dim), linewidth = 3,
                                marker = 'o', markersize = 7)

                plts.append(line[0])
            leg = plt.legend(handles = plts, fontsize = 10)
            leg.set_title("Dimensions", prop = {'size':'x-large'})
            plt.title("%s %s: %s" % (algo, benchmark, title), fontsize = 20)

            plt.ylabel(str(benchmark), fontsize = 20)
            plt.xlabel("Training Examples (2^x)", fontsize = 20)

            plt.tick_params(axis='both', which='major', labelsize=15)
            plt.tick_params(axis='both', which='minor', labelsize=15)

            plt.show()


def gen_data_Xy(n_samples, n_features, random_state=42):
    X_arr, y_arr = sklearn.datasets.make_regression(n_samples, n_features, random_state=random_state)
    return (pd.DataFrame(X_arr), pd.Series(y_arr))

def gen_data_X(n_samples, n_features, random_state=42):
    return gen_data_Xy(n_samples, n_features, random_state)[0]

class BaseAlgorithm(object):
    def __init__(self, load_data=gen_data_X):
        self.load_data = load_data


class AlgoComparisonWrapper(BaseAlgorithm):
    """
    Easy-to-use wrapper comparing runtimes of scikit-learn and cuml
    implementations of the same algorithm.

    """
    def __init__(self, sk_class, cuml_class,
                 shared_args,
                 cuml_args={},
                 sklearn_args={},
                 name=None,
                 load_data=gen_data_X):
        if name:
            self.name = name
        else:
            self.name = cuml_class.__name__
        self.sk_class = sk_class
        self.cuml_class = cuml_class
        self.shared_args = shared_args
        self.cuml_args = cuml_args
        self.sklearn_args = sklearn_args
        BaseAlgorithm.__init__(self, load_data=load_data)

    def __str__(self):
        return "AlgoComparison:%s" % (self.name)

    def sk(self, data):
        all_args = {**self.shared_args, **self.sklearn_args}
        sk_obj = self.sk_class(**all_args)
        if isinstance(data, tuple) and len(data) == 2:
            sk_obj.fit(data[0], data[1])
        else:
            sk_obj.fit(data)

    def cuml(self, data):
        all_args = {**self.shared_args, **self.cuml_args}
        cuml_obj = self.cuml_class(**all_args)
        if isinstance(data, tuple) and len(data) == 2:
            cuml_obj.fit(data[0], data[1])
        else:
            cuml_obj.fit(data)


