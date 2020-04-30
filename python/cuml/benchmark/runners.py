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
"""Wrappers to run ML benchmarks"""

import time
import numpy as np
import pandas as pd

from cuml.benchmark import datagen
from cudf.core import Series


class BenchmarkTimer:
    """Provides a context manager that runs a code block `reps` times
    and records results to the instance variable `timings`. Use like:

    .. code-block:: python

        timer = BenchmarkTimer(rep=5)
        for _ in timer.benchmark_runs():
            ... do something ...
        print(np.min(timer.timings))
    """

    def __init__(self, reps=1):
        self.reps = reps
        self.timings = []

    def benchmark_runs(self):
        for r in range(self.reps):
            t0 = time.time()
            yield r
            t1 = time.time()
            self.timings.append(t1 - t0)


class SpeedupComparisonRunner:
    """Wrapper to run an algorithm with multiple dataset sizes
    and compute speedup of cuml relative to sklearn baseline."""

    def __init__(
        self,
        bench_rows,
        bench_dims,
        dataset_name="blobs",
        input_type="numpy",
        n_reps=1,
    ):
        self.bench_rows = bench_rows
        self.bench_dims = bench_dims
        self.dataset_name = dataset_name
        self.input_type = input_type
        self.n_reps = n_reps

    def _run_one_size(
        self,
        algo_pair,
        n_samples,
        n_features,
        param_overrides={},
        cuml_param_overrides={},
        cpu_param_overrides={},
        run_cpu=True,
        verbose=False,
    ):
        data = datagen.gen_data(
            self.dataset_name, self.input_type, n_samples, n_features
        )

        setup_overrides = algo_pair.setup_cuml(
            data, **param_overrides, **cuml_param_overrides
        )

        cuml_timer = BenchmarkTimer(self.n_reps)
        for rep in cuml_timer.benchmark_runs():
            algo_pair.run_cuml(
                data,
                **param_overrides,
                **cuml_param_overrides,
                **setup_overrides
            )
        cu_elapsed = np.min(cuml_timer.timings)

        if run_cpu and algo_pair.cpu_class is not None:
            setup_overrides = algo_pair.setup_cpu(data, **param_overrides)

            cpu_timer = BenchmarkTimer(self.n_reps)
            for rep in cpu_timer.benchmark_runs():
                algo_pair.run_cpu(data, **param_overrides, **setup_overrides)
            cpu_elapsed = np.min(cpu_timer.timings)
        else:
            cpu_elapsed = 0.0

        speedup = cpu_elapsed / float(cu_elapsed)
        if verbose:
            print(
                "%s Speedup (n_samples=%s, n_features=%s) = %s"
                % (algo_pair.name, n_samples, n_features, speedup)
            )

        return dict(
            cu_time=cu_elapsed,
            cpu_time=cpu_elapsed,
            speedup=speedup,
            n_samples=n_samples,
            n_features=n_features,
            **param_overrides,
            **cuml_param_overrides
        )

    def run(
        self,
        algo_pair,
        param_overrides={},
        cuml_param_overrides={},
        cpu_param_overrides={},
        *,
        run_cpu=True,
        raise_on_error=False,
        verbose=False
    ):
        all_results = []
        for ns in self.bench_rows:
            for nf in self.bench_dims:
                try:
                    all_results.append(
                        self._run_one_size(
                            algo_pair,
                            ns,
                            nf,
                            param_overrides,
                            cuml_param_overrides,
                            cpu_param_overrides,
                            run_cpu,
                            verbose,
                        )
                    )
                except Exception as e:
                    print(
                        "Failed to run with %d samples, %d features: %s"
                        % (ns, nf, str(e))
                    )
                    if raise_on_error:
                        raise
                    all_results.append(dict(n_samples=ns, n_features=nf))
        return all_results


class AccuracyComparisonRunner(SpeedupComparisonRunner):
    """Wrapper to run an algorithm with multiple dataset sizes
    and compute accuracy and speedup of cuml relative to sklearn
    baseline."""

    def __init__(
        self,
        bench_rows,
        bench_dims,
        dataset_name="blobs",
        input_type="numpy",
        test_fraction=0.10,
        n_reps=1,
    ):
        super().__init__(
            bench_rows, bench_dims, dataset_name, input_type, n_reps
        )
        self.test_fraction = test_fraction

    def _run_one_size(
        self,
        algo_pair,
        n_samples,
        n_features,
        param_overrides={},
        cuml_param_overrides={},
        cpu_param_overrides={},
        run_cpu=True,
        verbose=False,
    ):
        data = datagen.gen_data(
            self.dataset_name,
            self.input_type,
            n_samples,
            n_features,
            test_fraction=self.test_fraction,
        )

        setup_override = algo_pair.setup_cuml(
            data, **{**param_overrides, **cuml_param_overrides}
        )

        cuml_timer = BenchmarkTimer(self.n_reps)
        for _ in cuml_timer.benchmark_runs():
            cuml_model = algo_pair.run_cuml(
                data,
                **{**param_overrides, **cuml_param_overrides, **setup_override}
            )
        cu_elapsed = np.min(cuml_timer.timings)

        if algo_pair.accuracy_function:
            if algo_pair.cuml_data_prep_hook is not None:
                X_test, y_test = algo_pair.cuml_data_prep_hook(data[2:])
            else:
                X_test, y_test = data[2:]

            if hasattr(cuml_model, "predict"):
                y_pred_cuml = cuml_model.predict(X_test)
            else:
                y_pred_cuml = cuml_model.transform(X_test)
            if isinstance(y_pred_cuml, Series):
                y_pred_cuml = y_pred_cuml.to_array()
            cuml_accuracy = algo_pair.accuracy_function(
                y_test, y_pred_cuml
            )
        else:
            cuml_accuracy = 0.0

        cpu_accuracy = 0.0
        if run_cpu and algo_pair.cpu_class is not None:
            setup_override = algo_pair.setup_cpu(data, **param_overrides)

            cpu_timer = BenchmarkTimer(self.n_reps)
            for rep in cpu_timer.benchmark_runs():
                cpu_model = algo_pair.run_cpu(
                    data, **param_overrides, **setup_override
                )
            cpu_elapsed = np.min(cpu_timer.timings)

            if algo_pair.accuracy_function:
                if algo_pair.cpu_data_prep_hook is not None:
                    X_test, y_test = algo_pair.cpu_data_prep_hook(data[2:])
                else:
                    X_test, y_test = data[2:]
                if hasattr(cpu_model, "predict"):
                    y_pred_cpu = cpu_model.predict(X_test)
                else:
                    y_pred_cpu = cpu_model.transform(X_test)
                cpu_accuracy = algo_pair.accuracy_function(
                    y_test, np.asarray(y_pred_cpu)
                )
        else:
            cpu_elapsed = 0.0

        return dict(
            cu_time=cu_elapsed,
            cpu_time=cpu_elapsed,
            cuml_acc=cuml_accuracy,
            cpu_acc=cpu_accuracy,
            speedup=cpu_elapsed / float(cu_elapsed),
            n_samples=n_samples,
            n_features=n_features,
            **param_overrides,
            **cuml_param_overrides
        )


def run_variations(
    algos,
    dataset_name,
    bench_rows,
    bench_dims,
    param_override_list=[{}],
    cuml_param_override_list=[{}],
    input_type="numpy",
    test_fraction=0.1,
    run_cpu=True,
    raise_on_error=False,
    n_reps=1,
):
    """
    Runs each algo in `algos` once per
    `bench_rows X bench_dims X params_override_list X cuml_param_override_list`
    combination and returns a dataframe containing timing and accuracy data.

    Parameters
    ----------
    algos : str or list
      Name of algorithms to run and evaluate
    dataset_name : str
      Name of dataset to use
    bench_rows : list of int
      Dataset row counts to test
    bench_dims : list of int
      Dataset column counts to test
    param_override_list : list of dict
      Dicts containing parameters to pass to __init__.
      Each dict specifies parameters to override in one run of the algorithm.
    cuml_param_override_list : list of dict
      Dicts containing parameters to pass to __init__ of the cuml algo only.
    test_fraction : float
      The fraction of data to use for testing.
    run_cpu : boolean
      If True, run the cpu-based algorithm for comparison
    """
    print("Running: \n", "\n ".join([str(a.name) for a in algos]))
    runner = AccuracyComparisonRunner(
        bench_rows, bench_dims, dataset_name, input_type,
        test_fraction=test_fraction, n_reps=n_reps
    )
    all_results = []
    for algo in algos:
        print("Running %s..." % (algo.name))
        for param_overrides in param_override_list:
            for cuml_param_overrides in cuml_param_override_list:
                results = runner.run(
                    algo,
                    param_overrides,
                    cuml_param_overrides,
                    run_cpu=run_cpu,
                    raise_on_error=raise_on_error,
                )
                for r in results:
                    all_results.append(
                        {"algo": algo.name, "input": input_type, **r}
                    )

    print("Finished all benchmark runs")
    results_df = pd.DataFrame.from_records(all_results)
    print(results_df)

    return results_df
