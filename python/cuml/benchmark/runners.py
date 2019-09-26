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

from cuml.benchmark import datagen
import time
import numpy as np
import pandas as pd


class SpeedupComparisonRunner:
    """Wrapper to run an algorithm with multiple dataset sizes
    and compute speedup of cuml relative to sklearn baseline."""

    def __init__(
        self, bench_rows, bench_dims, dataset_name='blobs', input_type='numpy'
    ):
        self.bench_rows = bench_rows
        self.bench_dims = bench_dims
        self.dataset_name = dataset_name
        self.input_type = input_type

    def _run_one_size(
        self,
        algo_pair,
        n_samples,
        n_features,
        param_overrides={},
        cuml_param_overrides={},
        cpu_param_overrides={},
        run_cpu=True,
    ):
        data = datagen.gen_data(
            self.dataset_name, self.input_type, n_samples, n_features
        )
        print("data type: ", data[0].__class__)

        cu_start = time.time()
        algo_pair.run_cuml(data, **param_overrides, **cuml_param_overrides)
        cu_elapsed = time.time() - cu_start

        if run_cpu and algo_pair.cpu_class is not None:
            cpu_start = time.time()
            algo_pair.run_cpu(data, **param_overrides)
            cpu_elapsed = time.time() - cpu_start
        else:
            cpu_elapsed = 0.0

        return dict(
            cu_time=cu_elapsed,
            cpu_time=cpu_elapsed,
            speedup=cpu_elapsed / float(cu_elapsed),
            n_samples=n_samples,
            n_features=n_features,
            **param_overrides,
            **cuml_param_overrides
        )

    def _run_one_size_fil(
        self,
        algo_pair,
        n_samples,
        n_features,
        param_overrides={},
        cuml_param_overrides={},
        cpu_param_overrides={},
        run_cpu=True,
        fil_algorithms=[],
    ):
        data = datagen.gen_data(
            self.dataset_name,
            self.input_type,
            n_samples,
            n_features,
            test_fraction=0.10,
        )
        print("data type: ", data[0].__class__)

        result = dict(
            n_samples=n_samples,
            n_features=n_features,
            **param_overrides,
            **cuml_param_overrides
        )
        
        algo_pair.prepare_xgboost(data, **param_overrides)

        if run_cpu and "sklearn" in fil_algorithms:
            algo_pair.prepare_sklearn(data, **param_overrides)
        if run_cpu and "treelite" in fil_algorithms: 
            algo_pair.prepare_treelite(data, **param_overrides)
        algo_pair.prepare_cuml(data, **param_overrides, **cuml_param_overrides)

        if run_cpu and "sklearn" in fil_algorithms:
            skl_start = time.time()
            algo_pair.run_sklearn(data)
            skl_elapsed = time.time() - skl_start
            result["skl_time"] = skl_elapsed

        if run_cpu and "xgboost_cpu" in fil_algorithms: 
            xgb_cpu_start = time.time()
            algo_pair.run_xgboost_cpu(data)
            xgb_cpu_elapsed = time.time() - xgb_cpu_start
            result["xgb_cpu_time"] = xgb_cpu_elapsed

        if "xgboost_gpu" in fil_algorithms: 
            xgb_gpu_start = time.time()
            algo_pair.run_xgboost_gpu(data)
            xgb_gpu_elapsed = time.time() - xgb_gpu_start
            result["xgb_gpu_time"] = xgb_gpu_elapsed

        if run_cpu and "treelite" in fil_algorithms: 
            tl_start = time.time()
            algo_pair.run_treelite(data)
            tl_elapsed = time.time() - tl_start
            result["tl_time"] = tl_elapsed

        cu_start = time.time()
        algo_pair.run_cuml(data)
        cu_elapsed = time.time() - cu_start
        result["cu_time"] = cu_elapsed

        return result

    def run(
        self,
        algo_pair,
        param_overrides={},
        cuml_param_overrides={},
        cpu_param_overrides={},
        *,
        run_cpu=True,
        raise_on_error=False,
        fil_algorithms=[],
    ):
        all_results = []
        for ns in self.bench_rows:
            for nf in self.bench_dims:
                try:
                    if algo_pair.name == "FIL":
                        all_results.append(
                            self._run_one_size_fil(
                                algo_pair,
                                ns,
                                nf,
                                param_overrides,
                                cuml_param_overrides,
                                cpu_param_overrides,
                                run_cpu,
                                fil_algorithms,
                            )
                        )
                    else:
                        all_results.append(
                            self._run_one_size(
                                algo_pair,
                                ns,
                                nf,
                                param_overrides,
                                cuml_param_overrides,
                                cpu_param_overrides,
                                run_cpu,
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
        dataset_name='blobs',
        input_type='numpy',
        test_fraction=0.10,
    ):
        super().__init__(bench_rows, bench_dims, dataset_name, input_type)
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
    ):
        data = datagen.gen_data(
            self.dataset_name,
            self.input_type,
            n_samples,
            n_features,
            test_fraction=self.test_fraction,
        )
        X_test, y_test = data[2:]

        cu_start = time.time()
        cuml_model = algo_pair.run_cuml(
            data, **{**param_overrides, **cuml_param_overrides}
        )
        cu_elapsed = time.time() - cu_start
        if algo_pair.accuracy_function:
            if hasattr(cuml_model, 'predict'):
                y_pred_cuml = cuml_model.predict(X_test)
            else:
                y_pred_cuml = cuml_model.transform(X_test)
            cuml_accuracy = algo_pair.accuracy_function(
                y_test, np.asarray(y_pred_cuml)
            )
        else:
            cuml_accuracy = 0.0

        cpu_accuracy = 0.0
        if run_cpu and algo_pair.cpu_class is not None:
            cpu_start = time.time()
            cpu_model = algo_pair.run_cpu(data, **param_overrides)
            cpu_elapsed = time.time() - cpu_start

            if algo_pair.accuracy_function:
                if hasattr(cpu_model, 'predict'):
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

    def _run_one_size_fil(
        self,
        algo_pair,
        n_samples,
        n_features,
        param_overrides={},
        cuml_param_overrides={},
        cpu_param_overrides={},
        run_cpu=True,
        fil_algorithms=[],
    ):
        data = datagen.gen_data(
            self.dataset_name,
            self.input_type,
            n_samples,
            n_features,
            test_fraction=0.10,
        )
        print("data type: ", data[0].__class__)

        result = dict(
            n_samples=n_samples,
            n_features=n_features,
            **param_overrides,
            **cuml_param_overrides
        )
        
        algo_pair.prepare_xgboost(data, **param_overrides)

        if run_cpu and "sklearn" in fil_algorithms:
            algo_pair.prepare_sklearn(data, **param_overrides)
        if run_cpu and "treelite" in fil_algorithms: 
            algo_pair.prepare_treelite(data, **param_overrides)
        algo_pair.prepare_cuml(data, **param_overrides, **cuml_param_overrides)

        if run_cpu and "sklearn" in fil_algorithms:
            skl_start = time.time()
            skl_preds = algo_pair.run_sklearn(data)
            skl_elapsed = time.time() - skl_start
            result["skl_time"] = skl_elapsed
            result["skl_acc"] = algo_pair.accuracy_function(skl_preds, data[1])

        if run_cpu and "xgboost_cpu" in fil_algorithms: 
            xgb_cpu_start = time.time()
            xgb_cpu_preds = algo_pair.run_xgboost_cpu(data)
            xgb_cpu_elapsed = time.time() - xgb_cpu_start
            result["xgb_cpu_time"] = xgb_cpu_elapsed
            result["xgb_cpu_acc"] = algo_pair.accuracy_function(xgb_cpu_preds > 0.5, data[1])

        if "xgboost_gpu" in fil_algorithms: 
            xgb_gpu_start = time.time()
            xgb_gpu_preds = algo_pair.run_xgboost_gpu(data)
            xgb_gpu_elapsed = time.time() - xgb_gpu_start
            result["xgb_gpu_time"] = xgb_gpu_elapsed
            result["xgb_gpu_acc"] = algo_pair.accuracy_function(xgb_gpu_preds > 0.5, data[1])

        if run_cpu and "treelite" in fil_algorithms: 
            tl_start = time.time()
            tl_preds = algo_pair.run_treelite(data)
            tl_elapsed = time.time() - tl_start
            result["tl_time"] = tl_elapsed
            result["tl_acc"] = algo_pair.accuracy_function(tl_preds > 0.5, data[1])

        cu_start = time.time()
        cuml_preds = algo_pair.run_cuml(data)
        cu_elapsed = time.time() - cu_start
        result["cu_time"] = cu_elapsed
        result["cuml_acc"] = algo_pair.accuracy_function(cuml_preds, data[1])

        return result


def run_variations(
    algos,
    dataset_name,
    bench_rows,
    bench_dims,
    param_override_list=[{}],
    cuml_param_override_list=[{}],
    input_type="numpy",
    run_cpu=True,
    raise_on_error=False,
    fil_algorithms=[],
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
    run_cpu : boolean
      If True, run the cpu-based algorithm for comparison
    fil_algorithms : list of str
      List contating the algorithms to run for FIL benchmarking other than cuml
    """
    print("Running: \n", "\n ".join([str(a.name) for a in algos]))
    runner = AccuracyComparisonRunner(
        bench_rows, bench_dims, dataset_name, input_type
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
                    fil_algorithms=fil_algorithms,
                )
                for r in results:
                    all_results.append(
                        {'algo': algo.name, 'input': input_type, **r}
                    )

    print("Finished all benchmark runs")
    results_df = pd.DataFrame.from_records(all_results)
    print(results_df)

    return results_df
