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
"""bench_runners.py - Wrappers to run ML benchmarks"""

from cuml.benchmark import bench_data, bench_algos
import time
import numpy as np

class SpeedupComparisonRunner:
    """Wrapper to run an algorithm with multiple dataset sizes
    and compute speedup of cuml relative to sklearn baseline."""
    def __init__(self, bench_rows, bench_dims,
                 dataset_name='blobs', input_type='numpy'):
        self.bench_rows = bench_rows
        self.bench_dims = bench_dims
        self.dataset_name = dataset_name
        self.input_type = input_type

    def _run_one_size(self, algo_pair, n_samples, n_features,
                      param_overrides={},
                      cuml_param_overrides={},
                      cpu_param_overrides={},
                      run_cpu=True):
        data = bench_data.gen_data(self.dataset_name,
                                   self.input_type,
                                   n_samples,
                                   n_features)
        cu_start = time.time()
        algo_pair.run_cuml(data, **param_overrides, **cuml_param_overrides)
        cu_elapsed = time.time() - cu_start

        if run_cpu:
            cpu_start = time.time()
            algo_pair.run_cpu(data, **param_overrides)
            cpu_elapsed = time.time() - cpu_start
        else:
            cpu_elapsed = 0.0

        return dict(cu_time=cu_elapsed,
                    cpu_time=cpu_elapsed,
                    speedup=cpu_elapsed / float(cu_elapsed),
                    n_samples=n_samples,
                    n_features=n_features,
                    **param_overrides,
                    **cuml_param_overrides)

    def run(self, algo_pair, param_overrides={},
            cuml_param_overrides={},
            cpu_param_overrides={}, *, run_cpu=True):
        all_results = []
        for ns in self.bench_rows:
            for nf in self.bench_dims:
                try:
                    all_results.append(self._run_one_size(algo_pair,
                                                          ns,
                                                          nf,
                                                          param_overrides,
                                                          cuml_param_overrides,
                                                          cpu_param_overrides, run_cpu))
                except Exception as e:
                    print("Failed to run with %d samples, %d features: %s" %
                          (ns, nf, str(e)))
                    raise
                    all_results.append(dict(n_samples=ns, n_features=nf))
        return all_results



class AccuracyComparisonWrapper(SpeedupComparisonRunner):
    """Wrapper to run an algorithm with multiple dataset sizes
    and compute accuracy and speedup of cuml relative to sklearn
    baseline."""
    def __init__(self, bench_rows, bench_dims,
                 dataset_name='blobs', input_type='numpy',
                 test_fraction=0.10):
        super().__init__(bench_rows, bench_dims, dataset_name, input_type)
        self.test_fraction = 0.10

    def _run_one_size(self, algo_pair, n_samples, n_features,
                      param_overrides={},
                      cuml_param_overrides={},
                      cpu_param_overrides={},
                      run_cpu=True):
        # keep training set size constant even as we add test
        n_samples_with_test = int(n_samples / (1 - self.test_fraction))

        data = bench_data.gen_data(self.dataset_name,
                                   self.input_type,
                                   n_samples_with_test,
                                   n_features,
                                   test_fraction=self.test_fraction)
        X_test, y_test = data[2:]

        cu_start = time.time()
        cuml_model = algo_pair.run_cuml(data, **param_overrides, **cuml_param_overrides)
        cu_elapsed = time.time() - cu_start

        if algo_pair.accuracy_function:
            if hasattr(cuml_model, 'predict'):
                y_pred_cuml = cuml_model.predict(X_test)
            else:
                y_pred_cuml = cuml_model.transform(X_test)
            cuml_accuracy = algo_pair.accuracy_function(y_test, np.asarray(y_pred_cuml))
        else:
            cuml_accuracy = 0.0

        cpu_accuracy = 0.0
        if run_cpu:
            cpu_start = time.time()
            cpu_model = algo_pair.run_cpu(data, **param_overrides)
            cpu_elapsed = time.time() - cpu_start

            if algo_pair.accuracy_function:
                if hasattr(cpu_model, 'predict'):
                    y_pred_cpu = cpu_model.predict(X_test)
                else:
                    y_pred_cpu = cpu_model.transform(X_test)
                cpu_accuracy = algo_pair.accuracy_function(y_test, np.asarray(y_pred_cpu))
        else:
            cpu_elapsed = 0.0

        return dict(cu_time=cu_elapsed,
                    cpu_time=cpu_elapsed,
                    cuml_acc=cuml_accuracy,
                    cpu_acc=cpu_accuracy,
                    speedup=cpu_elapsed / float(cu_elapsed),
                    n_samples=n_samples,
                    n_features=n_features,
                    **param_overrides,
                    **cuml_param_overrides)
