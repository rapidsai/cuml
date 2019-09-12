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

class SpeedupComparisonRunner:
    """Wrapper to run an algorithm with multiple dataset sizes
    and compute speedup of cuml relative to sklearn baseline."""
    def __init__(self, bench_rows, bench_dims,
                 dataset_name='blobs', input_type='numpy'):
        self.bench_rows = bench_rows
        self.bench_dims = bench_dims
        self.dataset_name = dataset_name
        self.input_type = input_type

    def _run_one_size(self, algo_pair, data, param_overrides={}, cuml_param_overrides={}, run_cpu=True):
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
                    n_samples=data[0].shape[0],
                    n_features=data[0].shape[1],
                    **param_overrides,
                    **cuml_param_overrides)

    def run(self, algo_pair, param_overrides={}, cuml_param_overrides={}, *, run_cpu=True):
        all_results = []
        for ns in self.bench_rows:
            for nf in self.bench_dims:
                try:
                    data = bench_data.gen_data(self.dataset_name,
                                               self.input_type,
                                               ns,
                                               nf)
                    all_results.append(self._run_one_size(algo_pair, data, param_overrides, cuml_param_overrides, run_cpu))
                except Exception as e:
                    print("Failed to run with %d samples, %d features: %s" %
                          (ns, nf, str(e)))
                    raise
                    all_results.append(dict(n_samples=ns, n_features=nf))
        return all_results


