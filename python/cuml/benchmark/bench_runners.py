"""bench_runners.py - Wrappers to run ML benchmarks"""

from cuml.benchmark import bench_data, bench_algos
import time

class SpeedupComparisonRunner:
    """Wrapper to run an algorithm with multiple dataset sizes
    and compute speedup of cuml relative to sklearn."""
    def __init__(self, bench_rows, bench_dims,
                 dataset_name='blobs', input_type='numpy'):
        self.bench_rows = bench_rows
        self.bench_dims = bench_dims
        self.dataset_name = dataset_name
        self.input_type = input_type

    def _run_one_size(self, algo_pair, n_samples, n_features):
        data = bench_data.gen_data(self.dataset_name,
                                   self.input_type,
                                   n_samples,
                                   n_features)
        cu_start = time.time()
        algo_pair.run_cuml(data)
        cu_elapsed = time.time() - cu_start

        sk_start = time.time()
        algo_pair.run_cpu(data)
        sk_elapsed = time.time() - sk_start

        return dict(cu_time=cu_elapsed,
                    sk_time=sk_elapsed,
                    speedup=sk_elapsed / float(cu_elapsed),
                    n_samples=n_samples,
                    n_features=n_features)

    def run(self, algo_pair):
        all_results = []
        for ns in self.bench_rows:
            for nf in self.bench_dims:
                try:
                    all_results.append(self._run_one_size(algo_pair, ns, nf))
                except Exception as e:
                    print("Failed to run with %d samples, %d features: %s" %
                          (ns, nf, str(e)))
                    all_results.append(dict(n_samples=ns, n_features=nf))
        return all_results
