# Provided as a minimal example, not for real checkin
from cuml.benchmark import utils, bench_data
from importlib import reload

import sklearn.cluster
import cuml.cluster

kmeans_compare =  utils.AlgoComparisonWrapper(
    sklearn.cluster.KMeans,
    cuml.cluster.KMeans,
    shared_args=dict(init='random',
                     n_clusters=8,
                     max_iter=300),
    accepts_labels=False,
    load_data=bench_data.gen_data_regression,
    name='KMeans')

bench_rows = [2**x for x in range(10,20,3)]
runner = utils.BenchmarkRunner(
    benchmarks=[utils.SpeedupBenchmark()],
    rerun=True,
    bench_rows=bench_rows)

runner.run(kmeans_compare)

print("Run finished!")
print(runner.results_df())
