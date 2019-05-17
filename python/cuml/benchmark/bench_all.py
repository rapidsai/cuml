import utils
from importlib import reload
reload(utils)

import sklearn.cluster, sklearn.neighbors
import cuml.cluster, cuml.neighbors
import umap
import sys

ALL_ALGOS = [
    utils.AlgoComparisonWrapper(
        sklearn.cluster.KMeans,
        cuml.cluster.KMeans,
        shared_args=dict(init='random',
                         n_clusters=8,
                         max_iter=300),
        name='KMeans'),
    utils.AlgoComparisonWrapper(
        sklearn.decomposition.PCA,
        cuml.PCA,
        shared_args=dict(n_components=10),
        name='PCA'),
    utils.AlgoComparisonWrapper(
        sklearn.neighbors.NearestNeighbors,
        cuml.neighbors.NearestNeighbors,
        shared_args=dict(n_neighbors=1024),
        sklearn_args=dict(algorithm='brute'),
        cuml_args=dict(n_gpus=1),
        name='NearestNeighbors'
        ),
    utils.AlgoComparisonWrapper(
        sklearn.cluster.DBSCAN,
        cuml.DBSCAN,
        shared_args=dict(eps=3, min_samples=2),
        sklearn_args=dict(algorithm='brute'),
        name='DBScan'
    ),
    utils.AlgoComparisonWrapper(
        sklearn.linear_model.LinearRegression,
        cuml.linear_model.LinearRegression,
        shared_args={},
        load_data=utils.gen_data_Xy,
        name='LinearRegression'),
    utils.AlgoComparisonWrapper(
        umap.UMAP,
        cuml.manifold.UMAP,
        shared_args=dict(n_neighbors=5, n_epochs=500),
        name='UMAP'),
]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-row-power', type=int, default=15,
                        help='Evaluate at most 2**max_row_power samples')
    parser.add_argument('--min-row-power', type=int, default=10,
                        help='Evaluate at least 2**min_row_power samples')
    parser.add_argument('--quiet', '-q', action='store_false', dest='verbose', default=True)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--csv', nargs='?')
    parser.add_argument('algorithms', nargs='*', help='List of algorithms to run, or omit to run all')
    args = parser.parse_args()


    bench_rows = [2**x for x in range(args.min_row_power, args.max_row_power, 2)]
    runner = utils.BenchmarkRunner(rerun=True,
                                   benchmarks=[utils.SpeedupBenchmark(name='cudf',
                                                                      converter=utils.cudf_convert),
                                               utils.SpeedupBenchmark(name='numpy')],
                                   verbose=args.verbose,
                                   bench_rows=bench_rows)

    if args.algorithms:
        algos_to_run = [a for a in ALL_ALGOS if a.name in args.algorithms]
    else:
        algos_to_run = ALL_ALGOS

    print("Running: \n", "\n ".join([a.name for a in algos_to_run]))

    for algo in algos_to_run:
        runner.run(algo)

    print("Run finished!")
    results = runner.results_df()
    print(results)

    if args.csv:
        results.to_csv(args.csv)
        print("Saved results to %s" % args.csv)
