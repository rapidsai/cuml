from cuml.benchmark import utils, bench_data
from importlib import reload

import sklearn.cluster, sklearn.neighbors
import cuml.cluster, cuml.neighbors
import umap
import sys

def get_all_algos(load_data):
    return [
        utils.AlgoComparisonWrapper(
            sklearn.cluster.KMeans,
            cuml.cluster.KMeans,
            shared_args=dict(init='random',
                             n_clusters=8,
                             max_iter=300),
            name='KMeans',
            accepts_labels=False,
            load_data=load_data),
        utils.AlgoComparisonWrapper(
            sklearn.decomposition.PCA,
            cuml.PCA,
            shared_args=dict(n_components=10),
            name='PCA',
            accepts_labels=False,
            load_data=load_data),
        utils.AlgoComparisonWrapper(
            sklearn.neighbors.NearestNeighbors,
            cuml.neighbors.NearestNeighbors,
            shared_args=dict(n_neighbors=1024),
            sklearn_args=dict(algorithm='brute'),
            cuml_args=dict(n_gpus=1),
            name='NearestNeighbors',
            accepts_labels=False,
            load_data=load_data),
        utils.AlgoComparisonWrapper(
            sklearn.cluster.DBSCAN,
            cuml.DBSCAN,
            shared_args=dict(eps=3, min_samples=2),
            sklearn_args=dict(algorithm='brute'),
            name='DBScan',
            accepts_labels=False
        ),
        utils.AlgoComparisonWrapper(
            sklearn.linear_model.LinearRegression,
            cuml.linear_model.LinearRegression,
            shared_args={},
            name='LinearRegression',
            accepts_labels=True,
            load_data=load_data),
        utils.AlgoComparisonWrapper(
            umap.UMAP,
            cuml.manifold.UMAP,
            shared_args=dict(n_neighbors=5, n_epochs=500),
            name='UMAP',
            accepts_labels=False,
            load_data=load_data),
    ]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-row-power', type=int, default=15,
                        help='Evaluate at most 2**max_row_power samples')
    parser.add_argument('--min-row-power', type=int, default=10,
                        help='Evaluate at least 2**min_row_power samples')
    parser.add_argument('--num-features', type=int, default=-1)
    parser.add_argument('--quiet', '-q', action='store_false', dest='verbose', default=True)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--csv', nargs='?')
    parser.add_argument('--dataset', default='gen_data_regression')
    parser.add_argument('--default-size', action='store_true', help='Only run datasets at default size')
    parser.add_argument('algorithms', nargs='*', help='List of algorithms to run, or omit to run all')
    args = parser.parse_args()

    bench_rows = [2**x for x in range(args.min_row_power, args.max_row_power, 2)]
    bench_dims = [64, 256, 512]
    if args.num_features > 0:
        bench_dims = [args.num_features]
    if args.default_size:
        bench_rows = [0]
        bench_dims = [0]

    # By default run with both cudf and numpy inputs to compare
    runner = utils.BenchmarkRunner(rerun=True,
                                   benchmarks=[utils.SpeedupBenchmark(name='cudf',
                                                                      converter=utils.cudf_convert),
                                               utils.SpeedupBenchmark(name='numpy')],
                                   verbose=args.verbose,
                                   bench_rows=bench_rows,
                                   bench_dims=bench_dims)
    dataset_loader = getattr(bench_data, args.dataset)
    print("Using dataset: %s" % (args.dataset,))

    all_algos = get_all_algos(dataset_loader)
    if args.algorithms:
        algos_to_run = [a for a in all_algos if a.name in args.algorithms]
    else:
        algos_to_run = all_algos

    print("Running: \n", "\n ".join([a.name for a in algos_to_run]))

    for algo in algos_to_run:
        runner.run(algo)

    print("Run finished!")
    results = runner.results_df()
    print(results)

    if args.csv:
        results.to_csv(args.csv)
        print("Saved results to %s" % args.csv)
