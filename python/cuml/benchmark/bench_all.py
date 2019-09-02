from cuml.benchmark import utils, bench_data

import sys

from .bench_algos import all_algorithms

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

    all_algos = all_algorithms(dataset_loader)
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
