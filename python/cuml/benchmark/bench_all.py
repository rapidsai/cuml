"""bench_all.py - Command-line ML benchmark runner"""
from cuml.benchmark import bench_data, bench_algos, bench_runners

import time
import sys
import pandas as pd


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-row-power', type=int, default=15,
                        help='Evaluate at most 2**max_row_power samples')
    parser.add_argument('--min-row-power', type=int, default=10,
                        help='Evaluate at least 2**min_row_power samples')
    parser.add_argument('--num-features', type=int, default=-1)
    parser.add_argument('--quiet', '-q', action='store_false', dest='verbose', default=True)
    parser.add_argument('--csv', nargs='?')
    parser.add_argument('--dataset', default='blobs')
    parser.add_argument('--input-type', default='numpy')
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

    if args.algorithms:
        algos_to_run = [bench_algos.algorithm_by_name(name)
                        for name in args.algorithms]
    else:
        # Run all by default
        algos_to_run = bench_algos.all_algorithms()

    print("Running: \n", "\n ".join([a.name for a in algos_to_run]))
    runner = bench_runners.SpeedupComparisonRunner(bench_rows,
                                                   bench_dims,
                                                   args.dataset,
                                                   args.input_type)
    all_results = []
    for algo in algos_to_run:
        results = runner.run(algo)
        for r in results:
            all_results.append({ 'algo': algo.name, 'input': args.input_type, **r })

    print("Run finished!")
    results_df = pd.DataFrame.from_records(all_results)
    print(results_df)

    if args.csv:
        results_df.to_csv(args.csv)
        print("Saved results to %s" % args.csv)
