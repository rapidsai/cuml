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
"""bench_all.py - Command-line ML benchmark runner"""

from cuml.benchmark import bench_data, bench_algos, bench_runners

import time
import sys
import pandas as pd
import numpy as np
import json


def extract_param_overrides(params_to_sweep):
    """
    Parameters
    ----------
      params_to_sweep : list[str]
        list of string key=[value] pairs, where values are to be interpreted
        as a json-style array. E.g. 'n_estimators=[10,100,1000]'

    Returns
    ---------
      List of dicts of params to evaluate. Always contains as least one dict.
    """
    import itertools

    if not params_to_sweep or len(params_to_sweep) == 0:
        return [{}]

    # Expand each arg into a list of (key,value) tuples
    single_param_lists = []
    for p in params_to_sweep:
        key, val_string = p.split("=")
        vals = json.loads(val_string)
        single_param_lists.append([(key, val) for val in vals])

    # Create dicts with the cartesian product of all arg-based lists
    tuple_list = itertools.product(*single_param_lists)
    dict_list = [dict(tl) for tl in tuple_list]
    return dict_list


def run(algos_to_run, dataset, bench_rows, bench_dims, input_type, param_override_list, cuml_param_override_list, output_csv=None, run_cpu=True):
    print("Running: \n", "\n ".join([a.name for a in algos_to_run]))
    runner = bench_runners.SpeedupComparisonRunner(bench_rows,
                                                   bench_dims,
                                                   dataset,
                                                   input_type)
    all_results = []
    for algo in algos_to_run:
        for param_overrides in param_override_list:
            for cuml_param_overrides in cuml_param_override_list:
                results = runner.run(algo, param_overrides, cuml_param_overrides, run_cpu=run_cpu)
                for r in results:
                    all_results.append({'algo': algo.name, 'input': input_type, **r})

    print("Finished all benchmark runs")
    results_df = pd.DataFrame.from_records(all_results)
    print(results_df)

    if output_csv:
        results_df.to_csv(output_csv)
        print("Saved results to %s" % output_csv)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='bench_all',
                                     description='''
                                     Testing
                                     ''')
    parser.add_argument('--max-rows', type=int, default=100000,
                        help='Evaluate at most max_row samples')
    parser.add_argument('--min-rows', type=int, default=10000,
                        help='Evaluate at least min_rows samples')
    parser.add_argument('--num-sizes', type=int, default=2,
                        help='Number of different sizes to test')
    parser.add_argument('--num-features', type=int, default=-1)
    parser.add_argument('--quiet', '-q', action='store_false', dest='verbose', default=True)
    parser.add_argument('--csv', nargs='?')
    parser.add_argument('--dataset', default='blobs')
    parser.add_argument('--skip-cpu', action='store_true')
    parser.add_argument('--input-type', default='numpy')
    parser.add_argument('--input-dimensions', default=[64,256,512], nargs='+', type=int,
                        help='Data dimension sizes (may provide multiple sizes)')
    parser.add_argument('--param-sweep', nargs='*', type=str,
                        help='''Parameter values to vary, in the form:
                                key=val_list, where val_list may be a comma-separated list''')
    parser.add_argument('--cuml-param-sweep', nargs='*', type=str,
                        help='''Parameter values to vary for cuML only, in the form:
                                key=val_list, where val_list may be a comma-separated list''')
    parser.add_argument('--default-size', action='store_true',
                        help='Only run datasets at default size')
    parser.add_argument('algorithms', nargs='*', help='List of algorithms to run, or omit to run all')
    args = parser.parse_args()

    bench_rows = np.logspace(np.log10(args.min_rows), np.log10(args.max_rows), num=args.num_sizes, dtype=np.int32)
    bench_dims = args.input_dimensions

    if args.num_features > 0:
        bench_dims = [args.num_features]
    if args.default_size:
        bench_rows = [0]
        bench_dims = [0]

    param_override_list = extract_param_overrides(args.param_sweep)
    cuml_param_override_list = extract_param_overrides(args.cuml_param_sweep)

    if args.algorithms:
        algos_to_run = []
        for name in args.algorithms:
            algo = bench_algos.algorithm_by_name(name)
            if not algo:
                raise ValueError("No %s 'algorithm' found" % name)
            algos_to_run.append(algo)
    else:
        # Run all by default
        algos_to_run = bench_algos.all_algorithms()

    run(algos_to_run,
        dataset=args.dataset,
        bench_rows=bench_rows,
        bench_dims=bench_dims,
        input_type=args.input_type,
        param_override_list=param_override_list,
        cuml_param_override_list=cuml_param_override_list,
        output_csv=args.csv,
        run_cpu=(not args.skip_cpu)
    )
