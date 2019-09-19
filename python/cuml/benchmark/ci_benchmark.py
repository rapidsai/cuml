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
# limitations under the License.`
#
"""Script to benchmark cuML modules in CI

NOTE: This is currently experimental as the ops team builds out the CI
platform to support benchmark reporting.
"""
from cuml.benchmark.bench_all import run_variations, report_asv
from cuml.benchmark import bench_algos
import numpy as np
import pandas as pd

def log_range(start, end, n):
    return np.logspace(np.log10(start),
                       np.log10(end),
                       num=n,
                       dtype=np.int32)

def expand_params(key, vals):
    return [{key: v} for v in vals]

def make_bench_configs(long_config):
    """Defines the configurations we want to benchmark
    If `long_config` is True, this may take over an hour.
    If False, the run should take only a few minutes."""

    configs = []
    if long_config:
        # Use large_rows for pretty fast algos,
        # use small_rows for slower ones
        small_rows = log_range(10000, 1000000, 3)
        large_rows = log_range(1e5, 1e7, 3)
    else:
        # Small config only runs a single size
        small_rows = log_range(2000, 2000, 1)
        large_rows = log_range(10000, 10000, 1)

    default_dims = [16,256]

    # Add all the simple algorithms that don't need special treatment
    algo_defs = [
        ("KMeans", "blobs", small_rows, default_dims, [{}]),
        ("DBScan", "blobs", large_rows, default_dims, [{}]),
        ("TSNE", "blobs", small_rows, default_dims, [{}]),
        ("NearestNeighbors", "blobs", small_rows, default_dims, [{}]),
        ("MBSGDClassifier", "blobs", large_rows, default_dims, [{}]),
        ("LogisticRegression", "classification", large_rows, default_dims,
         [{}]),
        ("LinearRegression", "regression", large_rows, default_dims, [{}]),
        ("Lasso", "regression", large_rows, default_dims, [{}]),
        ("ElasticNet", "regression", large_rows, default_dims, [{}]),
        ("PCA", "blobs", large_rows, [32,256],
         expand_params("n_components", [2,25])),
        ("tSVD", "blobs", large_rows, [32,256],
         expand_params("n_components", [2,25]))
    ]

    for algo_name, dataset_name, rows, dims, params in algo_defs:
        configs.append(dict(algo_name=algo_name,
                            dataset_name=dataset_name,
                            bench_rows=rows,
                            bench_dims=dims,
                            param_override_list=params))

    # Explore some more interesting params for RF
    if long_config:
        configs += [dict(algo_name="RandomForestClassifier",
                   dataset_name="classification",
                   bench_rows=small_rows,
                   bench_dims=default_dims,
                   cuml_param_override_list=[
                       {"n_bins": [8, 32]},
                       {"split_algo": [0,1]},
                       {"max_features": ['sqrt', 1.0]}
                   ])]

    return configs

bench_config = {
    "short": make_bench_configs(False),
    "long": make_bench_configs(True)
}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='ci_benchmark',
                                     description='''
                                     Tool for running benchmarks in CI
                                     ''')
    parser.add_argument("--benchmark", type=str,
                        choices=bench_config.keys(),
                        default="short")

    parser.add_argument('--asvdb', required=False, type=str)
    args = parser.parse_args()
    bench_to_run = bench_config[args.benchmark]

    default_args = dict(run_cpu=False)
    all_results = []
    for cfg_in in bench_to_run:
        # Pass an actual algo object instead of an algo_name string
        cfg = cfg_in.copy()
        algo = bench_algos.algorithm_by_name(cfg_in["algo_name"])
        cfg["algos"] = [algo]
        del cfg["algo_name"]
        res = run_variations(**{**default_args, **cfg})
        all_results.append(res)

    results_df = pd.concat(all_results)
    print(results_df)
    if args.asvdb:
        report_asv(results_df, args.asvdb)
