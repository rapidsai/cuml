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
from cuml.benchmark.bench_all import run_variations, report_asv
from cuml.benchmark import bench_algos
import numpy as np
import pandas as pd

def log_range(start, end, n):
    return np.logspace(np.log10(start),
                       np.log10(end),
                       num=n,
                       dtype=np.int32)


bench_config_short = [
    dict(algo="KMeans",
         dataset="blobs",
         bench_rows=log_range(100, 1000, 2),
         bench_dims=[32,256]),
    dict(algo="LogisticRegression",
         dataset="classification",
         bench_rows=log_range(1000, 100000, 2),
         bench_dims=[32,256]),
]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='ci_benchmark',
                                     description='''
                                     Tool for running benchmarks in CI
                                     ''')
    parser.add_argument("--benchmark", type=str,
                        choices=["short", "long"],
                        default="short")

    parser.add_argument('--asvdb', required=True, type=str)
    args = parser.parse_args()

    if args.benchmark == "short":
        bench_to_run = bench_config_short
    else:
        raise ValueError("Long config not ready yet")
        # TODO: add "short" and "long" configs"

    default_args = dict(run_cpu=False)
    all_results = []
    for cfg_in in bench_to_run:
        # XXX UGLY!
        cfg = cfg_in.copy()
        algo = bench_algos.algorithm_by_name(cfg_in["algo"])
        cfg["algos"] = [algo]
        del cfg["algo"]
        res = run_variations(**{**default_args, **cfg})
        all_results.append(res)

    results_df = pd.concat(all_results)
    print(results_df)
    report_asv(results_df, args.asvdb)
