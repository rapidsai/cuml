#
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import argparse
import json

from cuml.benchmark import algorithms, datagen
from cuml.benchmark.automated.utils.utils import setup_bench

parser = argparse.ArgumentParser(
    prog="launch-benchmark",
    description=r"""
    Command-line cuML benchmark runner.

    Examples:
        python run_benchmarks.py \
            --algo_name LinearRegression \
            --dataset_type regression
    """,
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "--algo_name",
    type=str,
    default="",
    help="Algorithm name",
)
parser.add_argument(
    "--dataset_type",
    type=str,
    default="",
    help="Dataset type",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=10000,
    help="Number of samples",
)
parser.add_argument(
    "--n_features",
    type=int,
    default=100,
    help="Number of features",
)
parser.add_argument(
    "--dataset_format",
    type=str,
    default="cupy",
    help="Dataset format",
)
parser.add_argument(
    "--data_kwargs",
    type=json.loads,
    default={},
    help="Data generation options",
)
parser.add_argument(
    "--setup_kwargs",
    type=json.loads,
    default={},
    help="Algorithm setup options",
)
parser.add_argument(
    "--training_kwargs",
    type=json.loads,
    default={},
    help="Algorithm training options",
)
parser.add_argument(
    "--inference_kwargs",
    type=json.loads,
    default={},
    help="Algorithm inference options",
)
parser.add_argument(
    "--json",
    type=str,
    default="",
    help="JSON file containing benchmark parameters",
)
args = parser.parse_args()


def parse_json(args):
    with open(args.json) as json_file:
        params = json.load(json_file)

    # Overwriting
    if "algo_name" in params:
        args.algo_name = params["algo_name"]
    if "dataset_type" in params:
        args.dataset_type = params["dataset_type"]
    if "n_samples" in params:
        args.n_samples = params["n_samples"]
    if "n_features" in params:
        args.n_features = params["n_features"]
    if "dataset_format" in params:
        args.dataset_format = params["dataset_format"]
    if "data_kwargs" in params:
        args.data_kwargs = params["data_kwargs"]
    if "setup_kwargs" in params:
        args.setup_kwargs = params["setup_kwargs"]
    if "training_kwargs" in params:
        args.training_kwargs = params["training_kwargs"]
    if "inference_kwargs" in params:
        args.inference_kwargs = params["inference_kwargs"]


if len(args.json):
    parse_json(args)

dataset = datagen.gen_data(
    args.dataset_type,
    args.dataset_format,
    n_samples=args.n_samples,
    n_features=args.n_features,
    **args.data_kwargs,
)

algo = algorithms.algorithm_by_name(args.algo_name)
cuml_setup = setup_bench(
    "cuml", algo, "inference", dataset, args.setup_kwargs, args.training_kwargs
)
algo.run_cuml(dataset, bench_args=args.inference_kwargs, **cuml_setup)
