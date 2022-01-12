#
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
from cuml.benchmark import datagen, algorithms

parser = argparse.ArgumentParser(
    prog='launch-benchmark',
    description=r'''
    Command-line cuML benchmark runner.

    Examples:
        python run_benchmarks.py \
            --algo LinearRegression \
            --dataset_type regression
    ''',
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    '--algo',
    type=str,
    default='',
    help='Algorithm name',
)
parser.add_argument(
    '--dataset_type',
    type=str,
    default='',
    help='Dataset type',
)
parser.add_argument(
    '--n_samples',
    type=int,
    default=10000,
    help='Number of samples',
)
parser.add_argument(
    '--n_features',
    type=int,
    default=100,
    help='Number of features',
)
parser.add_argument(
    '--input_type',
    type=str,
    default='cupy',
    help='Input type',
)
parser.add_argument(
    '--data_kwargs',
    type=json.loads,
    default={},
    help='Data generation options',
)
parser.add_argument(
    '--algo_args',
    type=json.loads,
    default={},
    help='Algorithm options',
)
parser.add_argument(
    '--json',
    type=str,
    default='',
    help='JSON file containing benchmark parameters',
)
args = parser.parse_args()


def parse_json(args):
    with open(args.json) as json_file:
        params = json.load(json_file)

    # Overwriting
    if 'algo' in params:
        args.algo = params['algo']
    if 'dataset_type' in params:
        args.dataset_type = params['dataset_type']
    if 'n_samples' in params:
        args.n_samples = params['n_samples']
    if 'n_features' in params:
        args.n_features = params['n_features']
    if 'input_type' in params:
        args.input_type = params['input_type']
    if 'data_kwargs' in params:
        args.data_kwargs = params['data_kwargs']
    if 'algo_args' in params:
        args.algo_args = params['algo_args']


if len(args.json):
    parse_json(args)

algo = algorithms.algorithm_by_name(args.algo)
data = datagen.gen_data(
    args.dataset_type,
    args.input_type,
    n_samples=args.n_samples,
    n_features=args.n_features,
    **args.data_kwargs
)
algo.run_cuml(data, **args.algo_args)
