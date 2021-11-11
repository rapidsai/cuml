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
    default='numpy',
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
args = parser.parse_args()

algo = algorithms.algorithm_by_name(args.algo)
data = datagen.gen_data(
    args.dataset_type,
    args.input_type,
    n_samples=args.n_samples,
    n_features=args.n_features,
    **args.data_kwargs
)
algo.run_cuml(data, **args.algo_args)
