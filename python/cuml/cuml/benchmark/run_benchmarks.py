#
# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
"""Command-line ML benchmark runner"""

import json
from cuml.benchmark import algorithms, datagen, runners
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")


PrecisionMap = {
    "fp32": np.float32,
    "fp64": np.float64,
}


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
        vals = val_string.split(",")

        if not isinstance(vals, list):
            vals = [vals]  # Handle single-element sweep cleanly

        # use json loads to convert to correct data type
        for idx, val in enumerate(vals):
            try:
                vals[idx] = json.loads(val)
            except ValueError:
                pass

        single_param_lists.append([(key, val) for val in vals])

    # Create dicts with the cartesian product of all arg-based lists
    tuple_list = itertools.product(*single_param_lists)
    dict_list = [dict(tl) for tl in tuple_list]
    return dict_list


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="run_benchmarks",
        description=r"""
        Command-line benchmark runner, logging results to
        stdout and/or CSV.

        Examples:
          # Simple logistic regression
          python run_benchmarks.py --dataset classification LogisticRegression

          # Compare impact of RF parameters and data sets for multiclass
          python run_benchmarks.py --dataset classification  \
                --max-rows 100000 --min-rows 10000 \
                --dataset-param-sweep n_classes=[2,8] \
                --cuml-param-sweep n_bins=[4,16] n_estimators=[10,100] \
                --csv results.csv \
                RandomForestClassifier

          # Run a bunch of clustering and dimensionality reduction algorithms
          # (Because `--input-dimensions` takes a varying number of args, you
          # need the extra `--` to separate it from the algorithm names
          python run_benchmarks.py --dataset blobs \
                --max-rows 20000 --min-rows 20000 --num-sizes 1 \
                --input-dimensions 16 256 \
                -- DBSCAN KMeans TSNE PCA UMAP

          # Use a real dataset at its default size
          python run_benchmarks.py --dataset higgs --default-size \
                RandomForestClassifier LogisticRegression

        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=100000,
        help="Evaluate at most max_row samples",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=10000,
        help="Evaluate at least min_rows samples",
    )
    parser.add_argument(
        "--num-sizes",
        type=int,
        default=2,
        help="Number of different sizes to test",
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=None,
        metavar="N",
        help="Shortcut for --min-rows N --max-rows N --num-sizes 1",
    )
    parser.add_argument("--num-features", type=int, default=-1)
    parser.add_argument(
        "--quiet", "-q", action="store_false", dest="verbose", default=True
    )
    parser.add_argument("--csv", nargs="?")
    parser.add_argument("--dataset", default="blobs")
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--input-type", default="numpy")
    parser.add_argument(
        "--test-split",
        default=0.1,
        type=float,
        help="Fraction of input data used for testing (between 0.0 and 1.0)",
    )
    parser.add_argument(
        "--input-dimensions",
        default=[64, 256, 512],
        nargs="+",
        type=int,
        help="Data dimension sizes (may provide multiple sizes)",
    )
    parser.add_argument(
        "--param-sweep",
        nargs="*",
        type=str,
        help="""Parameter values to vary, in the form:
                key=val_list, where val_list may be a comma-separated list""",
    )
    parser.add_argument(
        "--cuml-param-sweep",
        nargs="*",
        type=str,
        help="""Parameter values to vary for cuML only, in the form:
                key=val_list, where val_list may be a comma-separated list""",
    )
    parser.add_argument(
        "--cpu-param-sweep",
        nargs="*",
        type=str,
        help="""Parameter values to vary for CPU only, in the form:
                key=val_list, where val_list may be a comma-separated list""",
    )
    parser.add_argument(
        "--dataset-param-sweep",
        nargs="*",
        type=str,
        help="""Parameter values to vary for dataset generator, in the form
                key=val_list, where val_list may be a comma-separated list""",
    )
    parser.add_argument(
        "--default-size",
        action="store_true",
        help="Only run datasets at default size",
    )
    parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="Throw exception on a failed benchmark",
    )
    parser.add_argument(
        "--print-algorithms",
        action="store_true",
        help="Print the list of all available algorithms and exit",
    )
    parser.add_argument(
        "--print-datasets",
        action="store_true",
        help="Print the list of all available datasets and exit",
    )
    parser.add_argument(
        "algorithms",
        nargs="*",
        help="List of algorithms to run, or omit to run all",
    )
    parser.add_argument("--n-reps", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp64"],
        default="fp32",
        help="Precision of the dataset to benchmark with",
    )
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default=["gpu"],
        nargs="+",
        help="The device to use for cuML execution",
    )
    args = parser.parse_args()

    args.dtype = PrecisionMap[args.dtype]

    if args.print_algorithms:
        for algo in algorithms.all_algorithms():
            print(algo.name)
        sys.exit()

    if args.print_datasets:
        for dataset in datagen.all_datasets().keys():
            print(dataset)
        sys.exit()

    if not 0.0 <= args.test_split <= 1.0:
        raise ValueError(
            "test_split: got %f, want a value between 0.0 and 1.0"
            % args.test_split
        )

    bench_rows = np.logspace(
        np.log10(args.min_rows),
        np.log10(args.max_rows),
        num=args.num_sizes,
        dtype=np.int32,
    )

    bench_dims = args.input_dimensions

    if args.num_rows is not None:
        bench_rows = [args.num_rows]

    if args.num_features > 0:
        bench_dims = [args.num_features]
    if args.default_size:
        bench_rows = [0]
        bench_dims = [0]

    param_override_list = extract_param_overrides(args.param_sweep)
    cuml_param_override_list = extract_param_overrides(args.cuml_param_sweep)
    cpu_param_override_list = extract_param_overrides(args.cpu_param_sweep)
    dataset_param_override_list = extract_param_overrides(
        args.dataset_param_sweep
    )

    if args.algorithms:
        algos_to_run = []
        for name in args.algorithms:
            algo = algorithms.algorithm_by_name(name)
            if not algo:
                raise ValueError("No %s 'algorithm' found" % name)
            algos_to_run.append(algo)
    else:
        # Run all by default
        algos_to_run = algorithms.all_algorithms()

    results_df = runners.run_variations(
        algos_to_run,
        dataset_name=args.dataset,
        bench_rows=bench_rows,
        bench_dims=bench_dims,
        input_type=args.input_type,
        test_fraction=args.test_split,
        param_override_list=param_override_list,
        cuml_param_override_list=cuml_param_override_list,
        cpu_param_override_list=cpu_param_override_list,
        dataset_param_override_list=dataset_param_override_list,
        dtype=args.dtype,
        run_cpu=(not args.skip_cpu),
        device_list=args.device,
        raise_on_error=args.raise_on_error,
        n_reps=args.n_reps,
    )

    if args.csv:
        results_df.to_csv(args.csv)
        print("Saved results to %s" % args.csv)
