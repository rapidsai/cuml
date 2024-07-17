#
# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
from cuml.benchmark.runners import run_variations
from cuml.benchmark import algorithms
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
pd = cpu_only_import("pandas")


def log_range(start, end, n):
    return np.logspace(np.log10(start), np.log10(end), num=n, dtype=np.int32)


def expand_params(key, vals):
    return [{key: v} for v in vals]


def report_asv(
    results_df, output_dir, cudaVer="", pythonVer="", osType="", machineName=""
):
    """Logs the dataframe `results_df` to airspeed velocity format.
    This writes (or appends to) JSON files in `output_dir`

    Parameters
    ----------
    results_df : pd.DataFrame
      DataFrame with one row per benchmark run
    output_dir : str
      Directory for ASV output database
    """
    import asvdb
    import platform
    import psutil

    uname = platform.uname()
    (commitHash, commitTime) = asvdb.utils.getCommitInfo()

    b_info = asvdb.BenchmarkInfo(
        machineName=machineName or uname.machine,
        cudaVer=cudaVer or "unknown",
        osType=osType or "%s %s" % (uname.system, uname.release),
        pythonVer=pythonVer or platform.python_version(),
        commitHash=commitHash,
        commitTime=commitTime,
        gpuType="unknown",
        cpuType=uname.processor,
        arch=uname.machine,
        ram="%d" % psutil.virtual_memory().total,
    )
    (
        repo,
        branch,
    ) = asvdb.utils.getRepoInfo()  # gets repo info from CWD by default

    db = asvdb.ASVDb(dbDir=output_dir, repo=repo, branches=[branch])

    for index, row in results_df.iterrows():
        val_keys = ["cu_time", "cpu_time", "speedup", "cuml_acc", "cpu_acc"]
        params = [(k, v) for k, v in row.items() if k not in val_keys]
        result = asvdb.BenchmarkResult(
            row["algo"], params, result=row["cu_time"]
        )
        db.addResult(b_info, result)


preprocessing_algo_defs = [
    (
        "StandardScaler",
        "classification",
        [1000000],
        [256, 1024],
        [{"copy": False}],
    ),
    (
        "MinMaxScaler",
        "classification",
        [1000000],
        [256, 1024],
        [{"copy": False}],
    ),
    (
        "MaxAbsScaler",
        "classification",
        [1000000],
        [256, 1024],
        [{"copy": False}],
    ),
    (
        "Normalizer",
        "classification",
        [1000000],
        [256, 1024],
        [{"copy": False}],
    ),
    (
        "RobustScaler",
        "classification",
        [1000000],
        [128, 256],
        [{"copy": False}],
    ),
    (
        "SimpleImputer",
        "classification",
        [1000000],
        [256, 1024],
        [{"copy": False}],
    ),
    ("PolynomialFeatures", "classification", [1000000], [128, 256], [{}]),
    (
        "SparseCSRStandardScaler",
        "classification",
        [1000000],
        [512],
        [{"copy": False, "with_mean": False}],
    ),
    (
        "SparseCSRMaxAbsScaler",
        "classification",
        [300000],
        [512],
        [{"copy": False}],
    ),
    (
        "SparseCSRNormalizer",
        "classification",
        [1000000],
        [512],
        [{"copy": False}],
    ),
    (
        "SparseCSCRobustScaler",
        "classification",
        [1000000],
        [512],
        [{"copy": False, "with_centering": False}],
    ),
    (
        "SparseCSCSimpleImputer",
        "classification",
        [1000000],
        [512],
        [{"copy": False}],
    ),
    ("SparseCSRPolynomialFeatures", "classification", [30000], [128], [{}]),
]

preprocessing_algo_names = set([a[0] for a in preprocessing_algo_defs])


def make_bench_configs(long_config):
    """Defines the configurations we want to benchmark
    If `long_config` is True, this may take over an hour.
    If False, the run should take only a few minutes."""

    configs = []
    if long_config:
        # Use large_rows for pretty fast algos,
        # use small_rows for slower ones
        small_rows = log_range(10000, 1000000, 2)
        large_rows = log_range(1e5, 1e7, 2)
    else:
        # Small config only runs a single size
        small_rows = log_range(20000, 20000, 1)
        large_rows = log_range(100000, 100000, 1)

    default_dims = [16, 256]

    # Add all the simple algorithms that don't need special treatment
    algo_defs = [
        ("KMeans", "blobs", small_rows, default_dims, [{}]),
        ("DBScan", "blobs", small_rows, default_dims, [{}]),
        ("TSNE", "blobs", small_rows, default_dims, [{}]),
        ("NearestNeighbors", "blobs", small_rows, default_dims, [{}]),
        ("MBSGDClassifier", "blobs", large_rows, default_dims, [{}]),
        (
            "LogisticRegression",
            "classification",
            large_rows,
            default_dims,
            [{}],
        ),
        ("LinearRegression", "regression", large_rows, default_dims, [{}]),
        ("Lasso", "regression", large_rows, default_dims, [{}]),
        ("ElasticNet", "regression", large_rows, default_dims, [{}]),
        (
            "PCA",
            "blobs",
            large_rows,
            [32, 256],
            expand_params("n_components", [2, 25]),
        ),
        (
            "tSVD",
            "blobs",
            large_rows,
            [32, 256],
            expand_params("n_components", [2, 25]),
        ),
        (
            "GaussianRandomProjection",
            "blobs",
            large_rows,
            [32, 256],
            expand_params("n_components", [2, 25]),
        ),
    ]

    algo_defs += preprocessing_algo_defs

    for algo_name, dataset_name, rows, dims, params in algo_defs:
        configs.append(
            dict(
                algo_name=algo_name,
                dataset_name=dataset_name,
                bench_rows=rows,
                bench_dims=dims,
                param_override_list=params,
            )
        )

    # Explore some more interesting params for RF
    if long_config:
        configs += [
            dict(
                algo_name="RandomForestClassifier",
                dataset_name="classification",
                bench_rows=small_rows,
                bench_dims=default_dims,
                cuml_param_override_list=[
                    {"n_bins": [8, 32]},
                    {"max_features": ["sqrt", 1.0]},
                ],
            )
        ]

    return configs


bench_config = {
    "short": make_bench_configs(False),
    "long": make_bench_configs(True),
}

if __name__ == "__main__":
    import argparse

    allAlgoNames = set(
        [v["algo_name"] for tuples in bench_config.values() for v in tuples]
    )

    parser = argparse.ArgumentParser(
        prog="ci_benchmark",
        description="""
                                     Tool for running benchmarks in CI
                                     """,
    )
    parser.add_argument(
        "--benchmark", type=str, choices=bench_config.keys(), default="short"
    )

    parser.add_argument(
        "--algo",
        type=str,
        action="append",
        help='Algorithm to run, must be one of %s, or "ALL"'
        % ", ".join(['"%s"' % k for k in allAlgoNames]),
    )
    parser.add_argument(
        "--update_asv_dir",
        type=str,
        help="Add results to the specified ASV dir in ASV " "format",
    )
    parser.add_argument(
        "--report_cuda_ver",
        type=str,
        default="",
        help="The CUDA version to include in reports",
    )
    parser.add_argument(
        "--report_python_ver",
        type=str,
        default="",
        help="The Python version to include in reports",
    )
    parser.add_argument(
        "--report_os_type",
        type=str,
        default="",
        help="The OS type to include in reports",
    )
    parser.add_argument(
        "--report_machine_name",
        type=str,
        default="",
        help="The machine name to include in reports",
    )
    parser.add_argument("--n_reps", type=int, default=3)

    args = parser.parse_args()

    algos = set(args.algo)
    if "preprocessing" in algos:
        algos = algos.union(preprocessing_algo_names)
        algos.remove("preprocessing")
    invalidAlgoNames = algos - allAlgoNames
    if invalidAlgoNames:
        raise ValueError("Invalid algo name(s): %s" % invalidAlgoNames)

    bench_to_run = bench_config[args.benchmark]

    default_args = dict(run_cpu=True, n_reps=args.n_reps)
    all_results = []
    for cfg_in in bench_to_run:
        if (
            (algos is None)
            or ("ALL" in algos)
            or (cfg_in["algo_name"] in algos)
        ):
            # Pass an actual algo object instead of an algo_name string
            cfg = cfg_in.copy()
            algo = algorithms.algorithm_by_name(cfg_in["algo_name"])
            cfg["algos"] = [algo]
            alg_name = cfg["algo_name"]
            if alg_name.startswith("Sparse"):
                if alg_name.startswith("SparseCSR"):
                    input_type = "scipy-sparse-csr"
                elif alg_name.startswith("SparseCSC"):
                    input_type = "scipy-sparse-csc"
            else:
                input_type = "numpy"
            del cfg["algo_name"]
            res = run_variations(
                **{**default_args, **cfg}, input_type=input_type
            )
            all_results.append(res)

    results_df = pd.concat(all_results)
    print(results_df)
    if args.update_asv_dir:
        report_asv(
            results_df,
            args.update_asv_dir,
            cudaVer=args.report_cuda_ver,
            pythonVer=args.report_python_ver,
            osType=args.report_os_type,
            machineName=args.report_machine_name,
        )
