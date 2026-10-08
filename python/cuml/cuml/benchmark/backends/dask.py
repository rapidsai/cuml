# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run distributed cuML benchmarks on a local multi-GPU cluster."""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..suite import ResolvedCase, Suite

from .cuml import CumlBackend


class DaskBackend(CumlBackend):
    """Provide distributed inputs, estimators, and a multi-GPU runtime."""

    def convert_data(
        self, case: ResolvedCase, X: Any, y: Any
    ) -> tuple[Any, Any]:
        """Convert generated inputs to partitioned Dask collections.

        Parameters
        ----------
        case : ResolvedCase
            Case defining the dataset shape and representation.
        X : Any
            Complete generated feature data.
        y : Any
            Complete generated target data.
        """
        da = importlib.import_module("dask.array")
        chunks = (max(1, case.generated_rows // 2), case.features)
        if case.dataset == "categorical" and case.estimator not in {
            "LabelEncoder",
            "LabelBinarizer",
        }:
            cudf = importlib.import_module("cudf")
            dask_cudf = importlib.import_module("dask_cudf")
            X = dask_cudf.from_cudf(cudf.from_pandas(X), npartitions=2)
        else:
            X = da.from_array(X, chunks=chunks)
        y = da.from_array(y, chunks=(chunks[0],))
        return X, y

    def construct_estimator(
        self,
        estimator_class: type[Any],
        parameters: dict[str, Any],
        runtime: Any = None,
    ) -> Any:
        """Construct a distributed estimator using the runtime client.

        Parameters
        ----------
        estimator_class : type
            Distributed estimator class to instantiate.
        parameters : dict
            Estimator constructor arguments.
        runtime : Any, optional
            Dask client passed to the estimator.
        """
        return estimator_class(**{**parameters, "client": runtime})

    def synchronize(self, value: Any = None) -> None:
        """Wait for distributed output and local GPU work to complete.

        Parameters
        ----------
        value : Any, optional
            Distributed output to wait for or compute.
        """
        if value is not None:
            try:
                importlib.import_module("dask.distributed").wait(value)
            except (TypeError, AttributeError):
                if hasattr(value, "compute"):
                    value.compute()
        super().synchronize(value)

    @contextlib.contextmanager
    def runtime(self, suite: Suite) -> Iterator[Any]:
        """Create a local multi-GPU cluster and yield its Dask client.

        Parameters
        ----------
        suite : Suite
            Suite to validate for distributed execution.
        """
        from ..suite import SuiteError

        if any(case.input_format == "csr" for case in suite.cases):
            raise SuiteError("Dask CSR inputs are not supported yet")
        if any(case.timeout_sec is not None for case in suite.cases):
            raise SuiteError("Dask case timeouts are not supported yet")

        count = importlib.import_module("cupy").cuda.runtime.getDeviceCount()
        if count < 2:
            raise SuiteError(
                f"cuml.dask suite requires at least two visible GPUs; found {count}"
            )
        LocalCUDACluster = importlib.import_module(
            "dask_cuda"
        ).LocalCUDACluster
        Client = importlib.import_module("dask.distributed").Client
        with LocalCUDACluster(
            n_workers=count, threads_per_worker=1
        ) as cluster:
            with Client(cluster) as client:
                yield client
