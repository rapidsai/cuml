# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define backend hooks for benchmark execution and verification."""

from __future__ import annotations

import contextlib
import importlib
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .._utils import _version
from ..registry import EstimatorSpec

if TYPE_CHECKING:
    from ..suite import ResolvedCase, Suite


@dataclass
class VerificationResult:
    """Store observation verification evidence and an optional failure."""

    extensions: dict[str, Any] = field(default_factory=dict)
    failure: dict[str, Any] | None = None


class Backend:
    """Provide estimator, runtime, and metadata hooks for benchmarks."""

    extra_packages: tuple[str, ...] = ()

    def __init__(self, name: str, catalog: dict[str, EstimatorSpec]) -> None:
        """Initialize the backend name and supported estimator catalog."""
        self.name = name
        self.catalog = catalog

    def worker_environment(self) -> dict[str, str]:
        """Return an isolated startup environment without mutating the caller.

        Do not inherit acceleration into CPU or native workers. The accel
        backend also starts clean and enables itself through bootstrap_process.
        Backends may override this hook for other process-startup requirements.
        """
        environment = os.environ.copy()
        environment.pop("CUML_ACCEL_ENABLED", None)
        return environment

    def bootstrap_process(self) -> None:
        """Bootstrap process startup for the backend."""
        pass

    def prepare_process(self) -> None:
        """Prepare process prerequisites before benchmark execution."""
        pass

    def validate_profile(self, profile_name: str, warmups: int) -> None:
        """Validate backend-specific execution profile requirements.

        Parameters
        ----------
        profile_name : str
            Selected profile name.
        warmups : int
            Number of warmup repetitions.
        """
        pass

    def estimator_spec(self, estimator: str) -> EstimatorSpec:
        """Look up a supported estimator's import specification.

        Parameters
        ----------
        estimator : str
            Estimator name in the backend catalog.
        """
        try:
            return self.catalog[estimator]
        except KeyError as exc:
            raise ValueError(
                f"estimator {estimator!r} is not supported by {self.name!r}"
            ) from exc

    def load_estimator(self, spec: EstimatorSpec) -> type[Any]:
        """Import an estimator class from its specification.

        Parameters
        ----------
        spec : EstimatorSpec
            Estimator module and class name.
        """
        return getattr(importlib.import_module(spec.module), spec.name)

    def construct_estimator(
        self,
        estimator_class: type[Any],
        parameters: dict[str, Any],
        runtime: Any = None,
    ) -> Any:
        """Construct an estimator with the declared parameters.

        Parameters
        ----------
        estimator_class : type
            Estimator class to instantiate.
        parameters : dict
            Estimator constructor arguments.
        runtime : Any, optional
            Backend runtime context, unused by the default implementation.
        """
        return estimator_class(**parameters)

    def convert_data(
        self, case: ResolvedCase, X: Any, y: Any
    ) -> tuple[Any, Any]:
        """Convert generated inputs to the backend's data representation.

        Parameters
        ----------
        case : ResolvedCase
            Case defining the input representation.
        X : Any
            Complete generated feature data.
        y : Any
            Complete generated target data.
        """
        # Called on the complete generated dataset, BEFORE contiguous slicing.
        return X, y

    def runtime(self, suite: Suite) -> contextlib.AbstractContextManager[Any]:
        """Provide the runtime context for suite execution.

        Parameters
        ----------
        suite : Suite
            Suite whose runtime is being prepared.
        """
        return contextlib.nullcontext()

    def synchronize(self, value: Any = None) -> None:
        """Wait for backend work to complete before timing boundaries.

        Parameters
        ----------
        value : Any, optional
            Operation output to synchronize, if applicable.
        """
        pass

    def instrumentation(
        self, suite: Suite, role: str, repetition: int
    ) -> contextlib.AbstractContextManager[VerificationResult]:
        """Provide observation instrumentation and verification state.

        Parameters
        ----------
        suite : Suite
            Suite being executed.
        role : str
            Observation role: warmup or measurement.
        repetition : int
            Zero-based repetition index within the role.
        """
        return contextlib.nullcontext(VerificationResult())

    def package_source(self, package: str) -> dict[str, Any] | None:
        """Return available source provenance for a software package.

        Parameters
        ----------
        package : str
            Installed package name.
        """
        return None

    def software_packages(
        self, cases: Iterable[ResolvedCase]
    ) -> list[dict[str, Any]]:
        """Collect software package metadata used by the cases.

        Parameters
        ----------
        cases : iterable of ResolvedCase
            Cases whose estimator packages should be recorded.
        """
        names = {self.estimator_spec(c.estimator).package for c in cases}
        names.update(self.extra_packages)
        return [
            {
                "name": name,
                "version": _version(name),
                "build": None,
                "source": self.package_source(name),
            }
            for name in sorted(names)
        ]


class GPUBackend(Backend):
    """Provide device synchronization for GPU benchmark backends."""

    def synchronize(self, value: Any = None) -> None:
        """Wait for work on the current CUDA device to complete.

        Parameters
        ----------
        value : Any, optional
            Operation output, unused by device synchronization.
        """
        try:
            importlib.import_module("cupy").cuda.runtime.deviceSynchronize()
        except Exception:
            # Preserve the estimator's original exception path when GPU
            # synchronization is unavailable; execution will fail clearly.
            pass
