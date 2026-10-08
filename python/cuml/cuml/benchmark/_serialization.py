# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialize resolved benchmark workloads into artifact descriptors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._utils import _jsonable
from .datasets import DATA_SEED

if TYPE_CHECKING:
    from .suite import ResolvedCase

DATA_GENERATOR = "com.nvidia.cuml.benchmark.generate-data-v1"


def dataset_descriptor(case: ResolvedCase) -> dict[str, Any]:
    """Build a generated dataset descriptor for a resolved case.

    Parameters
    ----------
    case : ResolvedCase
        Case defining the dataset and input representation.
    """
    parameters = dict(case.dataset_parameters)
    parameters["dtypes"] = case.dtypes
    # Input attributes are descriptive, not identity-bearing in schema v2.
    # Record representation here so dense and CSR workloads have distinct IDs.
    parameters["input_format"] = case.input_format
    if case.lifecycle == "inference":
        parameters.update(
            train_rows=case.training_rows,
            inference_rows=case.measured_rows,
            partition="disjoint-contiguous-v1",
            fit_input_selection=list(case.fit_input_selection),
        )
    return {
        "name": case.dataset,
        "kind": "generated",
        "parameters": parameters,
        "generator": DATA_GENERATOR,
        "fingerprint": None,
        "random_seed": DATA_SEED,
        "legacy_identity": None,
    }


def operation_descriptor(case: ResolvedCase) -> dict[str, str]:
    """Build the operation name and lifecycle descriptor.

    Parameters
    ----------
    case : ResolvedCase
        Case defining the estimator operation.
    """
    return {
        "name": case.operation,
        "lifecycle": case.lifecycle,
    }


def case_artifact_fields(case: ResolvedCase) -> dict[str, Any]:
    """Build the workload fields for a case result artifact.

    Parameters
    ----------
    case : ResolvedCase
        Resolved workload to serialize.
    """
    return {
        "algorithm": case.estimator,
        "dataset": dataset_descriptor(case),
        "operation": operation_descriptor(case),
        "input": {
            "dimensions": [
                {
                    "name": "rows",
                    "size": case.measured_rows,
                },
                {"name": "features", "size": case.features},
            ],
            "data_type": case.dtypes[
                "y" if case.input_selection == ("y",) else "X"
            ],
            "selection": list(case.input_selection),
            "attributes": {"format": "csr"}
            if case.input_format == "csr"
            else {},
        },
        "parameters": {"declared": _jsonable(case.parameters)},
    }
