# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide GPU benchmark execution and cuML source provenance."""

from typing import Any

from .._utils import _source
from .base import GPUBackend


class CumlBackend(GPUBackend):
    """Provide cuML package metadata and GPU synchronization."""

    extra_packages = ("cuml",)

    def package_source(self, package: str) -> dict[str, Any] | None:
        """Return source provenance for cuML packages.

        Parameters
        ----------
        package : str
            Installed package name.
        """
        # Record cuML source provenance alongside its version in the artifact.
        return (
            _source() if package == "cuml" else super().package_source(package)
        )
