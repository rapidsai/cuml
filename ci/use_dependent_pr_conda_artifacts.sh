#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# rapids-pre-commit-hooks: disable-next-line
# Integration-only: remove this file and all `source` lines before merging to release/26.06.
# See https://docs.rapids.ai/resources/github-actions/#using-conda-ci-artifacts-in-other-prs

# shellcheck disable=SC2034
RAFT_PR=3005
# shellcheck disable=SC2034
CUVS_PR=2043

rapids-logger "Downloading RAFT PR #${RAFT_PR} conda artifacts"
LIBRAFT_CHANNEL=$(rapids-get-pr-artifact raft "${RAFT_PR}" cpp conda)

rapids-logger "Downloading cuVS PR #${CUVS_PR} conda artifacts"
LIBCUVS_CHANNEL=$(rapids-get-pr-artifact cuvs "${CUVS_PR}" cpp conda)

rapids-logger "Downloading RAFT PR #${RAFT_PR} python conda artifacts"
PYLIBRAFT_CHANNEL=$(rapids-get-pr-artifact raft "${RAFT_PR}" python conda --stable)

rapids-logger "Downloading cuVS PR #${CUVS_PR} python conda artifacts"
CUVS_PY_CHANNEL=$(rapids-get-pr-artifact cuvs "${CUVS_PR}" python conda --stable)

RAPIDS_PREPENDED_CONDA_CHANNELS=(
  "${LIBRAFT_CHANNEL}"
  "${LIBCUVS_CHANNEL}"
  "${PYLIBRAFT_CHANNEL}"
  "${CUVS_PY_CHANNEL}"
)
export RAPIDS_PREPENDED_CONDA_CHANNELS

for _channel in "${RAPIDS_PREPENDED_CONDA_CHANNELS[@]}"; do
  conda config --system --add channels "${_channel}"
done
