#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# rapids-pre-commit-hooks: disable-next-line
# Integration-only: remove this file and all `source` lines before merging to release/26.06.
# See https://docs.rapids.ai/resources/github-actions/#using-wheel-ci-artifacts-in-other-prs
#
# Must be sourced after `source rapids-init-pip` (and after `rapids-generate-pip-constraints`
# in test_wheel*.sh jobs).

# shellcheck disable=SC2034
RAFT_PR=3005
# shellcheck disable=SC2034
CUVS_PR=2043

if [[ -z "${PIP_CONSTRAINT:-}" || ! -f "${PIP_CONSTRAINT}" ]]; then
  rapids-logger "ERROR: PIP_CONSTRAINT must be set (source rapids-init-pip first)" >&2
  exit 1
fi

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

rapids-logger "Downloading RAFT PR #${RAFT_PR} cpp wheel artifacts"
LIBRAFT_WH=$(
  RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" \
    rapids-get-pr-artifact raft "${RAFT_PR}" cpp wheel
)

rapids-logger "Downloading cuVS PR #${CUVS_PR} cpp wheel artifacts"
LIBCUVS_WH=$(
  RAPIDS_PY_WHEEL_NAME="libcuvs_${RAPIDS_PY_CUDA_SUFFIX}" \
    rapids-get-pr-artifact cuvs "${CUVS_PR}" cpp wheel
)

rapids-logger "Downloading RAFT PR #${RAFT_PR} python wheel artifacts"
PYLIBRAFT_WH=$(rapids-get-pr-artifact raft "${RAFT_PR}" python wheel --stable)

rapids-logger "Downloading cuVS PR #${CUVS_PR} python wheel artifacts"
CUVS_WH=$(rapids-get-pr-artifact cuvs "${CUVS_PR}" python wheel --stable)

cat >>"${PIP_CONSTRAINT}" <<EOF
libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRAFT_WH}"/libraft_*.whl)
libcuvs-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUVS_WH}"/libcuvs_*.whl)
pylibraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${PYLIBRAFT_WH}"/pylibraft_*.whl)
cuvs-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${CUVS_WH}"/cuvs_*.whl)
EOF
