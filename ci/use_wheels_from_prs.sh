#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# initialize PIP_CONSTRAINT
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

# download wheels, store the directories holding them in variables
LIBRAFT_WHEELHOUSE=$(rapids-get-pr-artifact raft 3052 cpp wheel)
PYLIBRAFT_WHEELHOUSE=$(rapids-get-pr-artifact raft 3052 python wheel --pkg_name pylibraft)
LIBCUVS_WHEELHOUSE=$(rapids-get-pr-artifact cuvs 2227 cpp wheel)

# write a pip constraints file saying e.g. "whenever you encounter a requirement for 'librmm-cu12', use this wheel"
cat > "${PIP_CONSTRAINT}" <<EOF
libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRAFT_WHEELHOUSE}"/libraft_*.whl)
raft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${PYLIBRAFT_WHEELHOUSE}"/pylibraft_*.whl)
libcuvs-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUVS_WHEELHOUSE}"/libcuvs_*.whl)
EOF
