#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# initialize PIP_CONSTRAINT
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

# download wheels, store the directories holding them in variables
LIBRAFT_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact raft 2836 cpp wheel
)
LIBCUVS_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libcuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact cuvs 1605 cpp wheel
)
# write a pip constraints file saying e.g. "whenever you encounter a requirement for 'librmm-cu12', use this wheel"
cat > "${PIP_CONSTRAINT}" <<EOF
libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRAFT_WHEELHOUSE}"/libraft_*.whl)
libcuvs-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUVS_WHEELHOUSE}"/libcuvs_*.whl)
EOF
