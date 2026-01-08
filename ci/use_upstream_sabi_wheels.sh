#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# initialize PIP_CONSTRAINT
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then

# download wheels, store the directories holding them in variables
LIBRMM_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact rmm 2184 cpp wheel
)
RMM_WHEELHOUSE=$(
  rapids-get-pr-artifact rmm 2184 python wheel --stable
)
LIBRAFT_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact raft 2915 cpp wheel
)
PYLIBRAFT_WHEELHOUSE=$(
  rapids-get-pr-artifact raft 2915 python wheel --stable --pkg_name pylibraft
)
CUDF_WHEELHOUSE=$(
  rapids-get-pr-artifact cudf 20974 python wheel --stable
)

cat > "${PIP_CONSTRAINT}" <<EOF
librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRMM_WHEELHOUSE}"/librmm_*.whl)
rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${RMM_WHEELHOUSE}"/rmm_*.whl)
libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRAFT_WHEELHOUSE}"/libraft_*.whl)
pylibraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${PYLIBRAFT_WHEELHOUSE}"/pylibraft_*.whl)
cudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${CUDF_WHEELHOUSE}"/cudf_*.whl)
EOF

fi
