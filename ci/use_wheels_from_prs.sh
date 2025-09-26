#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# initialize PIP_CONSTRAINT
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

# download wheels, store the directories holding them in variables
LIBCUVS_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libcuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact cuvs 1234 cpp wheel
)


# write a pip constraints file saying e.g. "whenever you encounter a requirement for 'librmm-cu12', use this wheel"
cat > "${PIP_CONSTRAINT}" <<EOF
libcuvs-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUVS_WHEELHOUSE}/libcuvs_*.whl)
EOF
