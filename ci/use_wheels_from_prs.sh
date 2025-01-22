# Copyright (c) 2025, NVIDIA CORPORATION.
# TODO(jameslamb): remove this file when https://github.com/rapidsai/cuvs/pull/594 is merged

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

CUVS_COMMIT="97c56178cd0e07e4b6b138bb0904af78379f1bb3"
CUVS_CHANNEL=$(
  RAPIDS_PY_WHEEL_NAME="cuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact cuvs 594 python "${CUVS_COMMIT:0:7}"
)
LIBCUVS_CHANNEL=$(
  RAPIDS_PY_WHEEL_NAME="libcuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact cuvs 594 cpp "${CUVS_COMMIT:0:7}"
)

cat >> ./constraints.txt <<EOF
cuvs-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${CUVS_CHANNEL}/cuvs_*.whl)
libcuvs-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUVS_CHANNEL}/libcuvs_*.whl)
EOF

export PIP_CONSTRAINT=$(pwd)/constraints.txt
