# Copyright (c) 2025, NVIDIA CORPORATION.

# TODO(jameslamb): remove this when https://github.com/rapidsai/raft/pull/2531 is merged
RAFT_COMMIT="345f0e556b602ec65b5eebe825ffd000d61706fe"
LIBRAFT_CHANNEL=$(
  RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact raft 2531 cpp "${RAFT_COMMIT:0:7}"
)
PYLIBRAFT_CHANNEL=$(
  RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact raft 2531 python "${RAFT_COMMIT:0:7}"
)
RAFT_DASK_CHANNEL=$(
  RAPIDS_PY_WHEEL_NAME="raft_dask_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact raft 2531 python "${RAFT_COMMIT:0:7}"
)
cat > ./constraints.txt <<EOF
libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRAFT_CHANNEL}/libraft_*.whl)
pylibraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBRAFT_CHANNEL}/pylibraft_*.whl)
raft-dask-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${RAFT_DASK_CHANNEL}/raft_dask_*.whl)
EOF

export PIP_CONSTRAINT=$(pwd)/constraints.txt
