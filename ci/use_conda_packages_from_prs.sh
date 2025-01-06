# Copyright (c) 2025, NVIDIA CORPORATION.
# TODO(jameslamb): remove this file when https://github.com/rapidsai/raft/pull/2531 is merged

RAFT_COMMIT="345f0e556b602ec65b5eebe825ffd000d61706fe"

RAFT_CPP_CHANNEL=$(rapids-get-pr-conda-artifact raft 2531 cpp "${RAFT_COMMIT:0:7}")
RAFT_PYTHON_CHANNEL=$(rapids-get-pr-conda-artifact raft 2531 python "${RAFT_COMMIT:0:7}")

conda config --system --add channels "${RAFT_CPP_CHANNEL}"
conda config --system --add channels "${RAFT_PYTHON_CHANNEL}"
