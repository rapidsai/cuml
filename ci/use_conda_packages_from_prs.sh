# Copyright (c) 2025, NVIDIA CORPORATION.
# TODO(jameslamb): remove this file when https://github.com/rapidsai/cuvs/pull/594 is merged

CUVS_COMMIT="0bc1f0a77d46bda91eda6f816ea7b49b797676f9"

CUVS_CPP_CHANNEL=$(rapids-get-pr-conda-artifact cuvs 594 cpp "${RAFT_COMMIT:0:7}")
CUVS_PYTHON_CHANNEL=$(rapids-get-pr-conda-artifact cuvs 594 python "${RAFT_COMMIT:0:7}")

conda config --system --add channels "${CUVS_CPP_CHANNEL}"
conda config --system --add channels "${CUVS_PYTHON_CHANNEL}"
