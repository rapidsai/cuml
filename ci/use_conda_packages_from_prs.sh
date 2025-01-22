# Copyright (c) 2025, NVIDIA CORPORATION.
# TODO(jameslamb): remove this file when https://github.com/rapidsai/cuvs/pull/594 is merged

CUVS_COMMIT="86405194a1768b72fe4f8fcd7e7894e2a0b135c7"

CUVS_CPP_CHANNEL=$(rapids-get-pr-conda-artifact cuvs 594 cpp "${CUVS_COMMIT:0:7}")
CUVS_PYTHON_CHANNEL=$(rapids-get-pr-conda-artifact cuvs 594 python "${CUVS_COMMIT:0:7}")

conda config --system --add channels "${CUVS_CPP_CHANNEL}"
conda config --system --add channels "${CUVS_PYTHON_CHANNEL}"
