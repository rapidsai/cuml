# Copyright (c) 2025, NVIDIA CORPORATION.
# TODO(jameslamb): remove this file when https://github.com/rapidsai/cuvs/pull/594 is merged

CUVS_COMMIT="97c56178cd0e07e4b6b138bb0904af78379f1bb3"

CUVS_CPP_CHANNEL=$(rapids-get-pr-conda-artifact cuvs 594 cpp "${CUVS_COMMIT:0:7}")
CUVS_PYTHON_CHANNEL=$(rapids-get-pr-conda-artifact cuvs 594 python "${CUVS_COMMIT:0:7}")

conda config --system --add channels "${CUVS_CPP_CHANNEL}"
conda config --system --add channels "${CUVS_PYTHON_CHANNEL}"
