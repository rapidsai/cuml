#!/bin/bash

CUML_OS="${CUML_OS:-ubuntu18.04}"
CUML_CUDA="${CUML_CUDA:-11.0}"
CUML_PY="${CUML_PY:-3.8}"
CUML_VERSION="$(git describe --tags | grep -o -E '([0-9]+\.[0-9]+)')"
CUML_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-all}"

if [ -z "${CUML_BUILD_VOLUME}" ]
then
  # Use anonymous volume
  build_volume_opt="/cuml-dev/cpp/build"
else
  build_volume_opt="${CUML_BUILD_VOLUME}:/cuml-dev/cpp/build"
fi

if [ -z "${CUML_CCACHE_VOLUME}" ]
then
  # Use anonymous volume
  ccache_volume_opt="/ccache"
else
  ccache_volume_opt="${CUML_CCACHE_VOLUME}:/ccache"
fi

docker_image="rapidsai/rapidsai-core-dev-nightly:${CUML_VERSION}-cuda${CUML_CUDA}-devel-${CUML_OS}-py${CUML_PY}"

docker pull "${docker_image}"

docker run \
  --rm \
  -it \
  --gpus "device=${CUML_VISIBLE_DEVICES}" \
  --name cuml-dev \
  -v /cuml-dev/raft \
  -v /cuml-dev/python/raft \
  -v "${PWD}:/cuml-dev" \
  -v "${build_volume_opt}" \
  -v "${ccache_volume_opt}" \
  -w /cuml-dev \
  --entrypoint /bin/bash \
  "${docker_image}"
