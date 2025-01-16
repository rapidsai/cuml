# syntax=docker/dockerfile:1.5

ARG BASE
ARG PYTHON_PACKAGE_MANAGER=conda

FROM ${BASE} as pip-base

ENV DEFAULT_VIRTUAL_ENV=rapids
ENV RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true

RUN apt update -y \
 && DEBIAN_FRONTEND=noninteractive apt install -y \
    libblas-dev liblapack-dev \
 && rm -rf /tmp/* /var/tmp/* /var/cache/apt/* /var/lib/apt/lists/*;

FROM ${BASE} as conda-base

ENV DEFAULT_CONDA_ENV=rapids

FROM ${PYTHON_PACKAGE_MANAGER}-base

ARG CUDA
ENV CUDAARCHS="RAPIDS"
ENV CUDA_VERSION="${CUDA_VERSION:-${CUDA}}"

ARG PYTHON_PACKAGE_MANAGER
ENV PYTHON_PACKAGE_MANAGER="${PYTHON_PACKAGE_MANAGER}"

ENV PYTHONSAFEPATH="1"
ENV PYTHONUNBUFFERED="1"
ENV PYTHONDONTWRITEBYTECODE="1"

ENV SCCACHE_REGION="us-east-2"
ENV SCCACHE_BUCKET="rapids-sccache-devs"
ENV AWS_ROLE_ARN="arn:aws:iam::279114543810:role/nv-gha-token-sccache-devs"
ENV HISTFILE="/home/coder/.cache/._bash_history"
