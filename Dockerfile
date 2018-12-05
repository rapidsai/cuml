# From: https://github.com/rapidsai/cudf/blob/master/Dockerfile
FROM cudf

RUN apt install -y zlib1g-dev

ARG CUDA_MAJOR=9
ARG CUDA_MINOR=2
RUN source activate cudf && conda install -c pytorch faiss-gpu cuda${CUDA_MAJOR}${CUDA_MINOR}
RUN source activate cudf && conda install -c anaconda cython

ADD ml-prims /cuML/ml-prims
ADD cuML /cuML/cuML
WORKDIR /cuML/cuML
RUN source activate cudf && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make install

ADD docs /cuML/docs

ADD python /cuML/python
WORKDIR /cuML/python
RUN source activate cudf && python setup.py install
