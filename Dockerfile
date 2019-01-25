# From: https://github.com/rapidsai/cudf/blob/master/Dockerfile
FROM cudf

RUN apt install -y zlib1g-dev

ARG CUDA_MAJOR=9
ARG CUDA_MINOR=2
RUN source activate cudf && conda install -c pytorch faiss-gpu cuda${CUDA_MAJOR}${CUDA_MINOR}
RUN source activate cudf && conda install -c anaconda cython

ADD ml-prims /cuml/ml-prims
ADD cuML /cuml/cuML
WORKDIR /cuml/cuML
RUN source activate cudf && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install

ADD python /cuml/python
WORKDIR /cuml/python
RUN source activate cudf && \
    python setup.py build_ext --inplace && \
    python setup.py install

#ADD python/setup.cfg /cuml/setup.cfg
#ADD setup.py /cuML/setup.py
#ADD versioneer.py /cuML/versioneer.py

ADD docs /cuml/docs
WORKDIR /cuml/docs
CMD source activate cudf && make html && cd build/html && python -m http.server 8888
