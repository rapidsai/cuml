# From: https://github.com/rapidsai/cudf/blob/master/Dockerfile
FROM cudf

ENV CONDA_ENV=cudf

ADD thirdparty /cuml/thirdparty
ADD ml-prims /cuml/ml-prims
ADD cuML /cuml/cuML
WORKDIR /cuml/cuML
RUN source activate ${CONDA_ENV} && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX && \
    make -j && \
    make install

ADD python /cuml/python
WORKDIR /cuml/python
RUN source activate ${CONDA_ENV} && \
    python setup.py build_ext --inplace && \
    python setup.py install

ADD docs /cuml/docs
WORKDIR /cuml
