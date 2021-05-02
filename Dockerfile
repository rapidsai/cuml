# From: https://github.com/rapidsai/cudf/blob/main/Dockerfile
FROM cudf

ENV CONDA_ENV=cudf

ADD . /cuml/

WORKDIR /cuml

RUN conda env update --name ${CONDA_ENV} \
    --file /cuml/conda/environments/cuml_dev_cuda${CUDA_SHORT_VERSION}.yml

# libcuml build/install
RUN source activate ${CONDA_ENV} && \
    cd cpp && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} && \
    make -j && \
    make install

# cuML build/install
RUN source activate ${CONDA_ENV} && \
    cd python && \
    python setup.py build_ext --inplace && \
    python setup.py install
