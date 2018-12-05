# From: https://github.com/rapidsai/cudf/blob/master/Dockerfile
FROM cudf

ARG CUDA_MAJOR=9
ARG CUDA_MINOR=2
RUN source activate cudf && conda install -c pytorch faiss-gpu cuda${CUDA_MAJOR}${CUDA_MINOR}
RUN source activate cudf && conda install -c anaconda cython

ADD ml-prims /cuML/ml-prims
ADD cuML /cuML/cuML
ADD python /cuML/python
ADD python/setup.cfg /cuML/setup.cfg
ADD setup.py /cuML/setup.py
ADD versioneer.py /cuML/versioneer.py
WORKDIR /cuML
RUN source activate cudf && python setup.py install
