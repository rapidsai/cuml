# From: https://github.com/rapidsai/pygdf/blob/master/Dockerfile
FROM pygdf

ADD ml-prims /cuML/ml-prims
ADD cuML /cuML/cuML
ADD python /cuML/python

WORKDIR /cuML/python
RUN source activate gdf && conda install cython
RUN source activate gdf && python setup.py install
