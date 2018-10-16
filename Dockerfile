# From: https://github.com/rapidsai/pygdf/blob/master/Dockerfile
FROM pygdf

ADD ml-prims /cuML/ml-prims
ADD cuML /cuML/cuML
ADD python /cuML/python
ADD setup.py /cuML/setup.py

WORKDIR /cuML
RUN source activate gdf && conda install cython
RUN source activate gdf && python setup.py install
