# From: https://github.com/rapidsai/pygdf/blob/master/Dockerfile
FROM pygdf

ADD ml-prims /cuML/ml-prims
ADD cuML /cuML/cuML
ADD python /cuML/python
ADD docs /cuML/docs

WORKDIR /cuML/python
RUN source activate gdf && conda install cython
RUN source activate gdf && python setup.py install

RUN source activate gdf && cd /cuML/docs && pip install -r requirement.txt
RUN source activate gdf && cd /cuML/docs && make html && cd build/html
CMD source activate gdf && cd /cuML/docs/build/html && python -m http.server
