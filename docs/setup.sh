#!/bin/bash

CONTAINER_ID=$1

cd ..
docker cp docs $CONTAINER_ID:/
docker exec -it $CONTAINER_ID bash -c "source activate gdf && \
                                       cd /docs &&
                                       pip install -r requirement.txt"
