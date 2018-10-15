# Building Documentation

A basic python environment with packages listed in `./requirement.txt` is
enough to build the docs.

## For building locally:
```
# From project root:
conda create env --name docs -f conda_environments/builddocs_py35.yml
source activate docs
cd python
python setup.py install
cd ../docs
make html
```

## For building via Docker

## Start cuML container:
```
docker run -p 8000:8000 -it cuml bash
```

## Setup container's conda env for building docs:
```
sudo sh setup.sh ${cuML-container-id}
```

## Build & host docs from container:
```
sudo sh build.sh ${cuML-container-id}
```

## Copy docs from container to host:
```
docker cp ${cuML-container-id}:/docs/build/html .
```
