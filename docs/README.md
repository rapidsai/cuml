# Building Documentation
## Building locally:

#### [Build and install cuML](../BUILD.md)

#### Use your current cuml dev environment, or build a new one doc building environment.
```shell script
(new) conda env create --name docs -f ./conda/environment/builddocs_py37.yml
(existing) pip install -r docs/requirements.txt
```
#### Generate the docs
```shell script
cd docs
make clean; make html
```

#### Once the process finishes, documentation can be found in build/html
```shell script
xdg-open build/html/api.html`
```

## Building via Docker
Pull or create the cuML [docker container](https://hub.docker.com/r/rapidsai/rapidsai/).

### Start cuML container:
```
docker run -p 8000:8000 -it cuml bash
```

### Setup container's conda env for building docs:
```
sudo sh setup.sh ${cuML-container-id}
```

### Build & host docs from container:
```
sudo sh build.sh ${cuML-container-id}
```

### Copy docs from container to host:
```
docker cp ${cuML-container-id}:/docs/build/html .
```
