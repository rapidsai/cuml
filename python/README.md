# cuML Python Package

To use cuML, it must be cloned and built in an environment that already has the dependencies, including [cuDF](https://github.com/rapidsai/cudf) and its dependencies.

List of dependencies:

1. zlib
2. cmake (>= 3.8, version 3.11.4 is recommended and there are issues with version 3.12)
3. CUDA SDK (>= 9.2)
4. Cython (>= 0.28)
5. gcc (>=5.4.0)
6. nvcc
7. [cuDF](https://github.com/rapidsai/cudf)

### Setup steps

To clone:

```
git clone --recurse-submodules git@github.com:rapidsai/cuML.git
```

To build the python package, in the repository root folder:

```
cd python
python setup.py install
```

### Python Tests

Additional python tests can be found in the pythontests folder, along some useful scripts. Py.test based unit testing is still being worked on.
