# DecisionTree
This example code demonstrates use of C++ API of cuML decisiontree. It requires
`libcuml++.so` in order to build.

## Build

The example can be build either as part of cuML or can also be build as a
standalone. Two separate `CMakeLists.txt` files are provided for these two
cases.

1. `CMakeLists.txt` - To be used when example is build as part of cuML
2. `CMakeLists_standalone.txt` - To be used for building example standalone

### Standalone build
While building standalone use `CMakeLists_standalone.txt` and configure with:
```bash
$ cmake .. -DCUML_LIBRARY_DIR=/path/to/libcuml++.so -DCUML_INCLUDE_DIR=/path/to/cuml/headers
```
then build with `make`
On successful build, example should build `decisiontree_example` binary.

## Run
TBD
