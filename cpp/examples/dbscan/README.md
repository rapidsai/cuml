# DBSCAN
This example code demonstrates use of C++ API of cuML DBSCAN. It requires `libcuml.so` in order to build.

## Build

The example can be build either as part of cuML or can also be build as a standalone. Two separate `CMakeLists.txt` files are provided for these two cases.

1. `CMakeLists.txt` - To be used when example is build as part of cuML
2. `CMakeLists_standalone.txt` - To be used for building example standalone

### Standalone build
While building standalone use `CMakeLists_standalone.txt` and configure with:
```bash
$ cmake .. -Dcuml_ROOT=/path/to/cuml
```
then build with `make`
```bash
$ make
[ 50%] Building CXX object CMakeFiles/dbscan_example.dir/dbscan_example.cpp.o

[100%] Linking CUDA executable dbscan_example
[100%] Built target dbscan_example
```
On successful build, example should build `dbscan_example` binary.

## Run

1. Run with trivial dataset:

When `dbscan_example` is invoked without any options, it loads a default trivial dataset and runs DBSCAN algorithm on that. The output should appear as shown below,
```
Samples file not specified. (-input option)
Running with default dataset:
Running DBSCAN with following parameters:
Number of samples - 25
Number of features - 3
min_pts - 2
eps - 1
Histogram of samples
Cluster id, Number samples
         0, 13
         1, 12
Total number of clusters: 2
Noise samples: 0

```

2. Run with non-trivial dataset:
To use `dbscan_example` on non-trivial datasets, first input file needs to be prepared. If the dataset has N samples with M features each, the input file needs to be and ASCII file with N\*M rows with features linearized as below,
```
sample-0-feature-0
sample-0-feature-1
       ...
sample-0-feature-(M-1)
sample-1-feature-0
sample-1-feature-1
       ...
sample-1-feature-(M-1)
       ...
       ...
sample-(N-1)-feature-0
sample-(N-1)-feature-1
       ...
sample-(N-1)-feature-(M-1)
```
All the features must be single precision floating point numbers. The example demonstrates single precision DBSCAN, but the cuML DBSCAN works equally well with double precision floating point numbers.

Once input file is ready, the `dbscan_example` can be invoked as below,

```
$ ./dbscan_example -input <input file> -num_samples <#samples> -num_features <#features> [-min_pts <minPts>] [-eps <eps>]
```
The output would look similar to,

```
Trying to read samples from synthetic-10000x25-clusters-15.txt
Running DBSCAN with following parameters:
Number of samples - 10000
Number of features - 25
min_pts - 5
eps - 0.6
Histogram of samples
Cluster id, Number samples
         0, 665
         1, 664
         2, 663
         3, 666
         4, 665
         5, 666
         6, 662
         7, 664
         8, 666
         9, 666
        10, 663
        11, 662
        12, 665
        13, 667
        14, 666
Total number of clusters: 15
Noise samples: 30

```
The output of the example is a histogram of sample count in each cluster. Number of noise samples are also reported.

### Details of command line options

* `-dev_id`: The id of the CUDA GPU to use (default 0)
* `-num_samples`: Number of samples
* `-num_features`: Number of features
* `-input`: Plain text input file with samples in row major order
* `-min_pts`: Minimum number of samples in a cluster (default 3)
* `-eps`: Maximum distance between any two samples of a cluster (default 1.0)

If `-input` is specified, `-num_samples` and `-num_features` must be specified.

## Synthetic dataset generator

For convenience, a synthetic dataset generator `gen_dataset.py` is included with the example. It can be used as shown below,

```
./gen_dataset.py --num_samples 1000 --num_features 16 --num_clusters 10 --filename_prefix synthetic
Dataset file: synthetic-1000x16-clusters-10.txt
Generated total 1000 samples with 16 features each
Number of clusters = 10
```
Command line options

* `--num_samples` or `-ns`: Number of samples
* `--num_features` or `-nf`: Number of features
* `--num_clusters` or `-nc`: Number of clusters
* `--filename_prefix`: Prefix used for dataset output file. Number of samples, features and clusters are appended as shown in above example.
* `--standard_dev` or `-sd`: Standard deviation of samples generated (default 0.1)
* `--random_state` of `-rs`: Random state used for seeding the pseudo-random number generator (default 123456)
