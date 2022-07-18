# kmeans
This subfolder contains an example on how to use cuML kmeans from C++ application
There are two `CMakeLists.txt` in this folder:
1. `CMakeLists.txt` (default) which is included when building cuML
2. `CMakeLists_standalone.txt` as an example for a stand alone project linking to `libcuml.so`

## Build
`kmeans_example` is build as part of cuML. If it should be build as a standalone executable use `CMakeLists_standalone.txt` and configure with:
```bash
$ cmake .. -Dcuml_ROOT=/path/to/cuml
```
then build with `make`
```bash
$ make
Scanning dependencies of target kmeans_example
[ 50%] Building CXX object CMakeFiles/kmeans_example.dir/kmeans_example.cpp.o
[100%] Linking CUDA executable kmeans_example
[100%] Built target kmeans_example
```

## Run

1. With tiny test input:
```
./kmeans_example
Run KMeans with k=2, max_iterations=300
```

2. With larger test data from Kaggle
  1. Prepare input: Download Homesite Quote Conversion data from https://www.kaggle.com/c/homesite-quote-conversion/data and use `./prepare_input.py [train_file=train.csv] [test_file=test.csv] [output=output.txt]`:
```
$ unzip all.zip
$ ./prepare_input.py [train_file=train.csv] [test_file=test.csv] [output=output.txt]
Reading Input from train_file = train.csv and test_file = test.csv
Training dataset dimension:  (260753, 299)
Test dataset dimension:      (173836, 298)
Output dataset dimension:    (260753, 298)
Wrote 77704394 values in row major order to output output.txt
```
  2. Run
```
$ ./kmeans_example -num_rows 260753 -num_cols 298 -input output.txt
Reading input with 260753 rows and 298 columns from output.txt.
Run KMeans with k=10, max_iterations=300
   num_pts       inertia
0    18615  7.749915e+12
1    18419  7.592070e+12
2    30842  1.815066e+13
3    31247  1.832832e+13
4    31272  1.887647e+13
5    18362  7.749335e+12
6    31028  1.821217e+13
7    31040  1.869879e+13
8    18652  7.681686e+12
9    31276  1.877210e+13
Global inertia = 1.418115e+14
```

To run with other inputs the executable `kmeans_example` has the following commandline options
* `-dev_id`: The id of the CUDA GPU to use (default 0)
* `-num_rows`: Number of rows in the input file (default 0)
* `-num_cols`: Number of columns in the input file (default 0)
* `-input`: Input file name with input values as text in row major order (default empty string)
* `-k`: Number of clusters (default 10)
* `-max_iterations`: Maximum number of iterations to execute (default 300)
