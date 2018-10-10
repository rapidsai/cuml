# Introduction
This repo contains python test code for our ML algorithms with scikit-learn like APIs. 

# Setup
## Dependencies
1. pytest (>=3.5.1)
2. h2o4gpu (>=0.2.0.9999)
3. scikit-learn (>=0.19.1)

# Running unit tests
```bash
$ pytest -s  
```
# Customized test (recommended for testing correctness)
```bash
$ python test_pca.py --help 
```
# Batch test (recommended for testing speedup)
```bash
$ python write_script.py
$ sh test_mortgage.sh
```

# Test results
data/

# Visualize test results
cuml\_test\_figures.ipynb
