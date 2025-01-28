# GPU accelerated Scikit-Learn preprocessing

This directory contains code originating from the Scikit-Learn library. The Scikit-Learn license applies accordingly (see `/thirdparty/LICENSES/LICENSE.scikit_learn`). Original authors mentioned in the code do not endorse or promote this production.

This work is dedicated to providing GPU accelerated tools for preprocessing. The Scikit-Learn code is slightly modified to make it possible to take common inputs used throughout cuML such as Numpy and Cupy arrays, Pandas and cuDF dataframes and compute the results on GPU.

The code originates from the Scikit-Learn Github repository : https://github.com/scikit-learn/scikit-learn.git and is based on version/branch 0.23.1.

## For developers:
    When adding new preprocessors or updating, keep in mind:
    - Files should be copied as-is from the scikit-learn repo (preserving scikit-learn license text)
    - Changes should be kept minimal, large portions of modified imported code should lie in the thirdparty_adapter directory
    - Only well-tested, reliable accelerated preprocessing functions should be exposed in cuml.preprocessing.__init__.py
    - Tests must be added for each exposed function
    - Remember that a preprocessing model should always return the same datatype it received as input (NumPy, CuPy, Pandas, cuDF, Numba)
