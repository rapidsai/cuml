#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION.
#####################
# cuML Style Tester #
#####################

# Ignore errors and set path
set +e
PATH=/conda/bin:$PATH

# Activate common conda env
source activate gdf

# Run flake8 and get results/return code
FLAKE=`flake8 --exclude=cuML,ml-prims,__init__.py,versioneer.py && flake8 --config=python/.flake8.cython`
RETVAL=$?

# Output results if failure otherwise show pass
if [ "$FLAKE" != "" ]; then
  echo -e "\n\n>>>> FAILED: flake8 style check; begin output\n\n"
  echo -e "$FLAKE"
  echo -e "\n\n>>>> FAILED: flake8 style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: flake8 style check\n\n"
fi

exit $RETVAL
