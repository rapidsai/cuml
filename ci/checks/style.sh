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
FLAKE=`flake8 --exclude=cpp,thirdparty,__init__.py,versioneer.py && flake8 --config=python/.flake8.cython`
RETVAL=$?

# Output results if failure otherwise show pass
if [ "$FLAKE" != "" ]; then
  echo -e "\n\n>>>> FAILED: flake8 style check; begin output\n\n"
  echo -e "$FLAKE"
  echo -e "\n\n>>>> FAILED: flake8 style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: flake8 style check\n\n"
fi

# Check for copyright headers in the files modified currently
COPYRIGHT=`env PYTHONPATH=cpp/scripts python ci/checks/copyright.py 2>&1`
CR_RETVAL=$?
if [ "$RETVAL" = "0" ]; then
  RETVAL=$CR_RETVAL
fi

# Output results if failure otherwise show pass
if [ "$CR_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: copyright check; begin output\n\n"
  echo -e "$COPYRIGHT"
  echo -e "\n\n>>>> FAILED: copyright check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: copyright check\n\n"
fi

# Check for a consistent #include syntax
# TODO: keep adding more dirs as and when we update the syntax
HASH_INCLUDE=`python cpp/scripts/include_checker.py \
                     cpp/bench \
                     cpp/comms/mpi/include \
                     cpp/comms/mpi/src \
                     cpp/comms/std/include \
                     cpp/comms/std/src \
                     cpp/include \
                     cpp/examples \
                     2>&1`
HASH_RETVAL=$?
if [ "$RETVAL" = "0" ]; then
  RETVAL=$HASH_RETVAL
fi

# Output results if failure otherwise show pass
if [ "$HASH_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: #include check; begin output\n\n"
  echo -e "$HASH_INCLUDE"
  echo -e "\n\n>>>> FAILED: #include check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: #include check\n\n"
fi

exit $RETVAL
