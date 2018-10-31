#!/usr/bin/env bash
CMAKE_COMMON_VARIABLES=" -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release"


$PYTHON setup.py install --single-version-externally-managed --record=record.txt  # Python command to install the script.