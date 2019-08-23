#!/usr/bin/env bash

# This assumes the script is executed from the root of the repo directory
./build.sh cuml
$PYTHON -c 'import cuml'
