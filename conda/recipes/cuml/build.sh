#!/usr/bin/env bash
cd python
$PYTHON setup.py build_ext --inplace
$PYTHON setup.py install
