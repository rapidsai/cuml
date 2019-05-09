#!/usr/bin/env bash
cd python
$PYTHON setup.py build_ext --inplace --multigpu
$PYTHON setup.py install --multigpu
