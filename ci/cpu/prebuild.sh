#!/usr/bin/env bash

export BUILD_ABI=1
export BUILD_CUML=1

if [[ "$PYTHON" == "3.6" ]]; then
    export BUILD_LIBCUML=1
else
    export BUILD_LIBCUML=0
fi