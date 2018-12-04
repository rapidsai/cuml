#!/bin/bash
#
# Copyright (c) 2018, NVIDIA CORPORATION.

set -e

function upload() {
    echo "UPLOADFILE = ${UPLOADFILE}"
    test -e ${UPLOADFILE}
    source ./travisci/libcudf/upload-anaconda.sh
}

if [ "$BUILD_LIBCUDF" == "1" ]; then
    # Upload libcudf
    export UPLOADFILE=`conda build conda-recipes/libcuml -c defaults -c conda-forge --output`
    upload
fi
