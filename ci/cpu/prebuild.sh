#!/usr/bin/env bash
# Copyright (c) 2022, NVIDIA CORPORATION.

export UPLOAD_CUML=1
export UPLOAD_LIBCUML=1

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    #If project flash is not activate, always build both
    export BUILD_LIBCUML=1
    export BUILD_CUML=1
fi