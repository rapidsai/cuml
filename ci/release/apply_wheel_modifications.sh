#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Usage: bash apply_wheel_modifications.sh <new_version>

VERSION=${1}

sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/cuml/__init__.py

sed -i "s/version=.*,/version=\"${VERSION}\",/g" python/setup.py
