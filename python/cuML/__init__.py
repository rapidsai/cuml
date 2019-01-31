# Copyright (c) 2018, NVIDIA CORPORATION.
# Versioneer
from cuML import numba_utils

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
