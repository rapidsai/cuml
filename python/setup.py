#
# Copyright (c) 2018, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from Cython.Build import cythonize
from distutils.sysconfig import get_python_lib
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuputils import get_submodule_dependencies

import os
import subprocess
import sys
import sysconfig
import versioneer
import warnings
import numpy

install_requires = [
    'numba',
    'cython'
]


##############################################################################
# - Dependencies include and lib folder setup --------------------------------

CUDA_HOME = os.environ.get("CUDA_HOME", False)
if not CUDA_HOME:
    CUDA_HOME = (
        os.popen('echo "$(dirname $(dirname $(which nvcc)))"').read().strip()
    )
cuda_include_dir = os.path.join(CUDA_HOME, "include")
cuda_lib_dir = os.path.join(CUDA_HOME, "lib64")


##############################################################################
# - Subrepo checking and cloning ---------------------------------------------

subrepos = [
    'cub',
    'cutlass',
    'faiss',
    'treelite'
]

# We check if there is a libcuml++ build folder, by default in cpp/build
# or in CUML_BUILD_PATH env variable. Otherwise setup.py will clone the
# dependencies defined in cpp/CMakeListst.txt
if "clean" not in sys.argv:
    if os.environ.get('CUML_BUILD_PATH', False):
        libcuml_path = '../' + os.environ.get('CUML_BUILD_PATH')
    else:
        libcuml_path = '../cpp/build/'

    found_cmake_repos = get_submodule_dependencies(subrepos,
                                                   libcuml_path=libcuml_path)

    if found_cmake_repos:
        treelite_path = os.path.join(libcuml_path,
                                     'treelite/src/treelite/include')
        faiss_path = os.path.join(libcuml_path, 'faiss/src/')
        cub_path = os.path.join(libcuml_path, 'cub/src/cub')
        cutlass_path = os.path.join(libcuml_path, 'cutlass/src/cutlass')
    else:
        # faiss requires the include to be to the parent of the root of
        # their repo instead of the full path like the others
        faiss_path = 'external_repositories/'
        treelite_path = 'external_repositories/treelite/include'
        cub_path = 'external_repositories/cub'
        cutlass_path = 'external_repositories/cutlass'

else:
    subprocess.check_call(['rm', '-rf', 'external_repositories'])
    treelite_path = ""
    faiss_path = ""
    cub_path = ""
    cutlass_path = ""

##############################################################################
# - Cython extensions build and parameters -----------------------------------

libs = ['cuda',
        'cuml++',
        'cumlcomms',
        'nccl',
        'rmm']

include_dirs = ['../cpp/src',
                '../cpp/include',
                '../cpp/external',
                '../cpp/src_prims',
                cutlass_path,
                cub_path,
                faiss_path,
                treelite_path,
                '../cpp/comms/std/src',
                '../cpp/comms/std/include',
                cuda_include_dir,
                numpy.get_include(),
                os.path.dirname(sysconfig.get_path("include"))]

# Exclude multigpu components that use libcumlprims if --singlegpu is used
exc_list = []
if "--multigpu" in sys.argv:
    warnings.warn("Flag --multigpu is deprecated. By default cuML is"
                  "built with multi GPU support. To disable it use the flag"
                  "--singlegpu")
    sys.argv.remove('--multigpu')

if "--singlegpu" in sys.argv:
    exc_list.append('cuml/linear_model/ridge_mg.pyx')
    exc_list.append('cuml/linear_model/linear_regression_mg.pyx')
    exc_list.append('cuml/decomposition/tsvd_mg.pyx')
    exc_list.append('cuml/neighbors/nearest_neighbors_mg.pyx')
    exc_list.append('cuml/cluster/kmeans_mg.pyx')
    exc_list.append('cuml/decomposition/pca_mg.pyx')
    sys.argv.remove('--singlegpu')
else:
    libs.append('cumlprims')
    # ucx/ucx-py related functionality available in version 0.12+
    # libs.append("ucp")

    sys_include = os.path.dirname(sysconfig.get_path("include"))
    include_dirs.append("%s/cumlprims" % sys_include)

extensions = [
    Extension("*",
              sources=["cuml/**/**/*.pyx"],
              include_dirs=include_dirs,
              library_dirs=[get_python_lib()],
              runtime_library_dirs=[cuda_lib_dir,
                                    os.path.join(os.sys.prefix, "lib")],
              libraries=libs,
              language='c++',
              extra_compile_args=['-std=c++11'])
]


##############################################################################
# - Python package generation ------------------------------------------------

setup(name='cuml',
      description="cuML - RAPIDS ML Algorithms",
      version=versioneer.get_version(),
      classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
      ],
      author="NVIDIA Corporation",
      setup_requires=['cython'],
      ext_modules=cythonize(extensions,
                            exclude=exc_list),
      packages=find_packages(include=['cuml', 'cuml.*']),
      install_requires=install_requires,
      license="Apache",
      cmdclass=versioneer.get_cmdclass(),
      zip_safe=False
      )
