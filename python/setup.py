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

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import os
import versioneer
from distutils.sysconfig import get_python_lib
import sys
import subprocess

install_requires = [
    'numba',
    'cython'
]


##############################################################################
# - Dependencies include and lib folder setup --------------------------------

cuda_include_dir = '/usr/local/cuda/include'
cuda_lib_dir = "/usr/local/cuda/lib"

if os.environ.get('CUDA_HOME', False):
    cuda_lib_dir = os.path.join(os.environ.get('CUDA_HOME'), 'lib64')
    cuda_include_dir = os.path.join(os.environ.get('CUDA_HOME'), 'include')


conda_lib_dir = os.path.normpath(sys.prefix) + '/lib'
conda_include_dir = os.path.normpath(sys.prefix) + '/include'

if os.environ.get('CONDA_PREFIX', None):
    conda_prefix = os.environ.get('CONDA_PREFIX')
    conda_include_dir = conda_prefix + '/include'
    conda_lib_dir = conda_prefix + '/lib'


##############################################################################
# - Subrepo checking and cloning ---------------------------------------------


def clone_repo(name, GIT_REPOSITORY, GIT_TAG):
    """
    Function to clone repos in case they were not cloned by cmake.
    Variables are named identical to the cmake counterparts for clarity,
    in spite of not being very pythonic.
    """
    subprocess.check_call(['rm', '-rf', 'external'])
    subprocess.check_call(['git', 'clone',
                           GIT_REPOSITORY,
                           'external/' + name])
    wd = os.getcwd()
    os.chdir("external/" + name)
    subprocess.check_call(['git', 'checkout',
                          GIT_TAG])
    os.chdir(wd)


# This should match their equivalent repos and tags of the CMakeLists that was
# used to build libcuml++
if "--clonedeps" in sys.argv:
    clone_repo(name='treelite',
               GIT_REPOSITORY='https://github.com/dmlc/treelite.git',
               GIT_TAG='600afd55d1fa9bb94fc88fd3a3043cb2d5b20651')
    treelite_path = 'external/treelite'

    clone_repo(name='cub',
               GIT_REPOSITORY='https://github.com/NVlabs/cub.git',
               GIT_TAG='v1.8.0')
    cub_path = 'external/cub'

    clone_repo(name='cutlass',
               GIT_REPOSITORY='https://github.com/NVIDIA/cutlass.git',
               GIT_TAG='v1.0.1')
    cutlass_path = 'external/cutlass'

    sys.argv.remove('--clonedeps')
else:

    if os.environ.get('CUML_BUILD_PATH', False):
        libcuml_path = '../' + os.environ.get('CUML_BUILD_PATH')
    else:
        libcuml_path = '../cpp/build/'

    if not os.path.exists(libcuml_path):
        raise RuntimeError("Third party repositories have not been found. \
                           Use the --clondepes option to have setup.py clone \
                           them automatically, or set the environment \
                           variable CUML_BUILD_PATH, containing the relative \
                           path of the root of the repository to the folder \
                           where libcuml++ was built.")
    else:
        treelite_path = libcuml_path + 'treelite/src/treelite'
        cub_path = libcuml_path + 'cub/src/cub'
        cutlass_path = 'cutlass/src/cutlass'


##############################################################################
# - Cython extensions build and parameters -----------------------------------

libs = ['cuda',
        'cuml++',
        'cumlcomms',
        'nccl',
        'rmm']

include_dirs = ['../cpp/src',
                '../cpp/external',
                '../cpp/src_prims',
                cutlass_path,
                cub_path,
                treelite_path + '/include',
                '../cpp/comms/std/src',
                '../cpp/comms/std/include',
                cuda_include_dir,
                conda_include_dir]

# Exclude multigpu components that use libcumlprims if --singlegpu is ued
exc_list = []
if "--singlegpu" in sys.argv:
    exc_list.append('cuml/linear_model/linear_regression_mg.pyx')
    exc_list.append('cuml/decomposition/tsvd_mg.pyx')
    exc_list.append('cuml/cluster/kmeans_mg.pyx')
    sys.argv.remove('--singlegpu')
else:
    libs.append('cumlprims')

extensions = [
    Extension("*",
              sources=["cuml/**/**/*.pyx"],
              include_dirs=include_dirs,
              library_dirs=[get_python_lib()],
              runtime_library_dirs=[cuda_lib_dir,
                                    conda_lib_dir],
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
