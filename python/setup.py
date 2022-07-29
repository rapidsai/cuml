#
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

import os
import shutil
import sys
from pathlib import Path

from setuptools import find_packages

from setuputils import clean_folder
from setuputils import get_environment_option
from setuputils import get_cli_option

import versioneer
from skbuild import setup
from skbuild.command.build_ext import build_ext

install_requires = ['numba', 'cython']

##############################################################################
# - Print of build options used by setup.py  --------------------------------

cuda_home = get_environment_option("CUDA_HOME")
libcuml_path = get_environment_option('CUML_BUILD_PATH')

clean_artifacts = get_cli_option('clean')

##############################################################################
# - Dependencies include and lib folder setup --------------------------------

if not cuda_home:
    nvcc_path = shutil.which('nvcc')
    if (not nvcc_path):
        raise FileNotFoundError("nvcc not found.")

    cuda_home = str(Path(nvcc_path).parent.parent)
    print("-- Using nvcc to detect CUDA, found at " + str(cuda_home))

cuda_include_dir = os.path.join(cuda_home, "include")
cuda_lib_dir = os.path.join(cuda_home, "lib64")

##############################################################################
# - Clean target -------------------------------------------------------------

if clean_artifacts:
    print("-- Cleaning all Python and Cython build artifacts...")

    # Reset these paths since they may be deleted below
    treelite_path = False
    libcuml_path = False

    try:
        setup_file_path = str(Path(__file__).parent.absolute())
        shutil.rmtree(setup_file_path + '/.pytest_cache', ignore_errors=True)
        shutil.rmtree(setup_file_path + '/_external_repositories',
                      ignore_errors=True)
        shutil.rmtree(setup_file_path + '/cuml.egg-info', ignore_errors=True)
        shutil.rmtree(setup_file_path + '/__pycache__', ignore_errors=True)

        clean_folder(setup_file_path + '/cuml')
        shutil.rmtree(setup_file_path + '/build', ignore_errors=True)
        shutil.rmtree(setup_file_path + '/_skbuild', ignore_errors=True)
        shutil.rmtree(setup_file_path + '/dist', ignore_errors=True)

    except IOError:
        pass

    # need to terminate script so cythonizing doesn't get triggered after
    # cleanup unintendedly
    sys.argv.remove("clean")

    if "--all" in sys.argv:
        sys.argv.remove("--all")

    if len(sys.argv) == 1:
        sys.exit(0)


if not libcuml_path:
    libcuml_path = '../cpp/build/'

cmdclass = versioneer.get_cmdclass()

##############################################################################
# - Python package generation ------------------------------------------------

setup(name='cuml',
      description="cuML - RAPIDS ML Algorithms",
      version=versioneer.get_version(),
      classifiers=[
          "Intended Audience :: Developers",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9"
      ],
      author="NVIDIA Corporation",
      url="https://github.com/rapidsai/cudf",
      setup_requires=['Cython>=0.29,<0.30'],
      packages=find_packages(include=['cuml', 'cuml.*']),
      package_data={
          key: ["*.pxd"] for key in find_packages(include=['cuml', 'cuml.*'])
      },
      install_requires=install_requires,
      license="Apache",
      cmdclass=cmdclass,
      zip_safe=False)
