#
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

from distutils.sysconfig import get_python_lib
from pathlib import Path
from setuptools import find_packages
from setuptools import setup
from setuptools.extension import Extension
from setuputils import clean_folder
from setuputils import clone_repo_if_needed
from setuputils import get_environment_option
from setuputils import get_cli_option
from setuputils import use_raft_package

import numpy
import os
import shutil
import sys
import sysconfig
import versioneer
import warnings


if "--singlegpu" in sys.argv:
    from Cython.Build import cythonize
    from setuptools.command.build_ext import build_ext
else:
    try:
        from Cython.Distutils.build_ext import new_build_ext as build_ext
    except ImportError:
        from setuptools.command.build_ext import build_ext

install_requires = [
    'numba',
    'cython'
]

##############################################################################
# - Print of build options used by setup.py  --------------------------------

cuda_home = get_environment_option("CUDA_HOME")
libcuml_path = get_environment_option('CUML_BUILD_PATH')
raft_path = get_environment_option('RAFT_PATH')

clean_artifacts = get_cli_option('clean')
single_gpu_build = get_cli_option('--singlegpu')

##############################################################################
# - Dependencies include and lib folder setup --------------------------------

if not cuda_home:
    cuda_home = str(Path(shutil.which('nvcc')).parent.parent)
    print("-- Using nvcc to detect CUDA, found at " + str(cuda_home))
cuda_include_dir = os.path.join(cuda_home, "include")
cuda_lib_dir = os.path.join(cuda_home, "lib64")

##############################################################################
# - Clean target -------------------------------------------------------------

if clean_artifacts:
    print("-- Cleaning all Python and Cython build artifacts...")

    treelite_path = ""
    libcuml_path = ""

    try:
        setup_file_path = str(Path(__file__).parent.absolute())
        shutil.rmtree(setup_file_path + '/.pytest_cache', ignore_errors=True)
        shutil.rmtree(setup_file_path + '/_external_repositories',
                      ignore_errors=True)
        shutil.rmtree(setup_file_path + '/cuml.egg-info', ignore_errors=True)
        shutil.rmtree(setup_file_path + '/__pycache__', ignore_errors=True)

        os.remove(setup_file_path + '/cuml/raft')

        clean_folder(setup_file_path + '/cuml')
        shutil.rmtree(setup_file_path + '/build')

    except IOError:
        pass

    # need to terminate script so cythonizing doesn't get triggered after
    # cleanup unintendedly
    sys.argv.remove("clean")

    if "--all" in sys.argv:
        sys.argv.remove("--all")

    if len(sys.argv) == 1:
        sys.exit(0)

##############################################################################
# - Cloning RAFT and dependencies if needed ----------------------------------

# Use RAFT repository in cuml.raft

use_raft_package(raft_path, libcuml_path)

# Use treelite from the libcuml build folder, otherwise clone it
# Needed until there is a treelite distribution

treelite_path, _ = clone_repo_if_needed('treelite', libcuml_path)
treelite_path = os.path.join(treelite_path, "include")


##############################################################################
# - Cython extensions build and parameters -----------------------------------

# cumlcomms and nccl are still needed for multigpu algos not based
# on libcumlprims
libs = ['cuda',
        'cuml++',
        'rmm']

include_dirs = ['../cpp/src',
                '../cpp/include',
                '../cpp/src_prims',
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
    exc_list.append('cuml/cluster/kmeans_mg.pyx')
    exc_list.append('cuml/decomposition/base_mg.pyx')
    exc_list.append('cuml/decomposition/pca_mg.pyx')
    exc_list.append('cuml/decomposition/tsvd_mg.pyx')
    exc_list.append('cuml/linear_model/base_mg.pyx')
    exc_list.append('cuml/linear_model/ridge_mg.pyx')
    exc_list.append('cuml/linear_model/linear_regression_mg.pyx')
    exc_list.append('cuml/neighbors/nearest_neighbors_mg.pyx')

else:
    libs.append('cumlprims')
    libs.append('cumlcomms')
    libs.append('nccl')

    sys_include = os.path.dirname(sysconfig.get_path("include"))
    include_dirs.append("%s/cumlprims" % sys_include)

cmdclass = dict()
cmdclass.update(versioneer.get_cmdclass())
cmdclass["build_ext"] = build_ext

if not libcuml_path:
    libcuml_path = '../cpp/build/'

extensions = [
    Extension("*",
              sources=["cuml/**/**/*.pyx"],
              include_dirs=include_dirs,
              library_dirs=[get_python_lib(), libcuml_path],
              runtime_library_dirs=[cuda_lib_dir,
                                    os.path.join(os.sys.prefix, "lib")],
              libraries=libs,
              language='c++',
              extra_compile_args=['-std=c++11'])
]

for e in extensions:
    # TODO: this exclude is not working, need to research way to properly
    # exclude files for parallel build. See issue
    # https://github.com/rapidsai/cuml/issues/2037
    # e.exclude = exc_list
    e.cython_directives = dict(
        profile=False, language_level=3, embedsignature=True
    )

if "--singlegpu" in sys.argv:
    print("Full cythonization in parallel is not supported for singlegpu " +
          "target for now.")
    extensions = cythonize(extensions,
                           exclude=exc_list)
    sys.argv.remove('--singlegpu')

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
      ext_modules=extensions,
      packages=find_packages(include=['cuml', 'cuml.*']),
      install_requires=install_requires,
      license="Apache",
      cmdclass=cmdclass,
      zip_safe=False
      )
