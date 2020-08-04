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

import glob
import os
import shutil
import sys
import sysconfig
import warnings
from pprint import pprint
from pathlib import Path

from setuptools import find_packages
from setuptools import setup
from setuptools.extension import Extension
from distutils.sysconfig import get_python_lib
from distutils.command.clean import clean as _clean
from distutils.command.build import build as _build

import numpy

from setuputils import clean_folder
from setuputils import get_environment_option
from setuputils import get_cli_option
from setuputils import use_raft_package

import versioneer
from cython_build_ext import cython_build_ext

install_requires = ['numba', 'cython']

##############################################################################
# - Print of build options used by setup.py  --------------------------------

global libcuml_path

cuda_home = get_environment_option("CUDA_HOME")
libcuml_path = get_environment_option('CUML_BUILD_PATH')
raft_path = get_environment_option('RAFT_PATH')

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
# - Cloning RAFT and dependencies if needed ----------------------------------

# Use RAFT repository in cuml.raft

raft_include_dir = use_raft_package(raft_path, libcuml_path)

if "--multigpu" in sys.argv:
    warnings.warn("Flag --multigpu is deprecated. By default cuML is"
                  "built with multi GPU support. To disable it use the flag"
                  "--singlegpu")
    sys.argv.remove('--multigpu')

if not libcuml_path:
    libcuml_path = '../cpp/build/'

##############################################################################
# - Clean target -------------------------------------------------------------
# This derives from distutils clean to so we can use the derived values of
# 'build' and the base clean implementation


class cuml_clean(_clean):
    def run(self):

        global libcuml_path

        # Call the base first to get info from build
        super().run()

        if (self.all):
            # Reset libcuml_path
            libcuml_path = ""

            try:
                setup_file_path = str(Path(__file__).parent.absolute())
                shutil.rmtree(os.path.join(setup_file_path, ".pytest_cache"),
                              ignore_errors=True)
                shutil.rmtree(os.path.join(setup_file_path,
                                           '/_external_repositories'),
                              ignore_errors=True)
                shutil.rmtree(os.path.join(setup_file_path, '/cuml.egg-info'),
                              ignore_errors=True)
                shutil.rmtree(os.path.join(setup_file_path, '/__pycache__'),
                              ignore_errors=True)

                os.remove(setup_file_path + '/cuml/raft')

                clean_folder(setup_file_path + '/cuml')

            except IOError:
                pass


##############################################################################
# - Cython extensions build and parameters -----------------------------------
# Derive from `cython_build_ext` to add --singlegpu customization


class cuml_build(_build):

    user_options = [
        ("singlegpu", None, "Specifies whether to include multi-gpu or not")
    ] + _build.user_options

    boolean_options = ["singlegpu"] + _build.boolean_options

    def initialize_options(self):

        self.singlegpu = False

        super().initialize_options()

    def finalize_options(self):

        global libcuml_path

        # cumlcomms and nccl are still needed for multigpu algos not based
        # on libcumlprims
        libs = ['cuda', 'cuml++', 'rmm']

        include_dirs = [
            '../cpp/src', '../cpp/include', '../cpp/src_prims',
            raft_include_dir, '../cpp/comms/std/src',
            '../cpp/comms/std/include', cuda_include_dir,
            numpy.get_include(),
            os.path.dirname(sysconfig.get_path("include"))
        ]

        # Exclude multigpu components that use libcumlprims if
        # --singlegpu is used
        python_exc_list = []

        if (self.singlegpu):
            python_exc_list = ["*.dask", "*.dask.*"]
        else:
            libs.append('cumlprims')
            libs.append('cumlcomms')
            libs.append('nccl')

            sys_include = os.path.dirname(sysconfig.get_path("include"))
            include_dirs.append("%s/cumlprims" % sys_include)

        # Find packages now that --singlegpu has been determined
        self.distribution.packages = find_packages(include=['cuml', 'cuml.*'],
                                                   exclude=python_exc_list)

        # Build the extensions list
        extensions = [
            Extension("*",
                      sources=["cuml/**/*.pyx"],
                      include_dirs=include_dirs,
                      library_dirs=[get_python_lib(), libcuml_path],
                      runtime_library_dirs=[
                          cuda_lib_dir,
                          os.path.join(os.sys.prefix, "lib")
                      ],
                      libraries=libs,
                      language='c++',
                      extra_compile_args=['-std=c++11'])
        ]

        self.distribution.ext_modules = extensions

        super().finalize_options()


# This custom build_ext is only responsible for setting cython_exclude when
# --singlegpu is specified
class cuml_build_ext(cython_build_ext, object):
    user_options = [
        ("singlegpu", None, "Specifies whether to include multi-gpu or not"),
    ] + cython_build_ext.user_options

    boolean_options = ["singlegpu"] + cython_build_ext.boolean_options

    def initialize_options(self):

        self.singlegpu = None

        super().initialize_options()

    def finalize_options(self):

        # Ensure the base build class options get set so we can use singlegpu
        self.set_undefined_options(
            'build',
            ('singlegpu', 'singlegpu'),
        )

        # Exclude multigpu components that use libcumlprims if
        # --singlegpu is used
        if (self.singlegpu):
            cython_exc_list = glob.glob('cuml/*/*_mg.pyx')
            cython_exc_list = cython_exc_list + glob.glob('cuml/*/*_mg.pxd')
            cython_exc_list.append('cuml/nccl/nccl.pyx')
            cython_exc_list.append('cuml/dask/common/comms_utils.pyx')

            print('--singlegpu: excluding the following Cython components:')
            pprint(cython_exc_list)

            # Append to base excludes
            self.cython_exclude = cython_exc_list + \
                (self.cython_exclude or [])

        super().finalize_options()


# Specify the custom build class
cmdclass = dict()
cmdclass.update(versioneer.get_cmdclass())
cmdclass["clean"] = cuml_clean
cmdclass["build"] = cuml_build
cmdclass["build_ext"] = cuml_build_ext

##############################################################################
# - Python package generation ------------------------------------------------

setup(name='cuml',
      description="cuML - RAPIDS ML Algorithms",
      version=versioneer.get_version(),
      classifiers=[
          "Intended Audience :: Developers", "Programming Language :: Python",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7"
      ],
      author="NVIDIA Corporation",
      setup_requires=['cython'],
      install_requires=install_requires,
      license="Apache",
      cmdclass=cmdclass,
      zip_safe=False)
