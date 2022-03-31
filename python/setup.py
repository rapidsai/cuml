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
from distutils.command.build import build as _build

import numpy

from setuputils import clean_folder
from setuputils import get_environment_option
from setuputils import get_cli_option

import versioneer
from cython_build_ext import cython_build_ext

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


if "--multigpu" in sys.argv:
    warnings.warn("Flag --multigpu is deprecated. By default cuML is"
                  "built with multi GPU support. To disable it use the flag"
                  "--singlegpu")
    sys.argv.remove('--multigpu')

if not libcuml_path:
    libcuml_path = '../cpp/build/'

##############################################################################
# - Cython extensions build and parameters -----------------------------------
#
# We create custom build steps for both `build` and `build_ext` for several
#   reasons:
# 1) Custom `build_ext` is needed to set `cython_build_ext.cython_exclude` when
#    `--singlegpu=True`
# 2) Custom `build` is needed to exclude pacakges and directories when
#    `--singlegpu=True`
# 3) These cannot be combined because `build` is used by both `build_ext` and
#    `install` commands and it would be difficult to set
#    `cython_build_ext.cython_exclude` from `cuml_build` since the property
#    exists on a different command.
#
# Using custom commands also allows combining commands at the command line. For
# example, the following will all work as expected:
# `python setup.py clean --all build --singlegpu build_ext --inplace`
# `python setup.py clean --all build --singlegpu install --record=record.txt`
# `python setup.py build_ext --debug --singlegpu`


class cuml_build(_build):

    def initialize_options(self):

        self.singlegpu = False
        super().initialize_options()

    def finalize_options(self):

        # distutils plain build command override cannot be done just setting
        # user_options and boolean options like build_ext below. Distribution
        # object has all the args used by the user, we can check that.
        self.singlegpu = '--singlegpu' in self.distribution.script_args

        libs = ['cuml++', 'cudart', 'cusparse', 'cusolver']

        include_dirs = [
            '../cpp/src',
            '../cpp/include',
            '../cpp/src_prims',
            cuda_include_dir,
            numpy.get_include(),
            '../cpp/build/faiss/src/faiss',
            os.path.dirname(sysconfig.get_path("include"))
        ]

        python_exc_list = []

        if (self.singlegpu):
            python_exc_list = ["*.dask", "*.dask.*"]
        else:
            libs.append('cumlprims_mg')
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
                      library_dirs=[
                          get_python_lib(),
                          libcuml_path,
                          cuda_lib_dir,
                          os.path.join(os.sys.prefix, "lib")
                      ],
                      libraries=libs,
                      language='c++',
                      extra_compile_args=['-std=c++17'])
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

    def build_extensions(self):
        def remove_flags(compiler, *flags):
            for flag in flags:
                try:
                    compiler.compiler_so = list(
                        filter((flag).__ne__, compiler.compiler_so)
                    )
                except Exception:
                    pass
        # Full optimization
        self.compiler.compiler_so.append("-O3")

        # Ignore deprecation declaraction warnings
        self.compiler.compiler_so.append("-Wno-deprecated-declarations")

        # adding flags to always add symbols/link of libcuml++ and transitive
        # dependencies to Cython extensions
        self.compiler.linker_so.append("-Wl,--no-as-needed")

        # No debug symbols, full optimization, no '-Wstrict-prototypes' warning
        remove_flags(
            self.compiler, "-g", "-G", "-O1", "-O2", "-Wstrict-prototypes"
        )
        cython_build_ext.build_extensions(self)

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

            print('--singlegpu: excluding the following Cython components:')
            pprint(cython_exc_list)

            # Append to base excludes
            self.cython_exclude = cython_exc_list + \
                (self.cython_exclude or [])

        super().finalize_options()


# Specify the custom build class
cmdclass = dict()
cmdclass.update(versioneer.get_cmdclass())
cmdclass["build"] = cuml_build
cmdclass["build_ext"] = cuml_build_ext

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
      setup_requires=['cython'],
      install_requires=install_requires,
      license="Apache",
      cmdclass=cmdclass,
      zip_safe=False)
