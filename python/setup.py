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
import numpy
install_requires = [
    'numpy',
    'cython'
]

cuda_include_dir = '/usr/local/cuda/include'
cuda_lib_dir = "/usr/local/cuda/lib"

if os.environ.get('CUDA_HOME', False):
    cuda_lib_dir = os.path.join(os.environ.get('CUDA_HOME'), 'lib64')
    cuda_include_dir = os.path.join(os.environ.get('CUDA_HOME'), 'include')

exc_list = []

libs = ['cuda', 'cuml']

if "--multigpu" not in sys.argv:
    exc_list.append('cuml/linear_model/linear_regression_mg.pyx')
    exc_list.append('cuml/decomposition/tsvd_mg.pyx')
else:
    libs.append('cumlMG')
    sys.argv.remove("--multigpu")

extensions = [
    Extension("*",
              sources=['cuml/*/*.pyx'],
              include_dirs=['../cuML/src',
                            '../cuML/external',
                            '../cuML/external/ml-prims/src',
                            '../cuML/external/ml-prims/external/cutlass',
                            '../cuML/external/cutlass',
                            '../cuML/external/ml-prims/external/cub',
                            numpy.get_include(),
                            cuda_include_dir],
              library_dirs=[get_python_lib(), "../cuML/build/"],
              runtime_library_dirs=[cuda_lib_dir],
              libraries=libs,
              language='c++',
              extra_compile_args=['-std=c++11'])
]

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
