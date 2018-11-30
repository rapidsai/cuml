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

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

import versioneer
from distutils.sysconfig import get_python_lib


install_requires = [
    'numpy',
    'cython'
]

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

cython_files = ['cuML/cuml.pyx']

extensions = [
    Extension("cuml",
              sources=cython_files,
              include_dirs=[numpy_include,
                            '../cuML/src',
                            '../cuML/external/ml-prims/src',
                            '../cuML/external/ml-prims/external/cutlass',
                            '../cuML/external/cutlass',
                            '../cuML/external/ml-prims/external/cub'],
              library_dirs=[get_python_lib()],
              libraries=['cuml'],
              language='c++',
              extra_compile_args=['-std=c++11'])
]

setup(name='cuml',
      description="cuML - RAPIDS ML Algorithms",
      version=versioneer.get_version(),
      classifiers=[
        # "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        # "Operating System :: OS Independent",
        "Programming Language :: Python",
        # "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
      ],
      author="NVIDIA Corporation",
      setup_requires=['cython'],
      ext_modules=cythonize(extensions),
      install_requires=install_requires,
      license="Apache",
      cmdclass=versioneer.get_cmdclass(),
      zip_safe=False
      )
