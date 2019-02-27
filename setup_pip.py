#
# Copyright (c) 2019, NVIDIA CORPORATION.
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
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import shutil
from distutils.sysconfig import get_python_lib
from cmake_setuptools import CMakeBuildExt, CMakeExtension, \
    convert_to_manylinux, InstallHeaders, distutils_dir_name

cuda_version = ''.join(os.environ.get('CUDA', '9.2').split('.')[:2])

name = 'cuml-cuda{}'.format(cuda_version)
version = os.environ.get('GIT_DESCRIBE_TAG', '0.0.0.dev0').lstrip('v')

cudf_version = os.environ.get('MIN_CUDF_VERSION', version)
cudf_version_split = cudf_version.split('.')
cudf_version_split[1] = str(int(cudf_version_split[1]) + 1)
cudf_next_minor = '.'.join(cudf_version_split)
max_cudf_version = os.environ.get('MAX_CUDF_VERSION', cudf_next_minor)

install_requires = [
    'numpy',
    'cython>=0.29<0.30',
    'cudf-cuda{}>={},<{}'.format(cuda_version, cudf_version, max_cudf_version)
]


cython_files = ['python/cuml/*/*.pyx']

extensions = [
    CMakeExtension('cuml', 'cuML'),
    Extension("*",
              sources=cython_files,
              include_dirs=['cuML/src',
                            'cuML/external/ml-prims/src',
                            'cuML/external/ml-prims/external/cutlass',
                            'cuML/external/cutlass',
                            'cuML/external/ml-prims/external/cub'],
              library_dirs=[get_python_lib(), distutils_dir_name('lib')],
              libraries=['cuml'],
              language='c++',
              runtime_library_dirs=['$ORIGIN'],
              extra_compile_args=['-std=c++11'])
]

# setup does not clean up the build directory, so do it manually
shutil.rmtree('build', ignore_errors=True)

setup(name=name,
      description='cuML - RAPIDS ML Algorithms',
      long_description=open('README.md', encoding='UTF-8').read(),
      long_description_content_type='text/markdown',
      url='https://github.com/rapidsai/cuml',
      version=version,
      classifiers=[
          "Intended Audience :: Developers",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7"
      ],
      packages=find_packages(where='python'),
      package_dir={
          'cuML': 'python/cuml'
      },
      author="NVIDIA Corporation",
      license='Apache 2.0',
      install_requires=install_requires,
      python_requires='>=3.6,<3.8',
      ext_modules=cythonize(extensions),
      cmdclass={
          'build_ext': CMakeBuildExt,
          'install_headers': InstallHeaders
      },
      headers=['cuML/src/'],
      zip_safe=False
      )

convert_to_manylinux(name, version)
