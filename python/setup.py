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
from os.path import join as pjoin
import versioneer
from distutils.sysconfig import get_python_lib


install_requires = [
    'numpy',
    'cython'
]

def find_in_path(name, path):
    "Find a file in a search path"
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                                   'located in your $PATH.')

        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not '
                                   'be located in %s' % (k, v))

    return cudaconfig


CUDA = locate_cuda()

cython_files = ['cuML/cuml.pyx']

extensions = [
    Extension("cuml",
              sources=cython_files,
              include_dirs=['../cuML/src',
                            '../cuML/external',
                            '../cuML/external/ml-prims/src',
                            '../cuML/external/ml-prims/external/cutlass',
                            '../cuML/external/cutlass',
                            '../cuML/external/ml-prims/external/cub',
                            CUDA["include"]],
              library_dirs=[get_python_lib()],
              runtime_library_dirs=[CUDA["lib64"]],
              libraries=['cuda', 'cuml'],
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
      packages=find_packages(include=['cuML', 'cuML.*']),
      install_requires=install_requires,
      license="Apache",
      cmdclass=versioneer.get_cmdclass(),
      zip_safe=False
      )
