#
# Copyright (c) 2018-2023, NVIDIA CORPORATION.
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
from pathlib import Path

from setuptools import find_packages

from skbuild import setup


##############################################################################
# - Helper functions
def get_cli_option(name):
    if name in sys.argv:
        print("-- Detected " + str(name) + " build option.")
        return True

    else:
        return False


def clean_folder(path):
    """
    Function to clean all Cython and Python artifacts and cache folders. It
    cleans the folder as well as its direct children recursively.

    Parameters
    ----------
    path : String
        Path to the folder to be cleaned.
    """
    shutil.rmtree(path + "/__pycache__", ignore_errors=True)

    folders = glob.glob(path + "/*/")
    for folder in folders:
        shutil.rmtree(folder + "/__pycache__", ignore_errors=True)

        clean_folder(folder)

        cython_exts = glob.glob(folder + "/*.cpp")
        cython_exts.extend(glob.glob(folder + "/*.cpython*"))
        for file in cython_exts:
            os.remove(file)


##############################################################################
# - Print of build options used by setup.py  --------------------------------

clean_artifacts = get_cli_option("clean")


##############################################################################
# - Clean target -------------------------------------------------------------

if clean_artifacts:
    print("-- Cleaning all Python and Cython build artifacts...")

    # Reset these paths since they may be deleted below
    treelite_path = False

    try:
        setup_file_path = str(Path(__file__).parent.absolute())
        shutil.rmtree(setup_file_path + "/.pytest_cache", ignore_errors=True)
        shutil.rmtree(
            setup_file_path + "/_external_repositories", ignore_errors=True
        )
        shutil.rmtree(setup_file_path + "/cuml.egg-info", ignore_errors=True)
        shutil.rmtree(setup_file_path + "/__pycache__", ignore_errors=True)

        clean_folder(setup_file_path + "/cuml")
        shutil.rmtree(setup_file_path + "/build", ignore_errors=True)
        shutil.rmtree(setup_file_path + "/_skbuild", ignore_errors=True)
        shutil.rmtree(setup_file_path + "/dist", ignore_errors=True)

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
# - Python package generation ------------------------------------------------

setup(
    include_package_data=True,
    packages=find_packages(include=["cuml", "cuml.*"]),
    zip_safe=False,
)
