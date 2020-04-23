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
import re
import shutil
import subprocess
import warnings


def clean_folder(path):
    """
    Function to clean all Cython and Python artifacts and cache folders. It
    clean the folder as well as its direct children recursively.

    Parameters
    ----------
    path : String
        Path to the folder to be cleaned.
    """
    shutil.rmtree(path + '/__pycache__', ignore_errors=True)

    folders = glob.glob(path + '/*/')
    for folder in folders:
        shutil.rmtree(folder + '/__pycache__', ignore_errors=True)

        clean_folder(folder)

        cython_exts = glob.glob(folder + '/*.cpp')
        cython_exts.extend(glob.glob(folder + '/*.cpython*'))
        for file in cython_exts:
            os.remove(file)


def clone_repo(name, GIT_REPOSITORY, GIT_TAG, force_clone=False):
    """
    Function to clone repos if they have not been cloned already.
    Variables are named identical to the cmake counterparts for clarity,
    in spite of not being very pythonic.

    Parameters
    ----------
    name : String
        Name of the repo to be cloned
    GIT_REPOSITORY : String
        URL of the repo to be cloned
    GIT_TAG : String
        commit hash or git hash to be cloned. Accepts anything that
        `git checkout` accepts
    force_clone : Boolean
        Set to True to ignore already cloned repositories in
        external_repositories and clone

    """

    if not os.path.exists("external_repositories/" + name) or force_clone:
        print("Cloning repository "
              + name
              + " into external_repositories/"
              + name)
        subprocess.check_call(['git', 'clone',
                               GIT_REPOSITORY,
                               'external_repositories/' + name])
        wd = os.getcwd()
        os.chdir("external_repositories/" + name)
        subprocess.check_call(['git', 'checkout',
                              GIT_TAG])
        os.chdir(wd)
    else:
        print("Found repository "
              + name
              + " in external_repositories/"
              + name)


def get_repo_cmake_info(names, file_path):
    """
    Function to find information about submodules from cpp/CMakeLists file

    Parameters
    ----------
    name : List of Strings
        List containing the names of the repos to be cloned. Must match
        the names of the cmake git clone instruction
        `ExternalProject_Add(name`
    file_path : String
        Relative path of the location of the CMakeLists.txt (or the cmake
        module which contains ExternalProject_Add definitions) to extract
        the information.

    Returns
    -------
    results : dictionary
        Dictionary where results[name] contains an array,
        where results[name][0] is the url of the repo and
        repo_info[repo][1] is the tag/commit hash to be cloned as
        specified by cmake.

    """
    with open(file_path) as f:
        s = f.read()

    results = {}

    for name in names:
        res = re.findall(r'ExternalProject_Add\(' + re.escape(name)
                         + '\s.*GIT_REPOSITORY.*\s.*GIT_TAG.*',  # noqa: W605
                         s)

        res = re.sub(' +', ' ', res[0])
        res = res.split(' ')
        res = [res[2][:-1], res[4]]
        results[name] = res

    return results


def get_submodule_dependencies(repos,
                               file_path='../cpp/cmake/Dependencies.cmake',
                               libcuml_path='../cpp/build/'):

    """
    Function to check if sub repositories (i.e. submodules in git terminology)
    already exist in the libcuml build folder, otherwise will clone the
    repos needed to build the cuML Python package.

    Parameters
    ----------
    repos : List of Strings
        List containing the names of the repos to be cloned. Must match
        the names of the cmake git clone instruction
        `ExternalProject_Add(name`
    file_path : String
        Relative path of the location of the CMakeLists.txt (or the cmake
        module which contains ExternalProject_Add definitions) to extract
        the information. By default it will look in the standard location
        `cuml_repo_root/cpp`
    libcuml_path : String
        Relative location of the build folder to look if repositories
        already exist

    Returns
    -------
    result : boolean
        True if repos were found in libcuml cpp build folder, False
        if they were not found.
    """

    repo_info = get_repo_cmake_info(repos, file_path)

    if os.path.exists(libcuml_path):
        print("Third party modules found succesfully in the libcuml++ "
              "build folder.")

        return True

    else:

        warnings.warn("Third party repositories have not been found so they  "
                      "will be cloned. To avoid this set the environment "
                      "variable CUML_BUILD_PATH, containing the relative "
                      "path of the root of the repository to the folder "
                      "where libcuml++ was built.")

        for repo in repos:
            clone_repo(repo, repo_info[repo][0], repo_info[repo][1])

        return False
