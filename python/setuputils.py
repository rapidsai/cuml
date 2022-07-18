#
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
import sys

from pathlib import Path


def get_environment_option(name):
    env_variable = os.environ.get(name, False)

    if not env_variable:
        print("-- " + name + " environment variable not set.")

    else:
        print("-- " + name + " detected with value: " + str(env_variable))

    return env_variable


def get_cli_option(name):
    if name in sys.argv:
        print("-- Detected " + str(name) + " build option.")
        return True

    else:
        return False


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


def use_raft_package(raft_path, cpp_build_path,
                     git_info_file=None):
    """
    Function to use the python code in RAFT in package.raft

    - If RAFT symlink already exists, don't change anything. Use setup.py clean
        if you want to change RAFT location.
    - Uses RAFT located in $RAFT_PATH if $RAFT_PATH exists.
    - Otherwise it will look for RAFT in the libcuml build folder,
        located either in the default location ../cpp/build or in
        $CUML_BUILD_PATH.
    - Otherwise it will clone RAFT into _external_repositories.
        - Branch/git tag cloned is located in git_info_file in this case.

    """
    if os.path.islink('cuml/raft'):
        raft_path = os.path.realpath('cuml/raft')
        # walk up two dirs from `python/raft`
        raft_path = os.path.join(raft_path, '..', '..')
        print("-- Using existing RAFT folder")
    elif isinstance(raft_path, (str, os.PathLike)):
        print('-- Using RAFT_PATH argument')
    elif os.environ.get('RAFT_PATH', False) is not False:
        raft_path = str(os.environ['RAFT_PATH'])
        print('-- Using RAFT_PATH environment variable')
    else:
        raft_path, raft_cloned = \
            clone_repo_if_needed('raft', cpp_build_path,
                                 git_info_file=git_info_file)
        raft_path = os.path.join('../', raft_path)

    raft_path = os.path.realpath(raft_path)
    print('-- RAFT found at: ' + str(raft_path))

    try:
        os.symlink(
            os.path.join(raft_path, 'python/raft'),
            os.path.join('cuml/raft')
        )
    except FileExistsError:
        os.remove(os.path.join('cuml/raft'))
        os.symlink(
            os.path.join(raft_path, 'python/raft'),
            os.path.join('cuml/raft')
        )

    return os.path.join(raft_path, 'cpp/include')


def clone_repo_if_needed(name, cpp_build_path=None,
                         git_info_file=None):
    if git_info_file is None:
        git_info_file = \
            _get_repo_path() + '/cpp/cmake/thirdparty/get_raft.cmake'

    if cpp_build_path is None or cpp_build_path is False:
        cpp_build_path = _get_repo_path() + '/cpp/build/_deps/'

    repo_cloned = get_submodule_dependency(name,
                                           cpp_build_path=cpp_build_path,
                                           git_info_file=git_info_file)

    if repo_cloned:
        repo_path = (
            _get_repo_path() + '/python/_external_repositories/' + name + '/')
    else:
        repo_path = os.path.join(cpp_build_path, name + '-src/')

    return repo_path, repo_cloned


def get_submodule_dependency(repo,
                             git_info_file,
                             cpp_build_path):
    """
    Function to check if sub repositories (i.e. submodules in git terminology)
    already exist in the libcuml build folder, otherwise will clone the
    repos needed to build the cuML Python package.

    Parameters
    ----------
    repo : Strings or list of Strings
        Name of the repos to be cloned. Must match
        the names of the cmake git clone instruction
        `ExternalProject_Add(name`
    git_info_file : String
        Relative path of the location of the CMakeLists.txt (or the cmake
        module which contains ExternalProject_Add definitions) to extract
        the information. By default it will look in the standard location
        `cuml_repo_root/cpp`
    cpp_build_path : String
        Relative location of the build folder to look if repositories
        already exist

    Returns
    -------
    result : boolean
        True if repos were found in libcuml cpp build folder, False
        if they were not found.
    """

    if isinstance(repo, str):
        repos = [repo]

    repo_info = get_repo_cmake_info(repos, git_info_file)

    if os.path.exists(os.path.join(cpp_build_path, repos[0] + '-src/')):
        print("-- Third party modules found succesfully in the libcuml++ "
              "build folder:")
        print("  " + str(cpp_build_path))

        return False

    else:

        print("-- Third party repositories have not been found so they "
              "will be cloned. To avoid this set the environment "
              "variable CUML_BUILD_PATH, containing the absolute "
              "path to the build folder where libcuml++ was built. ")

        for repo in repos:
            clone_repo(repo, repo_info[repo][0], repo_info[repo][1])

        return True


def clone_repo(name, GIT_REPOSITORY, GIT_TAG,
               location_to_clone='_external_repositories/', force_clone=False):
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
        _external_repositories and clone
    location_to_clone: String (default: '_external_repositories/')
        Name that will contain clone of repository if needed.
    """

    if not os.path.exists(location_to_clone + name) or force_clone:
        print("Cloning repository " + name + " into " + location_to_clone +
              name)
        subprocess.check_call(['git', 'clone',
                               GIT_REPOSITORY,
                               location_to_clone + name])
        wd = os.getcwd()
        os.chdir(location_to_clone + name)
        subprocess.check_call(['git', 'checkout',
                               GIT_TAG])
        os.chdir(wd)
    else:
        print("Found repository " + name + " in _external_repositories/" +
              name)


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
        repo = re.findall(r'\s.*GIT_REPOSITORY.*', s)
        repo = repo[-1].split()[-1]
        fork = re.findall(r'\s.*FORK.*', s)
        fork = fork[-1].split()[-1]
        repo = repo.replace("${PKG_FORK}", fork)
        tag = re.findall(r'\s.*PINNED_TAG.*', s)
        tag = tag[-1].split()[-1]
        if tag == 'branch-${CUML_BRANCH_VERSION_raft}':
            loc = _get_repo_path() + '/cpp/CMakeLists.txt'
            with open(loc) as f:
                cmakelists = f.read()
                tag = re.findall(r'\s.*project\(CUML VERSION.*', cmakelists)
                tag = tag[-1].split()[2].split('.')
                tag = 'branch-{}.{}'.format(tag[0], tag[1])

        results[name] = [repo, tag]

    return results


def _get_repo_path():
    python_dir = Path(__file__).resolve()
    return str(python_dir.parent.parent.absolute())
