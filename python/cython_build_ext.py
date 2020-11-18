#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

import sys

# TODO: It should be possible to support Cython-less distribution following
# this guide and removing the direct import of Cython:
# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules

# Must import in this order:
#   setuptools -> Cython.Distutils.build_ext -> setuptools.command.build_ext
# Otherwise, setuptools.command.build_ext ends up inheriting from
# Cython.Distutils.old_build_ext which we do not want
import setuptools

try:
    from Cython.Distutils.build_ext import new_build_ext as _build_ext
except ImportError:
    from setuptools.command.build_ext import build_ext as _build_ext

import setuptools.command.build_ext


class cython_build_ext(_build_ext, object):
    """
    This class follows the design of `Cython.Distutils.build_ext.new_build_ext`
    to allow for parallel `cythonize()` but adds options for the various
    arguments that can be passed to `cythonize()` including separate options
    for `compiler_directives`. This build extension can be directly used in
    place of `new_build_ext` for any Cython project that needs to set global
    parameters in the build phase. See the documentation for more information
    on the `cythonize()` arguments.

    Parameters
    ----------
    annotate : bool, default=False
        If True, will produce a HTML file for each of the .pyx or .py files
        compiled. The HTML file gives an indication of how much Python
        interaction there is in each of the source code lines, compared to
        plain C code. It also allows you to see the C/C++ code generated for
        each line of Cython code. This report is invaluable when optimizing a
        function for speed, and for determining when to release the GIL: in
        general, a nogil block may contain only “white” code. See examples in
        Determining where to add types or Primes.
    binding : bool, default=True
        Controls whether free functions behave more like Python’s CFunctions
        (e.g. len()) or, when set to True, more like Python’s functions. When
        enabled, functions will bind to an instance when looked up as a class
        attribute (hence the name) and will emulate the attributes of Python
        functions, including introspections like argument names and
        annotations.

        Changed in version 3.0.0: Default changed from False to True
    cython_exclude : list of str
        When passing glob patterns as module_list, you can exclude certain
        module names explicitly by passing them into the exclude option.
    embedsignature : bool, default=False
        If set to True, Cython will embed a textual copy of the call signature
        in the docstring of all Python visible functions and classes. Tools
        like IPython and epydoc can thus display the signature, which cannot
        otherwise be retrieved after compilation.
    gdb_debug : bool, default=False
        Passes the `gdb_debug` argument to `cythonize()`. Setting up debugging
        for Cython can be difficult. See the debugging docs here
        https://cython.readthedocs.io/en/latest/src/userguide/debugging.html
    language_level : {"2", "3", "3str"}, default="2"
        Globally set the Python language level to be used for module
        compilation. Default is compatibility with Python 2. To enable Python 3
        source code semantics, set this to 3 (or 3str)
    linetrace : bool, default=False
        Write line tracing hooks for Python profilers or coverage reporting
        into the compiled C code. This also enables profiling. Default is
        False. Note that the generated module will not actually use line
        tracing, unless you additionally pass the C macro definition
        ``CYTHON_TRACE=1`` to the C compiler (e.g. using the setuptools option
        define_macros). Define ``CYTHON_TRACE_NOGIL=1`` to also include nogil
        functions and sections.
    profile : bool, default=False
        Write hooks for Python profilers into the compiled C code.
    """
    user_options = [
        ("annotate=",
         None,
         "Passes the `annotate` argument to `cythonize()`. See the Cython "
         "docs for more info."),
        ("binding",
         None,
         "Sets the binding Cython compiler directive. See the Cython docs for "
         "more info."),
        ("cython-exclude=",
         None,
         "Sets the exclude argument for `cythonize()`. See the Cython docs for"
         " more info."),
        ("embedsignature",
         None,
         "Sets the `embedsignature` Cython compiler directive. See the Cython "
         "docs for more info."),
        ("gdb-debug=",
         None,
         "Passes the `gdb_debug` argument to `cythonize()`. See the Cython "
         "docs for more info."),
        ('language-level=',
         None,
         'Sets the python language syntax to use "2", "3", "3str".'),
        ("linetrace=",
         None,
         "Passes the `linetrace` argument to `cythonize()`. See the Cython "
         "docs for more info."),
        ("profile",
         None,
         "Sets the profile Cython compiler directive. See the Cython docs for "
         "more info."),
    ] + _build_ext.user_options

    boolean_options = [
        "annotate",
        "binding",
        "embedsignature",
        "gdb-debug",
        "linetrace",
        "profile",
    ] + _build_ext.boolean_options

    def initialize_options(self):
        """
        Set the default values for the `user_options` to None to allow us to
        detect if they were set by the user
        """

        self.annotate = None
        self.binding = None
        self.cython_exclude = None
        self.embedsignature = None
        self.gdb_debug = None
        self.language_level = None
        self.linetrace = None
        self.profile = None

        super().initialize_options()

    def finalize_options(self):
        """
        Determines any user defined options and finalizes the Cython
        configuration before compilation
        """

        # Ensure the base build class options get set so we can use parallel
        self.set_undefined_options(
            'build',
            ('build_lib', 'build_lib'),
            ('build_temp', 'build_temp'),
            ('compiler', 'compiler'),
            ('debug', 'debug'),
            ('force', 'force'),
            ('parallel', 'parallel'),
            ('plat_name', 'plat_name'),
        )

        # If ext_modules is set, then build the cythonize argument list
        if self.distribution.ext_modules:
            if self.language_level is None:
                self.language_level = str(sys.version_info[0])

            assert self.language_level in (
                '2', '3',
                '3str'), 'Incorrect Cython language level ("{0}")'.format(
                    self.language_level)

            compiler_directives = dict(language_level=self.language_level)

            if (self.binding is not None):
                self.binding = bool(self.binding)
                compiler_directives.update({"binding": self.binding})

            if (self.profile is not None):
                self.profile = bool(self.profile)
                compiler_directives.update({"profile": self.profile})

            if (self.linetrace is not None):
                self.linetrace = bool(self.linetrace)
                compiler_directives.update({"linetrace": self.linetrace})

                # Also need the compiler directive. Only add if it hasnt been
                # specified yet
                for ext in self.distribution.ext_modules:
                    if (not hasattr(ext, "define_macros")):
                        ext.define_macros = []

                    found_macro = False

                    for mac in ext.define_macros:
                        if (mac[0] == "CYTHON_TRACE_NOGIL"):
                            found_macro = True
                            break

                    if not found_macro:
                        ext.define_macros.append(("CYTHON_TRACE_NOGIL", 1))

            if (self.embedsignature is not None):
                self.embedsignature = bool(self.embedsignature)
                compiler_directives.update(
                    {"embedsignature": self.embedsignature})

            cythonize_kwargs = {}

            if (self.annotate is not None):

                cythonize_kwargs.update({"annotate": self.annotate})

            if (self.cython_exclude is not None):

                if (isinstance(self.cython_exclude, str)):
                    self.cython_exclude = list(self.cython_exclude)

                cythonize_kwargs.update({"exclude": self.cython_exclude})

            if (self.gdb_debug is not None):

                cythonize_kwargs.update({"gdb_debug": self.gdb_debug})

            # Handle nthreads separately to mimic what Cython does
            nthreads = getattr(self, 'parallel', None)  # -j option in Py3.5+
            nthreads = int(nthreads) if nthreads else None

            # Delay import this to allow for Cython-less installs
            from Cython.Build.Dependencies import cythonize

            # Finally, cythonize the arguments
            self.distribution.ext_modules = cythonize(
                self.distribution.ext_modules,
                nthreads=nthreads,
                force=self.force,
                compiler_directives=compiler_directives,
                **cythonize_kwargs)

        # Skip calling super() and jump straight to setuptools
        setuptools.command.build_ext.build_ext.finalize_options(self)
