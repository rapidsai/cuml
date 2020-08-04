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
    user_options = [
        ('language-level=', None,
         'Sets the python language syntax to use "2", "3", "3str".'),
        ("binding", None, "Sets the binding Cython binding directive"),
        ("profile", None, "Sets the profile Cython binding directive"),
        ("embedsignature", None, "Sets the binding Cython binding directive"),
        ("cython-exclude=", None, "Sets the binding Cython binding directive")
    ] + _build_ext.user_options

    boolean_options = ["binding", "profile", "embedsignature"
                       ] + _build_ext.boolean_options

    def initialize_options(self):

        self.language_level = None
        self.binding = None
        self.profile = None
        self.embedsignature = None
        self.cython_exclude = None
        super().initialize_options()

    def finalize_options(self):

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

            if (self.embedsignature is not None):
                self.embedsignature = bool(self.embedsignature)
                compiler_directives.update(
                    {"embedsignature": self.embedsignature})

            cythonize_kwargs = {}

            if (self.cython_exclude is not None):

                if (isinstance(self.cython_exclude, str)):
                    self.cython_exclude = list(self.cython_exclude)

                cythonize_kwargs.update({"exclude": self.cython_exclude})

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
