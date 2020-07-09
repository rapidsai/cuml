import sys

if 'setuptools' in sys.modules:
    try:
        from Cython.Distutils.build_ext import new_build_ext as _build_ext
    except ImportError:
        from setuptools.command.build_ext import build_ext as _build_ext
else:
    from distutils.command.build_ext import build_ext as _build_ext


import setuptools.command.build_ext

class new_build_ext(_build_ext, object):
    user_options = [
        (
            'language-level=',
            None,
            'Sets the python language syntax to use "2", "3", "3str".' 
        ),
        (
            "binding=",
            None,
            "Sets the binding Cython binding directive"
        ),
        (
            "profile=",
            None,
            "Sets the profile Cython binding directive"
        ),
        (
            "embedsignature=",
            None,
            "Sets the binding Cython binding directive"
        ),
        (
            "cython-exclude=",
            None,
            "Sets the binding Cython binding directive"
        )
    ] + _build_ext.user_options

    boolean_options = [
        "binding", "profile", "embedsignature"
    ] + _build_ext.boolean_options

    def initialize_options(self):

        print("cuml_build_ext::initialize_options")

        self.language_level = None
        self.binding = None
        self.profile = None
        self.embedsignature = None
        self.cython_exclude = None
        super(new_build_ext, self).initialize_options()
        
    def finalize_options(self):

        print("cuml_build_ext::finalize_options")

        self.set_undefined_options('build',
                                   ('build_lib', 'build_lib'),
                                   ('build_temp', 'build_temp'),
                                   ('compiler', 'compiler'),
                                   ('debug', 'debug'),
                                   ('force', 'force'),
                                   ('parallel', 'parallel'),
                                   ('plat_name', 'plat_name'),
                                   )

        if self.distribution.ext_modules:
            if self.language_level is None:
                self.language_level = str(sys.version_info[0])

            assert self.language_level in ('2', '3', '3str'), 'Incorrect Cython language level ("{0}")'.format(self.language_level)

            compiler_directives = dict(language_level=self.language_level)

            if (self.binding is not None):
                compiler_directives.update({ "binding": bool(self.binding) })

            if (self.profile is not None):
                compiler_directives.update({ "profile": bool(self.profile) })

            if (self.embedsignature is not None):
                compiler_directives.update({ "embedsignature": bool(self.embedsignature) })

            cythonize_kwargs = {
            }

            if (self.cython_exclude is not None):

                if (type(self.cython_exclude) == str):
                    self.cython_exclude = list(self.cython_exclude)

                cythonize_kwargs.update({ "exclude": self.cython_exclude })
            
            nthreads = getattr(self, 'parallel', None)  # -j option in Py3.5+
            nthreads = int(nthreads) if nthreads else None
            
            from Cython.Build.Dependencies import cythonize
            
            self.distribution.ext_modules[:] = cythonize(
                self.distribution.ext_modules, 
                nthreads=nthreads, 
                force=self.force,
                compiler_directives=compiler_directives,
                **cythonize_kwargs
            )
        setuptools.command.build_ext.build_ext.finalize_options(self)
