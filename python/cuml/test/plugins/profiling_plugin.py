"""pytest: avoid already-imported warning: PYTEST_DONT_REWRITE."""

from __future__ import absolute_import

import sys
import os
import cProfile
import pstats
import pipes
import errno
from hashlib import md5

import six
import pytest

LARGE_FILENAME_HASH_LEN = 8


def clean_filename(s):
    forbidden_chars = set(r'/?<>\:*|"')
    return six.text_type("".join(
        c if c not in forbidden_chars and ord(c) < 127 else '_' for c in s))


class Profiling(object):
    """Profiling plugin for pytest."""
    svg = False
    svg_name = None
    profs = []
    combined = None

    def __init__(self, svg, dir=None):
        self.svg = svg
        self.dir = 'prof' if dir is None else dir[0]
        self.profs = []
        self.gprof2dot = os.path.abspath(
            os.path.join(os.path.dirname(sys.executable), 'gprof2dot'))
        if not os.path.isfile(self.gprof2dot):
            # Can't see gprof in the local bin dir, we'll just have to hope
            # it's on the path somewhere
            self.gprof2dot = 'gprof2dot'

    def pytest_sessionstart(self, session):  # @UnusedVariable
        try:
            os.makedirs(self.dir)
        except OSError:
            pass

    def pytest_sessionfinish(self, session, exitstatus):  # @UnusedVariable
        if self.profs:
            combined = pstats.Stats(self.profs[0])
            for prof in self.profs[1:]:
                combined.add(prof)
            self.combined = os.path.abspath(
                os.path.join(self.dir, "combined.prof"))
            combined.dump_stats(self.combined)
            if self.svg:
                self.svg_name = os.path.abspath(
                    os.path.join(self.dir, "combined.svg"))
                t = pipes.Template()
                t.append("{} -f pstats $IN".format(self.gprof2dot), "f-")
                t.append("dot -Tsvg -o $OUT", "-f")
                t.copy(self.combined, self.svg_name)

    def pytest_terminal_summary(self, terminalreporter):
        if self.combined:
            terminalreporter.write(
                "Profiling (from {prof}):\n".format(prof=self.combined))
            pstats.Stats(
                self.combined,
                stream=terminalreporter).sort_stats('cumulative').print_stats(
                    r"^((?!.*cuml/test|_pytest|pluggy|numba.*).)*$", 0.2)
        if self.svg_name:
            terminalreporter.write(
                "SVG profile in {svg}.\n".format(svg=self.svg_name))

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        prof_filename = os.path.abspath(
            os.path.join(self.dir, clean_filename(item.name) + ".prof"))

        os.makedirs(os.path.dirname(prof_filename), exist_ok=True)

        prof = cProfile.Profile()
        prof.enable()
        yield
        prof.disable()
        try:
            prof.dump_stats(prof_filename)
        except EnvironmentError as err:
            if err.errno != errno.ENAMETOOLONG:
                raise

            if len(item.name) < LARGE_FILENAME_HASH_LEN:
                raise

            hash_str = md5(item.name.encode(
                'utf-8')).hexdigest()[:LARGE_FILENAME_HASH_LEN]
            prof_filename = os.path.join(self.dir, hash_str + ".prof")
            prof.dump_stats(prof_filename)
        self.profs.append(prof_filename)


def pytest_addoption(parser):
    """pytest_addoption hook for profiling plugin"""
    group = parser.getgroup('Profiling')
    group.addoption("--profile",
                    action="store_true",
                    help="generate profiling information")
    group.addoption(
        "--profile-svg",
        action="store_true",
        help="generate profiling graph (using gprof2dot and dot -Tsvg)")
    group.addoption("--pstats-dir",
                    nargs=1,
                    help="configure the dump directory of profile data files")


def pytest_configure(config):
    """pytest_configure hook for profiling plugin"""
    profile_enable = any(
        config.getvalue(x) for x in ('profile', 'profile_svg'))
    if profile_enable:

        # Monkey patch cprofile reporting
        import pstats

        def f8(x):

            if (x >= 1.0):
                ret = "%8.3fs" % x
                return ret
            elif (x >= 0.001):
                ret = "%7.3fms" % (x * 1000.0)
                return ret
            else:
                ret = "%7.3fµs" % (x * 1000000.0)
                return ret

        pstats.f8 = f8

        config.pluginmanager.register(
            Profiling(config.getvalue('profile_svg'),
                      config.getvalue('pstats_dir')))
