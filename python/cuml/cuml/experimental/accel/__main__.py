#
# Copyright (c) 2024, NVIDIA CORPORATION.
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

import click
import code
import logging
import runpy
import sys

from . import install
from cuml.internals import logger


@click.command()
@click.option("-m", "module", required=False, help="Module to run")
@click.option(
    "--profile",
    is_flag=True,
    default=False,
    help="Perform per-function profiling of this script.",
)
@click.option(
    "--line-profile",
    is_flag=True,
    default=False,
    help="Perform per-line profiling of this script.",
)
@click.argument("args", nargs=-1)
def main(module, profile, line_profile, args):
    """ """

    # todo (dgd): add option to lower verbosity
    logger.set_level(logger.level_debug)
    logger.set_pattern("%v")

    install()

    if module:
        (module,) = module
        # run the module passing the remaining arguments
        # as if it were run with python -m <module> <args>
        sys.argv[:] = [module] + args  # not thread safe?
        runpy.run_module(module, run_name="__main__")
    elif len(args) >= 1:
        # Remove ourself from argv and continue
        sys.argv[:] = args
        runpy.run_path(args[0], run_name="__main__")
    else:
        if sys.stdin.isatty():
            banner = f"Python {sys.version} on {sys.platform}"
            site_import = not sys.flags.no_site
            if site_import:
                cprt = 'Type "help", "copyright", "credits" or "license" for more information.'
                banner += "\n" + cprt
        else:
            # Don't show prompts or banners if stdin is not a TTY
            sys.ps1 = ""
            sys.ps2 = ""
            banner = ""

        # Launch an interactive interpreter
        code.interact(banner=banner, exitmsg="")


if __name__ == "__main__":
    main()
