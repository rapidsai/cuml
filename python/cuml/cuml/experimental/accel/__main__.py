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
import os
import runpy
import sys

from . import install


@click.command()
@click.option("-m", "module", required=False, help="Module to run")
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Turn strict mode for hyperparameters on.",
)
@click.argument("args", nargs=-1)
def main(module, strict, args):
    
    if strict:
        os.environ["CUML_ACCEL_STRICT_MODE"] = "ON"

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
