#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import code
import pickle
import runpy
import sys

import click
import joblib

from cuml.internals import logger
from cuml.accel.core import install


@click.command()
@click.option("-m", "module", required=False, help="Module to run")
@click.option(
    "--convert-to-sklearn",
    type=click.Path(exists=True),
    required=False,
    help="Path to a pickled accelerated estimator to convert to a sklearn estimator.",
)
@click.option(
    "--format",
    "format",
    type=click.Choice(["pickle", "joblib"], case_sensitive=False),
    default="pickle",
    help="Format to save the converted sklearn estimator.",
)
@click.option(
    "--output",
    type=click.Path(writable=True),
    default="converted_sklearn_model.pkl",
    help="Output path for the converted sklearn estimator file.",
)
@click.option(
    "--disable-uvm",
    is_flag=True,
    default=False,
    help="Disable UVM (managed memory) allocations.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase output verbosity (can be used multiple times, e.g. -vv). Default shows warnings only.",
)
@click.argument("args", nargs=-1)
def main(
    module, convert_to_sklearn, format, output, disable_uvm, verbose, args
):
    default_logger_level_index = list(logger.level_enum).index(
        logger.level_enum.warn
    )
    logger_level_index = max(0, default_logger_level_index - verbose)
    logger_level = list(logger.level_enum)[logger_level_index]
    logger.set_level(logger_level)
    logger.set_pattern("%v")

    # Enable acceleration
    install(disable_uvm=disable_uvm)

    # If the user requested a conversion, handle it and exit
    if convert_to_sklearn:

        with open(convert_to_sklearn, "rb") as f:
            if format == "pickle":
                serializer = pickle
            elif format == "joblib":
                serializer = joblib
            accelerated_estimator = serializer.load(f)

        sklearn_estimator = accelerated_estimator.as_sklearn()

        with open(output, "wb") as f:
            serializer.dump(sklearn_estimator, f)

        sys.exit()

    if module:
        (module,) = module
        # run the module passing the remaining arguments
        # as if it were run with python -m <module> <args>
        sys.argv[:] = [module, *args.args]  # not thread safe?
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
