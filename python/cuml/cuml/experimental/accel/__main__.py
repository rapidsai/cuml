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
@click.option(
    "--convert_to_sklearn",
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
@click.argument("args", nargs=-1)
def main(module, strict, convert_to_sklearn, format, output, args):

    if strict:
        os.environ["CUML_ACCEL_STRICT_MODE"] = "ON"

    install()

    # If the user requested a conversion, handle it and exit
    if convert_to_sklearn:

        # Load the accelerated estimator
        with open(convert_to_sklearn, "rb") as f:
            if format == "pickle":
                import pickle as serializer
            elif format == "joblib":
                import joblib as serializer
            else:
                raise ValueError(f"Serializer {format} not supported.")
            accelerated_estimator = serializer.load(f)

        # Convert to sklearn estimator
        sklearn_estimator = accelerated_estimator.as_sklearn()

        # Save using chosen format
        with open(output, "wb") as f:
            serializer.dump(sklearn_estimator, f)

        # Exit after conversion
        sys.exit(0)

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
