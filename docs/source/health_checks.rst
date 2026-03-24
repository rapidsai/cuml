.. SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
.. SPDX-License-Identifier: Apache-2.0

Health Checks
=============

cuML provides a small set of health checks (smoke tests) to verify that cuML is
working correctly after installation or as part of automated processes such as
CI. These checks are also used by the RAPIDS CLI's ``rapids doctor`` command
when the CLI is installed.

Run standalone
--------------

You can run all cuML health checks from the command line:

.. code-block:: console

   python -m cuml.health_checks

Use ``--verbose`` or ``-v`` for extra output when a check passes. The command
exits with 0 if all checks pass, or 1 if any check fails.

Run via RAPIDS CLI
------------------

When `rapids-cli <https://github.com/rapidsai/rapids-cli>`_ is installed, the
same cuML checks are registered as plugins and run as part of:

.. code-block:: console

   rapids doctor

See the `rapids-cli documentation
<https://github.com/rapidsai/rapids-cli#check-plugins>`_ for how checks are
discovered and how to run with ``--verbose`` or filter by name.
