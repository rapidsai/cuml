.. _developers_guide:

Developer's Guide
==================

This section contains information for developers who want to contribute to the cuML project.

Deprecation Policy
------------------

cuML follows the policy of deprecating code for one release prior to removal. This applies
to publicly accessible functions, classes, methods, attributes and parameters. During the
deprecation cycle the old name or value is still supported, but will raise a deprecation
warning when it is used.

Code in cuML should not use deprecated cuML APIs.


.. code-block:: python

    warnings.warn(
        (
            "Attribute `foo` was deprecated in version 25.06 and will be"
            " removed in 25.08. Use `metric` instead."
        ),
        FutureWarning,
    )


The warning message should always give both the version in which the deprecation happened
and the version in which the old behavior will be removed. The message should also include
a brief explanation of the change and point users to an alternative.

In addition, a deprecation note should be added in the docstring, repeating the information
from the warning message::

    .. deprecated:: 25.06
        Attribute `foo` was deprecated in version 25.06 and will be removed
        in 25.08. Use `metric` instead.


A deprecation requires a test which ensures that the warning is raised in relevant cases
but not in other cases. The warning should be caught in all other tests (using e.g., ``@pytest.mark.filterwarnings``).
