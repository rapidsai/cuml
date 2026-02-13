cuml
====

.. automodule:: cuml

.. _output-data-type-configuration:

Output Data Type Configuration
------------------------------

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: base.rst

   set_global_output_type
   using_output_type

.. _verbosity-levels:

Verbosity Levels
----------------

cuML follows a verbosity model similar to Scikit-learn's: The verbose parameter
can be a boolean, or a numeric value, and higher numeric values mean more
verbosity. The exact values can be set directly, or through the
cuml.common.logger module, and they are:

.. list-table:: Verbosity Levels
   :widths: 25 25 50
   :header-rows: 1

   * - Numeric value
     - cuml.common.logger value
     - Verbosity level
   * - 0
     - cuml.common.logger.level_enum.off
     - Disables all log messages
   * - 1
     - cuml.common.logger.level_enum.critical
     - Enables only critical messages
   * - 2
     - cuml.common.logger.level_enum.error
     - Enables all messages up to and including errors.
   * - 3
     - cuml.common.logger.level_enum.warn
     - Enables all messages up to and including warnings.
   * - 4 or False
     - cuml.common.logger.level_enum.info
     - Enables all messages up to and including information messages.
   * - 5 or True
     - cuml.common.logger.level_enum.debug
     - Enables all messages up to and including debug messages.
   * - 6
     - cuml.common.logger.level_enum.trace
     - Enables all messages up to and including trace messages.
