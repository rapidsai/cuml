cuml.benchmark
==============

The built-in harness runs YAML-defined estimator workloads and writes JSON
results files. Start with the
:doc:`CLI benchmarking guide </developer_guide/benchmarking>` for commands,
suite/profile selection, manifest examples, timing semantics, and resume rules.

Suite loading and execution
---------------------------

For programmatic use, load a built-in suite name or a YAML path, then pass the
resolved plan's individual ``runs`` to the single-provider runner. The loader
accepts an optional ``providers`` list; omission selects all declared
providers. Direct callers must isolate provider runs in separate processes and
configure provider backend startup. Prefer the CLI for automatic isolation, sequential
execution, and one results file per selected provider.

.. autofunction:: cuml.benchmark.suite.load_suite_reference

.. autofunction:: cuml.benchmark.harness.run_suite

Schemas
-------

The generated JSON Schema describes the strict suite manifest structure.
Suite loading additionally checks semantic constraints
such as inference inputs, provider compatibility, and duplicate workload IDs.

.. autofunction:: cuml.benchmark.suite.suite_manifest_json_schema

Results files contain ``run`` metadata and ``results`` with raw
warmup/measurement observations. The canonical JSON Schema
is packaged at ``cuml/benchmark/schemas/benchmark-result.schema.json``; retrieve
the resource with:

.. autofunction:: cuml.benchmark.schemas.benchmark_result_schema
