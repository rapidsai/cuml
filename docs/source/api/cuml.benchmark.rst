cuml.benchmark
==============

The built-in harness runs YAML-defined estimator workloads and writes JSON
observation artifacts. Start with the
:doc:`CLI benchmarking guide </developer_guide/benchmarking>` for commands,
suite/profile selection, manifest examples, timing semantics, and resume rules.

Suite loading and execution
---------------------------

For programmatic use, load a built-in suite name or a YAML path, then pass the
resolved suite to the runner. Direct callers are responsible for backend process
startup; the CLI handles this automatically, including ``cuml.accel`` activation.

.. autofunction:: cuml.benchmark.suite.load_suite_reference

.. autofunction:: cuml.benchmark.harness.run_suite

Schemas
-------

Suite manifests require ``version: 2``. The generated JSON Schema describes the
strict manifest structure; suite loading additionally checks semantic constraints
such as inference inputs, backend compatibility, and duplicate workload IDs.

.. autofunction:: cuml.benchmark.suite.suite_manifest_json_schema

Result artifacts use ``schema_version: 2`` and contain ``run`` metadata and
``results`` with raw warmup/measurement observations. The canonical JSON Schema
is packaged at ``cuml/benchmark/schemas/benchmark-result.schema.json``; retrieve
the resource with:

.. autofunction:: cuml.benchmark.schemas.benchmark_result_schema
