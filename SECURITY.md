# Security Policy

cuML is a GPU-accelerated machine-learning library — a scikit-learn-compatible
Python package (`cuml`), a Cython/CUDA core (`libcuml`), and multi-GPU /
multi-node distributed estimators built on Dask (`cuml.dask`). It is a
library, not a service: it runs in-process inside a Python interpreter, a
Dask worker, or a custom application, and inherits the caller's privilege.

Its security posture is shaped by the inputs it ingests — arrays, dataframes,
and (notably) serialized model artifacts — and by the language and process
boundaries it crosses.

## Reporting a Vulnerability

Please report security vulnerabilities privately through one of the channels
below. **Do not open a public GitHub issue, PR, or discussion** for a
suspected vulnerability.

1. **NVIDIA Vulnerability Disclosure Program (preferred)**
   <https://www.nvidia.com/en-us/security/>
   Submit through the NVIDIA PSIRT web form. This is the fastest path to
   triage and tracking.

2. **Email NVIDIA PSIRT**
   psirt@nvidia.com — encrypt sensitive reports with the
   [NVIDIA PSIRT PGP key](https://www.nvidia.com/en-us/security/pgp-key).

3. **GitHub Private Vulnerability Reporting**
   Use the **Security** tab on this repository → *Report a vulnerability*.

Please include, where possible:

- Affected component (e.g. FIL Treelite importer, `cuml.dask`, a specific
  estimator, the Cython bindings)
- cuML / libcuml version, CUDA version, GPU model, and OS
- Reproduction steps and a minimal proof-of-concept (PoC) input or model
- Impact assessment (memory corruption, code execution, DoS, info disclosure)
- Any relevant CWE / CVE identifiers

NVIDIA PSIRT will acknowledge receipt and coordinate triage, fix development,
and coordinated disclosure. More on NVIDIA's response process:
<https://www.nvidia.com/en-us/security/psirt-policies/>.

## Security Architecture & Context

**Classification:** Library (C++/CUDA core with Cython and Python bindings;
distributed estimators via Dask).

**Primary security responsibility:** Safely ingest tabular / numeric inputs
and serialized model artifacts, train and run inference on the GPU, and
hand results back to the caller without crashing the host process,
corrupting memory, or executing attacker-supplied code.

**Components and trust boundaries:**

- **libcuml** (`cpp/`) — C++/CUDA algorithm core. Major subsystems include
  `fil/` (Forest Inference Library), `randomforest/`, `glm/`, `svm/`,
  `kmeans/`, `dbscan/`, `hdbscan/`, `umap/`, `tsne/`, `arima/`, `knn/`,
  and the math primitives in `prims/`.
- **FIL — Forest Inference Library** (`cpp/src/fil`, `cpp/include/cuml/fil`,
  `python/cuml/cuml/fil`). Imports decision-forest models via
  [Treelite](https://treelite.readthedocs.io/) (XGBoost, LightGBM,
  scikit-learn ensembles converted to Treelite's binary format) and runs
  inference on the GPU. Treelite model bytes are an external-input attack
  surface.
- **Python estimators** (`python/cuml/cuml/{cluster,linear_model,svm,…}`) —
  scikit-learn-compatible classes. Inputs cross Cython bindings into the
  CUDA kernels; output shapes and dtypes are caller-controlled.
- **`cuml.accel`** (`python/cuml/cuml/accel`) — zero-code-change accelerator
  that proxies scikit-learn calls onto cuML. Inherits the trust assumptions
  of the underlying estimators.
- **`cuml.dask`** (`python/cuml/cuml/dask`) — multi-GPU / multi-node
  distributed estimators built on Dask Distributed. Cluster communication
  uses Dask's pickle-based serialization.
- **`cuml.comm`** (`python/cuml/cuml/comm/serialize.py`) — custom
  serialization helpers for cuML state across Dask workers.
- **Cython bindings** (`python/cuml/cuml/**/*.pyx`, `.pxd`) — translate
  Python array shapes, dtypes, and dimensions into the C++/CUDA layer.
  Integer casts at this boundary have historically been a source of
  memory-safety bugs.

**Out of scope for this policy:** vulnerabilities in CUDA, the NVIDIA driver,
Treelite itself, RAFT/pylibraft, cuDF, RMM, scikit-learn, scipy, joblib,
cupy, numba, Dask, or the JVM. Report those to their respective projects
(NVIDIA driver and CUDA bugs still go to PSIRT).

## Threat Model

The threats below trace to specific components in this repository. Several
have already been observed and remediated through the
[RAPIDS Security Audit](https://github.com/orgs/rapidsai/projects/207); they
are listed so that callers and integrators understand the classes of bugs
the library defends against.

1. **Malicious serialized model artifacts (FIL Treelite import).**
   FIL accepts Treelite-serialized models (typically produced from XGBoost,
   LightGBM, or scikit-learn ensembles) and constructs in-memory tree
   structures from them. A hostile model file can drive heap overflows,
   integer underflows, or OOB reads in the importer
   (`cpp/include/cuml/fil/treelite_importer.hpp` and surrounding code).
   Treat any Treelite model from an external source as untrusted.

2. **Pickle / joblib deserialization on persistence and distributed paths.**
   `cuml.dask` participates in Dask Distributed's pickle-based RPC, and
   scikit-learn-compatible estimators can be persisted via `joblib.dump`
   / `pickle.dump` and reloaded with `joblib.load` / `pickle.load`.
   Loading a model file or accepting a pickled object from an untrusted
   source is equivalent to arbitrary code execution by design of those
   formats — this is not specific to cuML, but cuML inherits the risk.

3. **Integer safety at the Cython / C++ boundary.**
   Array shapes, dimension counts, and index ranges crossing
   `python/cuml/cuml/**/*.pyx` into the C++/CUDA layer have caused
   unsigned underflows in bounds checks and overflows in dimension casts.
   A caller passing pathological shapes (zero-length, near-`SIZE_MAX`,
   or carefully chosen sizes that overflow on multiplication) can drive
   undefined behavior in kernels downstream.

4. **Pathological input shapes causing DoS or OOM.**
   Several estimators have super-linear memory or time complexity in their
   inputs (kNN, SVM kernel computations, hierarchical clustering, the
   pairwise-distance primitives in `prims/`). A caller able to influence
   `n_samples` × `n_features` × `n_neighbors` can exhaust GPU or host
   memory without producing a memory-safety warning/error.

5. **NaN / Inf / extreme numeric inputs.**
   Many CUDA kernels assume finite, well-formed `float`/`double` inputs.
   NaN-propagation through reduction kernels, divide-by-zero in
   normalization primitives, or extreme values feeding exp/log
   transforms can yield incorrect results or kernel hangs.

6. **`cuml.accel` zero-code-change interception.**
   When enabled, `cuml.accel` rewrites scikit-learn calls to run on cuML.
   This does not add new external attack surface, but it shifts which
   code path actually executes a user's `sklearn` script — auditors and
   reviewers should be aware that an `import sklearn` does not guarantee
   sklearn is doing the work.

7. **Distributed-training trust boundary.**
   `cuml.dask` estimators assume the Dask scheduler and workers are
   mutually authenticated and trusted. Untrusted peers can both run
   arbitrary code (via pickle) and submit malicious inputs (per threats
   1–5).

## Critical Security Assumptions

cuML is a library and inherits the caller's privilege; the following are
assumed of the caller / deployer.

- **Serialized model artifacts are trusted.**
  Treelite models loaded into FIL, and pickled / joblib-serialized cuML
  estimators, must come from a source the caller trusts. cuML does not
  authenticate, sandbox, or sanitize model bytes. Treat untrusted models
  as you would treat untrusted code.

- **Array inputs are well-formed.**
  cuML assumes caller-supplied arrays have valid dtypes, finite contents
  (or that the algorithm being called documents NaN/Inf behavior), and
  shapes that do not overflow when multiplied. Callers ingesting data
  from external sources should validate `shape`, `dtype`, and basic
  statistics before passing it to an estimator.

- **Resource limits are imposed externally.**
  cuML does not cap memory or time per call. Callers operating on
  untrusted inputs should run cuML in a process with cgroup / ulimit /
  container memory and CPU limits, and should bound the input dimensions
  they accept.

- **The host process, environment, and library load path are trusted.**
  cuML does not authenticate environment variables or sibling libraries
  loaded into its address space; a caller who lets an attacker influence
  these accepts the resulting risk.

- **Distributed cluster peers are mutually trusted.**
  Dask Distributed and the pickle-based serialization used by `cuml.dask`
  and `cuml.comm.serialize` are unsafe across trust boundaries. Run
  clusters on private, authenticated networks; do not accept pickled
  cuML payloads from untrusted sources.

- **Transport security is provided externally.**
  cuML does not implement TLS, authentication, or authorization. Any
  network use (Dask cluster traffic, model artifacts fetched from object
  storage) depends on the surrounding stack — Dask's TLS configuration,
  fsspec backends, or the caller's networking — for confidentiality and
  integrity.

- **GPU memory is not a confidentiality boundary.**
  Multiple processes sharing a GPU, or co-tenants on a shared host, may
  observe each other's GPU memory through driver-level side channels.
  cuML assumes the caller has provisioned the GPU appropriately (MIG,
  exclusive process, container isolation) when confidentiality matters.

## Supported Versions

Security fixes are issued against the current release line published on the
[RAPIDS release schedule](https://docs.rapids.ai/releases/). Older minor
releases are generally not backported; upgrade to the latest supported
version to receive fixes.

## Dependency Security

cuML tracks CVEs in its direct dependencies — notably `treelite`,
`scikit-learn`, `scipy`, `joblib`, `numpy`, `numba`, `cuda-python`, `cupy`,
`pylibraft`, `cudf`, and `rmm`. Dependency updates ship with regular
releases; high-severity upstream CVEs may trigger out-of-band patch
releases.
