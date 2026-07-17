# NNLS Lawson Kernel: Detailed Walkthrough

This document explains `nnls_lawson_batched_kernel` in cuML and how it maps onto the MSA
(Mutational Signature Attribution) workload. The kernel implements the **Lawson–Hanson
active-set** method (same family as `scipy.optimize.nnls`), reformulated in **Gram space**
so each CUDA block can solve one masked NNLS problem without touching the tall design
matrix `A` again.

Source: `nnls_lawson.cuh`, launched via `nnls_batched_impl` in `nnls_batched.cuh`.

---

## 1. What Problem Is Being Solved?

For each problem `p` in a batch:


\min_{x \ge 0} \frac{1}{2}A x - b_p_2^2
\quad\text{subject to}\quad x_j = 0 \text{ if } \texttt{mask}(j,p)=0


Define the Gram form (constant terms dropped):


G = A^\top A \in \mathbb{R}^{n\times n}, \qquad c_p = A^\top b_p \in \mathbb{R}^n



f(x) = \frac{1}{2} x^\top G x - c^\top x, \qquad w(x) = c - Gx = -\nabla f(x)


At a KKT point for NNLS:

- **Active set** P = j : x_j > 0: w_j = 0 for j \in P
- **Inactive set** Z = j : x_j = 0: w_j \le 0 for j \in Z (no inactive column wants to enter)

The kernel tests optimality with: **stop when** \max_{j \in Z,\texttt{mask}(j)\neq 0} w_j \le \texttt{tol}.

---

## 2. Two-Level Parallelism

```mermaid
flowchart TB
  subgraph host["Host / cuBLAS (once per batch launch)"]
    G["G = AᵀA  (n×n)"]
    C["C = AᵀB  (n×P)"]
  end
  subgraph grid["CUDA grid: one block per problem"]
    B0["Block 0 → problem p=0"]
    B1["Block 1 → problem p=1"]
    BP["Block P-1 → problem p=P-1"]
  end
  host --> grid
  B0 --> X0["X[:,0]"]
  B1 --> X1["X[:,1]"]
  BP --> XP["X[:,P-1]"]
```




| Level      | Unit                           | Responsibility                                                 |
| ---------- | ------------------------------ | -------------------------------------------------------------- |
| **Grid**   | `blockIdx.x = p`               | One independent NNLS solve per block; `P` problems in parallel |
| **Block**  | `threadIdx.x ∈ [0, BlockSize)` | Cooperatively runs Lawson–Hanson for problem `p`               |
| **Warp**   | 32 threads                     | Lane cooperation inside RAFT block reductions (argmax, min)    |
| **Thread** | 1 lane                         | Strided loops over `n`, `n²`, `np²` indices                    |


**Launch:** `<<<n_problems, BlockSize, smem_bytes, stream>>>`

**Block size:** occupancy-driven, starting at 1024 threads (32 warps), halving down to 32 if
needed (`LawsonBlockDispatch`).

**MSA mapping:** one kernel launch solves up to `gpu_batch_size` masked leave-one-out trials
(default 4096; GPU optimised path uses 65536 in `process_samples_batch`).

---

## 3. Data Movement (Global ↔ Shared ↔ Registers)

### 3.1 Before the Kernel (Host / Device Global Memory)

From `nnls_batched_impl`:

```cpp
// G = A^T A  (n x n),  C = A^T B  (n x P).  Formed once and reused by every
// problem in the batch.
raft::linalg::gemm(handle, At_view, A_view, G.view());
raft::linalg::gemm(handle, At_view, B_view, C.view());
```


| Buffer  | Shape              | Precision         | Shared by    | Notes                                                          |
| ------- | ------------------ | ----------------- | ------------ | -------------------------------------------------------------- |
| `A`     | `(m, n)` col-major | `float64` typical | all problems | MSA: `(n_channels, n_signatures)`                              |
| `B`     | `(m, P)`           | same              | —            | Target spectra; `b_index` selects column per problem in Python |
| `G`     | `(n, n)`           | `T`               | all blocks   | Read-only in kernel, straight from global (L2-cached)          |
| `C`     | `(n, P)`           | `T`               | —            | Block `p` reads column `p`                                     |
| `masks` | `(n, P)` uint8     | —                 | —            | Optional; masked columns never enter active set                |
| `X`     | `(n, P)`           | `T`               | —            | Output                                                         |


**Key design choice:** `A` is never read inside the kernel; all work uses `G` and `c = C[:,p]`.
`G` stays in **global memory** (never staged into shared) and is read directly — the whole
grid shares it, so the L2 cache absorbs the reuse. Because the active-set Cholesky factor is
maintained incrementally, each block touches `G` only once per outer iteration: the active
columns for the projected gradient plus the single entering column for the factor append.

### 3.2 Inside One Block (Dynamic Shared Memory)

Layout from `lawson_smem_layout`:


| Array                | Size           | Stored as                         | Role                                   |
| -------------------- | -------------- | --------------------------------- | -------------------------------------- |
| `c`                  | `n`            | `T`                               | RHS projection `Aᵀb`                   |
| `x`                  | `n`            | `T`                               | Current solution                       |
| `w`                  | `n`            | `T`                               | Gradient / dual residual `c − Gx` (also removed-index scratch) |
| `s`                  | `n`            | `T`                               | Trial solve + Cholesky RHS + downdate scratch |
| `red_val`, `red_idx` | `WarpSize` each (contiguous) | scratch            | RAFT block-reduction scratch + scalar broadcast |
| `idx`                | `n`            | `int`                             | Compact active-set indices             |
| `Gp`                 | `n×n` (ld = n) | `narrow_t<T>` (float if T=double) | Incrementally-maintained Cholesky factor `L` |
| `act`                | `n`            | `int8`                            | 1 = active, 0 = inactive               |

The Gram matrix `G` is **not** in this list — it lives in global memory only.

**Memory trick:** `Gp` (the factor `L`) is stored in float when `T=double`, halving its
`n²` footprint; arithmetic still accumulates in `T`. `Gp` is placed after the wider `T`/`int`
arrays and before the 1-byte `act[]` so its 4-byte float alignment holds without misaligning
the `double` arrays (a `4·n²`-byte block is not a multiple of 8 for odd `n²`).

**Typical smem (double, narrowed L):** dropping the resident `G` array halves the dominant
`2n²` term to `n²`, so the footprint is roughly half of the previous design (e.g. ~18 KB for
`n=65`), leaving more room for larger `n` and/or higher occupancy under the 48–96 KB cap.

### 3.3 Per-Block Data Flow Diagram

```
GLOBAL                          SHARED (one block)                    GLOBAL
──────                          ──────────────────                    ──────
G[n,n]  (stays in global, L2-cached; read directly, never staged)
C[:,p]  ──load──────────►  S.c[n]
masks[:,p] ──view───────►  mask_col (read in argmax only)

Each outer iter:
  G[:,idx], S.x ──matvec (global read of active cols)──► S.w[n]
  S.w, S.act, mask ──reduce──► j_star ; activate (idx, act, np++)
  G[idx,j*] ──bordering append (one global column)──► extend S.Gp = L

Each inner iter (no G reads):
  S.c, S.idx ──gather──► S.s[np] = c_P
  S.Gp = L ──tri-solve──► s (unconstrained LS on P)
  S.x, S.s ──line search──► updated S.x, S.act, S.idx
  on prune: Givens downdate of S.Gp = L for the dropped columns

S.x[n] ──store──────────► X[:,p]
```

---

## 4. Lawson–Hanson at Three Levels of Detail

### Level A — One Paragraph

Maintain active set `P`. Repeatedly: compute dual residual `w = c − Gx`; if some inactive
masked column has `w_j > tol`, add it to `P`; solve the unconstrained quadratic on `P`; if
the result is nonnegative, accept it; otherwise take the largest feasible step toward it,
zero out binding variables, and repeat. Stop when no column wants to enter or iteration budget
is exhausted.

### Level B — Pseudocode (Algorithmic)

```
INPUT: G, c, mask, tol, max_iter
INIT: x ← 0, act ← 0, idx ← [], n_active ← 0

FOR outer = 1 .. max_iter:
    w ← c − G·x                          // projected gradient (global, active cols only)

    (j*, w*) ← argmax{ w_j : act[j]=0 AND mask[j]≠0 }
    IF j* < 0 OR w* ≤ tol: BREAK         // optimal

    act[j*] ← 1; idx[n_active] ← j*; n_active++
    L ← chol_append(L, G[idx, j*])       // bordering update; reads one global column
    IF new pivot ≤ 0:                    // activation breaks positive-definiteness
        undo j*; BREAK outer

    FOR inner = 1 .. (3n+1):              // inner budget (no G reads)
        np ← n_active
        c_P ← c[idx[0:np]]
        s ← solve(L L^T · s = c_P)      // triangular solve on the maintained factor

        IF min(s) > 0:                   // feasible unconstrained step
            x ← 0; x[idx] ← s
            BREAK inner

        α ← min{ x_j / (x_j − s_j) : s_j ≤ 0, x_j − s_j > 0 }
        x[idx] ← x[idx] + α·(s − x[idx])

        // Drop variables that hit zero, then downdate the factor to match
        compact idx; act[j]=0 where x_j ≈ 0; record removed local positions
        L ← chol_downdate(L, removed positions)   // Givens delete, descending
        IF n_active = 0: BREAK inner

WRITE x to output
```

### Level C — What Each CUDA Phase Does (Block Steps)

#### Phase 1–2: Initialization (once per block)

```cpp
for (int j = tid; j < n; j += BlockSize) {
  S.c[j]   = C(j, p);
  S.x[j]   = T(0);
  S.act[j] = 0;
}
```

`G` is **not** copied into shared memory; it is read directly from global memory where it is
needed. There is no longer an `O(n²)` staging load.


| Step                      | Parallelism            | Work                |
| ------------------------- | ---------------------- | ------------------- |
| Load `c`, zero `x`, `act` | `tid` strides over `n` | One column of `C`   |
| Init scalars              | `tid==0`               | `n_active=0`        |


#### Phase 3: Outer Loop — Add a Column

**Step 3a — Gradient** (`block_matvec_gradient`, parallel matvec per row):


w_j = c_j - \sum_{kk=0}^{np-1} G_{j,\texttt{idx}[kk]}\, x_{\texttt{idx}[kk]}


`G` is read straight from global memory. Since `x` is zero outside the active set, only the
`np` active columns contribute: each thread `tid` owns rows `j = tid, tid+BlockSize, …` and
loops over the active columns `kk`. For a fixed active column the lanes stride over rows `j`,
so consecutive lanes read consecutive (column-major) elements of `G` — a coalesced access,
well served by L2 since the whole grid shares `G`.

**Step 3b — Entering column** (`block_argmax_inactive`):


j^ = \arg\max_{\substack{j: \texttt{act}[j]=0  \texttt{mask}(j)\neq 0}} w_j


Parallel pattern: per-thread local max → warp shuffle → cross-warp reduction in warp 0 →
broadcast via `red_val[0]`, `red_idx[0]`.

**Convergence (outer):** if `j* < 0` or `max_w ≤ tol`, break.

**Step 3c — Activate:** thread 0 sets `act[j*]=1`, appends `j`* to `idx`, increments `n_active`.

**Step 3d — Factor append** (`block_chol_append`, once per outer iteration):

Extend the lower Cholesky factor `L` (kept in `Gp`, leading dimension `n`) with the entering
column, read directly from global `G`:


a_{12}[i] = G_{\texttt{idx}[i],\,j^*},\quad l = L_{11}^{-1} a_{12},\quad L_{22} = \sqrt{a_{22} - l\cdot l}


This is a device analogue of `raft::linalg::choleskyRank1Update` (a host/cuBLAS routine, not
callable inside a block). The trace-based Tikhonov term of the old from-scratch factorisation
is replaced by a **per-pivot guard**: `\varepsilon = 10^{-7}` (float) / `10^{-14}` (double)
times `a_{22}`, added before the `sqrt`. A non-positive pivot (`a_{22}+\varepsilon - l\cdot l
\le 0`) means the activation breaks positive-definiteness — thread 0 rejects it (undo `j*`)
and the outer loop stops, exactly reproducing the old Cholesky-failure behaviour.

#### Phase 4: Inner Loop — Solve on Current Active Set (no `G` reads)

The factor `L` is already current, so the inner loop never touches `G`.

**Step 4a — RHS gather:** `s_i = c_{\texttt{idx}[i]}` (parallel over `np`).

**Step 4b — Triangular solve** (`block_chol_solve`):

Solve L L^\top s = c_P via forward Ly=c_P and back L^\top s=y, reading `L` from `Gp` with
leading dimension `n` (only the `np × np` leading block is used, so no repacking is needed).
Sequential in row index `i`, but each update row parallelizes over trailing/prior indices —
standard small-n cooperative pattern.

**Step 4d — Feasibility test:**


s_{\min} = \min_{j=0}^{n_p-1} s_j


If s_{\min} > 0: trial solution is feasible — zero `x`, scatter `x[idx[j]] = s[j]`, exit
inner loop (outer iteration complete).

**Step 4e — Line search (partial step):**

For binding indices (s_j \le 0):


\alpha = \min_{\substack{j: s_j \le 0  x_j - s_j > 0}} \frac{x_j}{x_j - s_j}


Then:


x_j \leftarrow x_j + \alpha(s_j - x_j) \quad \forall j \in P


**Step 4f — Drop zeros and downdate the factor:**

Thread 0 compacts `idx`, clearing `act[j]` and `x[j]` where x_j \le \varepsilon
(`1e-15` double / `1e-12` float), and records the removed **local** positions. The factor is
then shrunk to match with `block_chol_delete_one` applied to each removed position in
**descending** order (so lower-index deletions stay valid as `np` shrinks). Deleting an
interior row/column reduces to a positive rank-1 Cholesky update of the trailing block by the
below-diagonal part of the deleted column, applied with Givens rotations — no `G` read.

**Inner exit conditions:** feasible solution found; `n_active==0`; inner budget `3n+1`
exhausted. (The positive-definiteness check now lives in the Phase-3d append, not the inner
loop.)

#### Phase 5: Writeback

```cpp
for (int j = tid; j < n; j += BlockSize)
  X(j, p) = S.x[j];
```

---

## 5. Convergence: When Does a Block Stop?


| Criterion            | Condition                                         | Meaning                                        |
| -------------------- | ------------------------------------------------- | ---------------------------------------------- |
| **Optimality**       | \max_{j \in Z,\text{masked}} w_j \le \texttt{tol} | No inactive column wants weight; KKT satisfied |
| **Outer cap**        | `outer == max_iter`                               | Default `max_iter = 3n+1` if unset             |
| **Inner cap**        | `inner == 3n+1` per outer step                    | Prevents infinite inner cycling                |
| **Append failure**   | non-positive pivot in the factor append           | Undo last activation; stop outer               |
| **Empty active set** | `n_active == 0` after binding                     | Degenerate; exit inner                         |


Default `tol = 1e-6` (`NnlsBatchedParams`). MSA does not override this in
`_gpu_batched_solve_and_score`.

**MSA-level convergence** (outside the kernel): greedy signature removal stops per-sample when
no leave-one-out trial drops similarity by more than `weak_threshold` (default 0.01), or only
one signature remains.

---

## 6. Parallelization Patterns Inside One Block


| Operation                | Threads do                                                    | Synchronization          |
| ------------------------ | ------------------------------------------------------------- | ------------------------ |
| Init `c,x,act`           | Strided `for (i=tid; i<n; i+=BlockSize)`                      | `__syncthreads`          |
| Gradient `w = c - Gx`    | One row per strided thread; inner loop over active cols of global `G` | `__syncthreads`  |
| Argmax / min / min-α     | Local scan → `raft::blockRankedReduce` (min/max) / `raft::blockReduce` (count) | `__syncthreads` |
| Factor append            | Forward solve (t0 divide + parallel axpy) + block-reduced `l·l`; t0 pivot | `__syncthreads` |
| Tri-solve                | Per row: t0 divide; parallel axpy                             | `__syncthreads` each row |
| Line search update       | Strided over `np`                                             | `__syncthreads`          |
| Compact active set       | Thread 0 only                                                 | `__syncthreads`          |
| Factor downdate (prune)  | t0 compaction; per-pivot Givens (t0 rotation + parallel apply) | `__syncthreads` each `k` |


**Occupancy trade-off:** larger `BlockSize` (1024) helps parallelize `n²` factor updates;
smaller blocks allow more concurrent problems when smem-bound or batch is huge
(`LawsonBlockDispatch` targets `8 × resident_blocks ≥ n_problems`).

---

## 7. MSA Dimension Variables (from Project Data)

Symbols follow cuML (`m`, `n`, `P`) and MSA (`n_channels`, `n_signatures`, etc.).


| Symbol              | Code name                    | Meaning                                   | Typical MSA values                                                                                                             |
| ------------------- | ---------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `m`                 | `n_channels`, `A.shape[0]`   | Mutation channels (rows of `A`)           | **96** (SBS-96), 192, 288, 1536, 4608; **78** (DBS), **83** (ID)                                                               |
| `n`                 | `n_signatures`, `A.shape[1]` | Reference signatures (cols of `A`)        | **65** (SBS-96 catalog), **54** (SBS-192), **10** (SBS-288/1536/4608), **11** (DBS), **17** (ID)                               |
| `P`                 | `n_problems`                 | Problems per kernel launch                | **4096** default `--gpu_batch_size`; up to **65536** in GPU optimised path; ≤ `#active_samples × #active_sigs` per greedy step |
| `np`                | `sm_n_active`                | Active-set size during solve              | Starts at `n` (all masked-in sigs); shrinks to **~1–15** after optimisation                                                    |
| `n_samples`         | `B.shape[1]`                 | Tumour samples in one run                 | **1000** (`SIM_ESCC` SBS-96 input)                                                                                             |
| `max_iter`          | kernel param                 | Outer Lawson iterations                   | `**3n+1`** default → **196** for `n=65`, **34** for `n=11`                                                                     |
| `tol`               | kernel param                 | Dual residual threshold                   | **1e-6** (cuML default)                                                                                                        |
| `inner_budget`      | `3*n+1`                      | Inner loop cap per outer step             | Same as default `max_iter`                                                                                                     |
| `BlockSize`         | template param               | Threads per block                         | **1024** down to **32** (occupancy dispatch)                                                                                   |
| `smem`              | `lawson_smem_bytes(n)`       | Dynamic shared memory / block (one `n²` factor, G in global) | **~18 KB** (`n=65`), **~26 KB** (`n=78`), **~1 KB** (`n=10`)                                                 |
| `chunk_size`        | `--gpu_batch_size`           | Max NNLS problems per `nnls_batched` call | **4096** (CLI default), **65536** (GPU batch in code)                                                                          |
| `samples_per_group` | `chunk_size // n_sig`        | Samples grouped per greedy step           | **~63** for SBS-96 (`4096//65`), **~372** (`65536//65`)                                                                        |
| `n_trials` / step   | `Σ active_sigs per sample`   | Leave-one-out problems per greedy step    | Roughly `**#active_samples × avg_active_sigs`** (≤ chunk)                                                                      |


### Example: `run_NNLS.py -d SIM_ESCC -t SBS -c 96 -x --use_gpu --nnls-backend cuml --nnls-solver lawson`


| Quantity         | Value                                                                       |
| ---------------- | --------------------------------------------------------------------------- |
| `A` (signatures) | `(96, 65)`                                                                  |
| `B` (mutations)  | `(96, 1000)` samples                                                        |
| `G` precompute   | `(65, 65)` — once per batch launch                                          |
| `C` precompute   | `(65, P)` — `P` = problems in chunk                                         |
| Per kernel block | One masked leave-one-out fit for one sample                                 |
| Dominant cost    | Many thousands of small `n=65` active-set solves, not one big least-squares |


### Mutation-Type Quick Reference


| Mutation | Context      | `(m, n)` channels × signatures |
| -------- | ------------ | ------------------------------ |
| SBS      | 96 (default) | (96, 65)                       |
| SBS      | 192          | (192, 54)                      |
| SBS      | 288          | (288, 10)                      |
| SBS      | 1536         | (1536, 10)                     |
| SBS      | 4608         | (4608, 10)                     |
| DBS      | —            | (78, 11)                       |
| ID       | —            | (83, 17)                       |


For large contexts (1536, 4608), Lawson is less attractive: `m` is huge but the kernel only
uses `n` (small); the **cuBLAS `G = AᵀA` setup** dominates, and other solvers (APG/CD) may be
preferable. For the default MSA hot path (SBS-96, `n≈65`), Lawson in Gram form is a good fit.

---

## 8. End-to-End MSA + Kernel Flow

```
run_NNLS.py: optimise_signatures_batched()
  └─ each greedy step: batched_solve_and_score()
       └─ chunks of masks[:, start:stop]  (P problems)
            └─ cuml nnls_batched(solver="lawson")
                 ├─ cuBLAS: G = AᵀA, C = AᵀB     [global, once]
                 └─ nnls_lawson_batched_kernel
                      ├─ block p: load G, c=C[:,p], solve masked NNLS
                      └─ write X[:,p]
                 └─ (optional) fitted = A @ X for similarity score
```

Each block is **fully independent** after `G` and `C` are built; the batch scales with GPU SM
count × occupancy. Convergence is **per-block exact active-set** (up to `tol` and numerical
safeguards), while MSA **outer greedy logic** decides which signature masks to try next.