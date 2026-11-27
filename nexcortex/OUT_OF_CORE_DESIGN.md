# Out-of-core / mmap: analysis and design for nexcortex

## Short answer
**mmap is the right tool for three things in this system, and the wrong first tool for the thing that currently hurts.**
What hurts (memory ~n², 5.2 GB peak at 8,000 entities) is *dense storage of sparse data*: the graph at 8,000 entities is 0.3% dense.
Fixing that with sparse structures is worth ~100x and needs no mmap. mmap then earns its place for (1) a restart snapshot of the frozen
reservoir, (2) sharing one read-only copy across processes, and (3) the readout's training table, which has its own, separate blow-up.
Everything below is measured unless marked *estimate*.

## 1. Where the memory goes (8,000 entities, 185k ties; `examples/mem_profile.rs`)
n² x 8 B = 488 MB. Steady state after construction is 2.7 GB (~5.5 n²); the **transient peak is 5.2 GB inside `WorkingMemory::new`** (~2x steady).

| Structure | Where | Size at 8k | Really needs |
|---|---|---|---|
| `Connectome` weights + lengths (dense) | hcrnn | 2 n² = 976 MB | CSR ~9 MB |
| `scaled_to_mean_strength` clone, `ConnectomeRnn::new` clone | hcrnn | transient ~2–4 n² | none (scale in place / on read) |
| Input matrix W_in (identity, stored n x n) | hcrnn layout | n² = 488 MB | a diagonal (n values) |
| Laplacian | hcrnn | n² = 488 MB | CSR 4.4 MB; read only by the training-time energy penalty |
| `kb.adjacency()` dense, built **3 times** (components, to_connectome, kept by `Pipeline`) | nexcortex | n² each; 1 retained | build once; CSR |
| `allowed` mask | hcrnn | r² bytes = 64 MB | bitset / CSR pattern |
| Reservoir edges (post, pre, delay, kind) + weights | hcrnn | ~770k edges, tens of MB | already sparse |
| Rollout history `x`, `r` | per query | (T+1) x 16,000 x 8 x 2 = 11 MB | already small |

## 2. Where mmap helps and where it does not (`examples/ooc_bench.rs`)
Matrix-vector product is the per-step reservoir workload. Page cache dropped with `posix_fadvise(DONTNEED)` before "cold" runs (inside a VM; the host cache may still help, so cold numbers are optimistic).

| Layout | Entities | Size | Per step | Notes |
|---|---|---|---|---|
| Dense, mmap, warm | 8,000 | 1,953 MB | **284 ms** | RSS rises to 1.98 GB |
| Dense, mmap, cold first pass | 8,000 | 1,953 MB | 1,096 ms | disk-bound |
| CSR in RAM | 8,000 | 8.9 MB | **0.76 ms** | 370x faster than dense |
| CSR mmap warm / cold first step | 8,000 | 8.9 MB | 0.73 / 1.13 ms | bit-identical to RAM (max diff 0.0) |
| CSR in RAM | 100,000 | 107 MB | 29.6 ms | dense would be 305 GB |
| CSR mmap warm / cold first step | 100,000 | 107 MB | 28.4 / 119 ms | RSS 220 MB |
| CSR in RAM | 1,000,000 | 1,068 MB | 624 ms | dense would be 30 PB |
| CSR mmap warm / cold first step | 1,000,000 | 1,068 MB | 724 / 1,060 ms | warm penalty ~16% |

Conclusions:
1. **Sparse beats out-of-core by orders of magnitude.** mmap-ing the dense layout keeps it 390x slower than sparse-in-RAM and the pages still count toward RSS. Out-of-core only makes sense for data that is irreducibly large.
2. Once sparse, the reservoir graph fits in RAM up to ~10⁶ entities (1 GB), so mmap is **not needed for capacity** below that; it is needed for *restart and sharing*.
3. mmap adds no numerical change (bit-identical results) and ~0–16% warm overhead; a cold first pass is 1.4–4x slower but happens once.
4. At 10⁶ entities latency is the limit, not memory: ~0.6 s per step x 44 steps x 2 runs ≈ 55 s per query. That needs parallel rows (rayon), f32 values, and fewer settle steps, not mmap.

## 3. The readout's training table is a separate, real out-of-core problem
`Example.rows` stores one 26-feature row (208 B) per pool candidate, and `fit` makes a standardised copy (~424 B/row total in RAM). An **override** pool is *every entity not shown*, so rows scale as decisions x entities:

| Entities | Decisions (50% overrides) | Rows | RAM today | 256-negative sampling | + f32, no copy |
|---|---|---|---|---|---|
| 435 | 300 | 0.07 M | 0.03 GiB | 0.02 GiB | 0.004 GiB |
| 8,000 | 1,000 | 4 M | 1.6 GiB | 0.05 GiB | 0.014 GiB |
| 8,000 | 10,000 | 40 M | **15.8 GiB** | 0.53 GiB | 0.14 GiB |
| 100,000 | 10,000 | 500 M | **197 GiB** | 0.53 GiB | 0.14 GiB |

(Pure arithmetic from the row layout; not run at those sizes. Replay would also take ~4 s per decision at 8k entities today, i.e. hours for 10k decisions.)
The cheapest fix is algorithmic: **sample negatives** (sampled softmax with a log-correction, fixed seed so fitting stays deterministic). mmap is the fallback if exact full-pool fitting is required.

## 4. Proposed design, in priority order
**P0 — Sparse core (no mmap; ~100x memory, removes the transient peak). *Estimate:* steady state tens of MB instead of 2.7 GB at 8k.**
- `SparseConnectome` (CSR weights + lengths) behind the `Connectome` accessors (`weight`, `length`, `strength`); dense kept for the small-n/test path.
- Identity input as `InputMap::Diagonal(gain)` rather than an n x n parameter block; keep `forward_with_reference` as the equivalence oracle (already bit-identical test).
- Build the adjacency once, as CSR, and pass it down (today: 3 dense builds). Laplacian lazily/CSR. Scale weights on read, not by cloning.
- Acceptance: all 28 + 33 tests unchanged; `mem_profile` HWM at 8k below 300 MB (*target*); outputs bit-identical on the existing fixtures.

**P1 — Reservoir snapshot (mmap, read-only).** One file `reservoir-<fingerprint>.nxs`: header (magic, version, endianness, section offsets, FNV-1a fingerprint already computed by `Pipeline::fingerprint`, checksum), then CSR graph, reservoir edge arrays + calibrated weights, spectral `Structure` (embedding, clusters, PageRank, eigenvalues). On start: map, verify header and fingerprint, use in place; on mismatch or corruption, rebuild and rewrite.
- Benefit: skip the 15.7 s build at 8k (spectral 6.7 s + reservoir 6.4 s + conversions) on restart; N worker processes share one page-cache copy. *Target; cold-start time not yet measured.*
- Safety: write to a temp file and `rename` (never modify a mapped file in place); map read-only; validate every offset/length before casting; treat a short or truncated file as "rebuild", never as data (a truncated mapped file raises SIGBUS).

**P2 — Readout feature store + streaming fit.** Append-only columnar `f32` file (graph+reservoir columns, memory-mapped), keyed by (evidence-set hash, fingerprint) so replay is cached; the 4 human features are recomputed at fit time (the existing "decisions do not perturb features" test shows the other 22 do not depend on history). `fit` streams: one pass for mean/std, then Newton passes accumulating gradient and the 26x26 Hessian block-wise, so memory is O(26² + block). Fixed pass order keeps the fit deterministic. Combine with negative sampling.
- Acceptance: on the existing fixtures, weights agree with the in-memory fit to ≤1e-9 (full pool, f64 path) and held-out metrics stay within CI when sampling is on.

**P3 — Beyond RAM (≥10⁷ ties).** Store reservoir edges sorted by `post` so each step reads sequentially; replace the full (T+1) x N history with a ring buffer of `max_delay+1` steps; parallelise rows; f32 values. Only then does mmap'd streaming of the edge arrays matter. The benchmark shows ~1.7x first-pass and ≤16% warm cost, so the design is viable but latency (above) is the binding constraint. Spectral features at that size (nexgraph) were not tested.

## 5. Risks and cautions
- `unsafe` is unavoidable with mmap; isolate it in one module with length/alignment checks (u64/f64 sections 8-byte aligned).
- Development is on WSL2: keep snapshot and feature files on the Linux filesystem; memory-mapping files under `/mnt/c` (9p/drvfs) is slow and flaky.
- Mapped pages count toward RSS and cgroup limits until evicted; they are reclaimable, but memory limits can still cause I/O thrash on dense layouts, which is another reason not to map dense data.
- Do not assume mmap speeds anything up: it matters for restart and sharing. Per-query scratch, rollout buffers and readout weights stay in ordinary memory.
- Files are not portable across endianness/versions: header must carry both, and a mismatch means rebuild.
- Single-machine, single-writer assumed; concurrent writers need a lock file.

## 6. What I did and did not do
Added `examples/mem_profile.rs` and `examples/ooc_bench.rs` (dev-deps `memmap2`, `libc`) and measured the above. I did **not** implement P0–P3; the "target" and "estimate" figures are unmeasured. The benchmark uses synthetic random CSR, not the real reservoir, so it characterises storage and access patterns, not end-to-end query time.

Reproduce: `cargo run --release --example mem_profile -- 8000 46` and `cargo run --release --example ooc_bench -- 200000 9200000 20 1`.
