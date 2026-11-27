# nexgraph v0.26

Merges v0.25 (supplied as `nexgraph_v0_24_tar.gz`; its Cargo.toml is 0.25.0) into the `nex_workspace_perf` workspace and
adds two traversal fast paths found by the cross-database benchmark (`bench/BENCHMARK_REPORT.md`).

## Changes
| Area | Before | Now |
|---|---|---|
| `algo.shortestPath` | bidirectional BFS read every incident edge body (version chain + `Arc` deref), in and out lists together | nodes with only directed edges walk the `outs` / `ins` list and read binary edges from compact metadata (`edge_fast`); hyperedges use the body path as before |
| Variable-length `-[:L*a..b]->` | `incident_dir` + body per edge | plain binary edges (no edge properties) read from compact metadata; other edges unchanged |
| `Entity::und` | n/a | new flag, set when an undirected hyperedge (empty head) ever touches the node. Undirected edges are only in `incident`, so such nodes keep the general path. Never cleared (conservative after deletes). |

Results are unchanged (benchmark checksums identical across 3 runs; `tests/v026.rs` compares both paths with a reference
model on graphs with binary edges, directed and undirected hyperedges, self-loops, parallel edges, several labels and deleted edges).

## Measured (100k nodes / 1M edges, 1 vCPU, median of 3 runs)
| Workload | v0.22 | v0.25 | v0.26 | FalkorDB |
|---|---|---|---|---|
| Point lookup | 3.2 µs | 3.3 µs | **3.3 µs** | 90.5 µs |
| 1-hop count | 6.9 µs | 7.1 µs | **7.2 µs** | 135.7 µs |
| 2-hop count | 28.6 µs | 28.8 µs | **30.3 µs** | 167.5 µs |
| Distinct reach ≤3 hops | 1.1 ms | 1.0 ms | **738.5 µs** | 409.0 µs |
| Shortest path | 2.0 ms | 1.7 ms | **1.3 ms** | 284.8 µs |
| Full edge count | 4.0 ms | 3.4 ms | **3.5 ms** | 30.9 ms |
| Group by, top-5 | 32.6 ms | 29.1 ms | **30.4 ms** | 219.2 ms |
| `WITH` aggregate + filter | 29.0 ms | 31.0 ms | **31.7 ms** | 211.3 ms |
| Bulk load (1M edges) | 1352.8 ms | 1109.6 ms | **1217.3 ms** | 77004.0 ms |
| PageRank, 30 iterations | 251.6 ms | 261.0 ms | **280.6 ms** | n/a |

Run-to-run noise on untouched rows is about ±10%, so only the path (-25%) and reach (-26%) changes are real.

## Not changed / remaining gaps
FalkorDB is still ~4.5x faster on shortest path and ~1.8x on 3-hop reach; DuckDB is ~2.4x faster on group-by. Untried ideas:
reusable scratch arrays for BFS (about 2.4 MB allocated per call on this graph), BFS-based `count(DISTINCT b)` for
variable-length patterns, columnar property storage, batched execution (see `docs/performance.md`).

## Tests
Full suite: 291 passed, 0 failed (release, default features), including 2 new in `tests/v026.rs`. `nexcortex` type-checks
against this tree (rustc 1.92+; its tests were not run). HTTPS server, PostgreSQL paths and the WASM frontend were not exercised.
