# Testing the claim: sparse/matrix-free Laplacian -> 100x less memory, seconds -> milliseconds

## Verdict
| Claim | Result |
|---|---|
| Sparse / matrix-free form matches the dense operator to 1e-10 | **True, and trivially so.** Max difference 3e-16 to 7e-16 on random vectors; Rayleigh quotient 2e-13 (n = 2,000 / 4,000 / 8,000). |
| Memory cut of ~100x | **True for the Laplacian alone at 8,000 entities** (CSR 111x, matrix-free 83x). At 4,000: 56x / 42x; at 2,000: 28x / 21x. The ratio is about n²/(2E+n), so it grows with n and shrinks with density. |
| Query latency from seconds to milliseconds | **False for this change.** The Laplacian is never touched at query time (it is read only by the training-time energy penalty, `rayleigh()`), so replacing it changes query latency by exactly 0. Even in the best case a dense L·x costs 3–56 ms and a CSR 0.1–0.6 ms, but nothing queries it. |
| Overall memory cut ~100x | **False.** The dense Laplacian is 8% / 21% / 28% of process memory at 2k / 4k / 8k entities. Removing it saves that share, not 100x. |

So the claim holds for the operator in isolation and fails for the system as a whole.

## What actually dominated query latency
Profiling showed `analyze` = feature fields + **two** reservoir runs (the second, in reversed evidence order, is the stability check) + ranking.
Inside the reservoir, the input term did an N x N multiply **every step** even though nexcortex sets the input matrix to a scaled identity
(~85 ms per step at 16,000 units). I changed `forward_with` to iterate only the non-zero entries of the input matrix (collected once per call).
The old loop is kept as `forward_with_reference`, and a new test proves the two are **bit-identical** (`to_bits` equality on x, r, y; random dense
input matrix and the identity case). All 33 hcrnn integration tests and all 28 nexcortex tests pass.

Query latency (s), synthetic planted-community graph, ~46 ties/entity, 2 cores:

| Entities | Before | After | `memory.run` before -> after |
|---|---|---|---|
| 1,000 | 0.26 | 0.18 | — |
| 2,000 | 0.59 | 0.49 | 0.24 -> 0.12 |
| 4,000 | 3.2 | 1.3–1.6 | 1.21 -> 0.29 |
| 8,000 | 10.9–11.7 | 4.0 | 3.74 -> 0.66 |

That is a 2–3x end-to-end gain at the large sizes (5.7x for the reservoir run itself), from a different change than the Laplacian. It is **not** milliseconds:
at 8,000 entities a query is still ~4 s. Of that, about 0.6 s is feature fields, ~1.3 s the two reservoir runs, and roughly 2 s is ranking/feature extraction that I did not profile.
Peak memory is unchanged (5.2 GB at 8,000), set by other dense n x n structures built at construction.

## On patentability
I can't assess that, and I would not rely on my view. What I can say technically: the construction I used (L = I − D^-½AD^-½ from an edge list in CSR or
matrix-free form) is the standard sparse normalized Laplacian; `spectral_hypergraph`'s own docs already point to matrix-free operators for large graphs;
and the one genuine inefficiency was an avoidable dense E x E detour in this codebase, which is a bug fix. Whether any of it is novel is a question for a
patent attorney and a prior-art search, and the measured results above (no query-latency gain from the Laplacian, 8–28% of memory) are the facts such a review would start from.

## Reproduce
```
cargo run --release --example laplacian_check -- 8000 46     # exactness, memory, apply time, per-query breakdown
cargo run --release --example scale_eval -- 8000 46 5
cd ../nexgraph/human_connectome_rnn && cargo test --release --test integration -- direct_laplacian sparse_input
```
Files changed: `human_connectome_rnn/src/network.rs` (sparse input, reference forward), `.../tests/integration.rs`; new `examples/laplacian_check.rs`.
