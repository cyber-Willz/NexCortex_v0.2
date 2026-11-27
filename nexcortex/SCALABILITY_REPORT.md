# Scalability: finding, fix and measurements

## Problem (from the earlier land-registry run)
A graph with 92,151 ties made `Pipeline::new` request 68 GB and abort; sparsifying to top-12 ties per entity was a workaround.

## Root cause
Not the reservoir and not the number of entities. `hcrnn::topology::sym_laplacian` built the connectome Laplacian through
`spectral_hypergraph::dense_normalized_laplacian`, which materialises **dense E x E** weight and edge-degree matrices
(E = number of edges): 92,151² x 8 bytes = 67.9 GB. For a 2-uniform graph that matrix product is just
L = I − D^-½ A D^-½, which needs only O(n²).

## Fix
`sym_laplacian` now builds L directly (O(n²) memory and time). The old route is kept as `sym_laplacian_via_hypergraph`
for cross-checking. Regression test `direct_laplacian_equals_the_hypergraph_route` (3 random connectomes) shows
max |direct − hypergraph| < 1e-10. Existing Laplacian/harmonics test and all 28 nexcortex tests pass. No numerical behaviour changes.
Changed files: `nexgraph/human_connectome_rnn/src/topology.rs`, `.../tests/integration.rs`.

## Measurements (`examples/scale_eval.rs`; synthetic planted-community graph, ~46 ties/entity, 2 CPU cores, 7 GB RAM)
| Entities | Ties | Before fix | After: build | After: peak RAM | After: s/query |
|---|---|---|---|---|---|
| 500 | 5.2k | works (506 MB) | 0.3 s | 33 MB | 0.06 |
| 1,000 | 20k | **crashed** | 2.9 s | 122 MB | 0.26 |
| 2,000 | 46k | not run (predicted ~17 GB) | 1.3 s | 393 MB | 0.59 |
| 4,000 | 92k (the original failing size) | 68 GB request (seen earlier on 435 entities with 92k ties) | 4.1 s | 1.4 GB | 3.2 |
| 6,000 | 138k | — | 9.7 s | 3.1 GB | 6.4 |
| 8,000 | 185k | — | 13.3 s | 5.2 GB | 11.7 |
| 10,000 | 231k | — | — | **out of memory** (6.5 GB limit) | — |

"Before" column: 500 and 1,000 were run before the fix; 2,000+ were not run, the figures are predictions from E² x 8 bytes.
Sanity: on this synthetic task (recover the rest of the evidence's 20-member community) precision@10 was 0.98–1.00 at every size
against 0.002–0.034 for random. That checks the pipeline still works at scale; it is an easy planted structure, **not an accuracy claim**.

Original failing real graph: land-registry, 435 entities, 87,703 ties, **no sparsification** — now builds in 0.9 s and runs to completion.

## What is still limited
- Memory still grows ~n² (about 80 bytes x n²). The remaining cost is dense n x n storage elsewhere (the identity input matrix,
  the connectome weight/length matrices, the adjacency, the stored Laplacian, dense recurrent weights). On a 7 GB machine the
  ceiling is about **8–9k entities**; time per query grows ~n² as well (11.7 s at 8,000). Tens of thousands of entities need sparse
  storage for those structures (not done).
- Edge count is no longer the limit, entity count is.
- **Sparsification changed accuracy, not only speed.** On the dense 88k-tie land-registry graph the learned readout scored precision@5
  0.000 (company set, default split) vs 0.027 on the top-12 sparsified graph, with random at 0.010. Uniform dense ties wash out
  structure, so keep `TIE_TOPK` as a modelling choice even though it is no longer required for memory. Single split, small sample.
- Only synthetic graphs were scaled past 435 entities; I did not train a readout at scale.

## Reproduce
```
cargo run --release --example scale_eval -- <entities> <avg_degree> [queries] [community_size]
```
