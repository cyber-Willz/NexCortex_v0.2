# Changelog

## Learned readout release (2026-11-23/24)
- **New:** `src/readout.rs`, a learned readout over graph features + reservoir trajectory + human decisions (26 features, softmax over candidates, exact Newton fit, leave-one-out and forward-chained evaluation, save/load, fingerprint check against the reservoir).
- **New:** CLI `--train-readout`, `--readout`, `--readout-groups`, `--l2`, `--prior-weight`.
- **New:** decisions are logged with their evidence, chosen and refused entities (`DecisionRecord`, `record_decision_logged`, `decision_log`).
- **Fix:** logged decisions no longer add a `DECIDED_ON` edge; NexGraph diffuses over every hyperedge, so those edges changed features and leaked labels.
- **Fix:** an `override` training example now competes against every entity that was not shown (it used to be appended to the shown set, which made it always the lowest-ranked candidate in its pool).
- **New:** `Analysis.table`, `baseline_order`, `final_order`; `Candidate.learned`, `baseline_rank`; `PipelineConfig.rerank_pool`.
- **New:** live-data runs: `examples/attack_live.rs` + `scripts/attack_prepare.py` (MITRE ATT&CK), `examples/tabular_eval.rs` + `scripts/tabular_prepare.py`, `scripts/tabular_score.py` (flat tables).
- Tests: 16 pipeline + 8 readout, all passing. Reports: `ATTACK_LIVE_RUN_REPORT.md`, `ACCOUNTING_DATASET_ACCURACY_REPORT.md`.

## Claude API explainer
- `ClaudeExplainer` (`src/claude.rs`), `--claude <model>`, `examples/llm_eval.rs`, `tests/claude.rs` (4 tests), dep `serde_json`.

## Land-registry evaluation
- `scripts/landreg_prepare.py`; `TIE_TOPK` env var in `examples/attack_live.rs` (sparsify ties; dense 92k-tie graphs exhaust memory); `BELIZE_LAND_REGISTRY_ACCURACY_REPORT.md`.

## Scalability fix
- `hcrnn::topology::sym_laplacian` is now O(n^2) (was O(E^2) via dense hypergraph matrices; 92k edges = 68 GB). Old route kept as `sym_laplacian_via_hypergraph` + equivalence test. `examples/scale_eval.rs`, `SCALABILITY_REPORT.md`. Ceiling now ~8-9k entities on 7 GB.

## Query latency
- `ConnectomeRnn::forward_with` iterates only non-zero input weights (bit-identical to `forward_with_reference`, tested); reservoir run 5.7x faster at 8k entities. `examples/laplacian_check.rs`, `LAPLACIAN_CLAIM_TEST_REPORT.md`.

## Out-of-core analysis
- `OUT_OF_CORE_DESIGN.md`; `examples/mem_profile.rs`, `examples/ooc_bench.rs`; dev-deps memmap2, libc. No runtime behaviour change.
