# nexcortex — complete system summary

## 1. What it is
A Rust decision-support pipeline. Given a few observed entities ("evidence"), it ranks which *other* entities deserve
attention, explains the ranking in plain language, and records the human operator's decision so the system can learn from it.

```
NexGraph (persistent knowledge)         kb.rs        nexgraph (property hypergraph DB)
  -> spectral / diffusion features      features.rs  heat diffusion, PageRank, spectral clusters, paths
  -> connectome RNN (working memory)    memory.rs    human_connectome_rnn (hcrnn), frozen reservoir
  -> learned readout (optional)         readout.rs   graph + reservoir trajectory + human decisions
  -> LLM explanation layer              explain.rs / claude.rs   template | local Burn LLM | Claude API
  -> human decision maker               human.rs     decision written back into NexGraph
```
Workspace: crates `nexcortex` (this), `nexgraph`, `nexgen_llm`. About 2.7k lines in `src/`.

## 2. Stages
1. **Knowledge base (`kb.rs`)** — entities and weighted relations built with `KbBuilder`; validated (no unknown endpoints,
   duplicate names, non-positive weights); must be one connected component for the spectral stages. Decisions are logged as
   properties only (chosen, refused, evidence names/severity, logged), *not* as hyperedges, because NexGraph diffuses over all
   hyperedges and decision edges would leak labels into features.
2. **Features (`features.rs`)** — heat diffusion from the evidence at several time scales, PageRank, spectral communities,
   shortest paths, direct ties.
3. **Working memory (`memory.rs`)** — a connectome-constrained recurrent network (Dale's law, delayed excitatory/inhibitory
   paths, stable spectral radius). Evidence is injected in observed order; entities that keep activity after the input stops
   rank highest ("salience"). The network is frozen; nothing in it is trained.
4. **Learned readout (`readout.rs`, optional)** — a conditional-logit (listwise softmax) model over 26 features
   (graph 11, reservoir 11, human 4) with one weight vector shared across entities. Fitted by damped Newton + Cholesky with
   L2 (default 1.0); deterministic. Trained by *replay*: example k sees only history before decision k (no leakage).
   Approve → pool is the shown top-k; override → pool is all entities not shown. It re-ranks the top `rerank_pool`
   (default 10, ≥ `top_k`). Saved files are bound to their reservoir by an FNV-1a fingerprint (memory config, feature config,
   prior weight, top_k, entity names, adjacency); loading on a different setup is refused.
   `evaluate` reports leave-one-out and forward-chained held-out numbers against the untrained rules.
5. **Explanation (`explain.rs`, `claude.rs`)** — sees only `Brief::facts()`. Three explainers:
   - `TemplateExplainer`: deterministic, always available.
   - `LlmExplainer` (feature `llm`): local Burn model (Llama/Mistral/Qwen2/Qwen3 safetensors).
   - `ClaudeExplainer`: Anthropic Messages API through a `Transport` trait (curl in production, key passed on stdin, never in
     argv or body; mock in tests). Honors `ANTHROPIC_BASE_URL`; records calls, errors, grounded/unverified counts, latency, tokens.
   Safeguards on both LLM paths: facts-only system prompt; injection guard (local); grounding check (must name the top candidate
   and mention no knowledge-base entity outside the brief); failed check → template leads and model text is shown as
   `[unverified model text]`; any error or empty text → template fallback. **Explanation never changes the ranking.**
   The grounding check is name-based; it does not verify numbers or causal claims.
6. **Human (`human.rs`)** — approve / override / reject / defer, interactive (stdio) or scripted. Invalid decisions are refused
   and nothing is written. Decisions survive restarts and feed later runs (confirmed entities are re-seeded as prior evidence,
   which is why `prior_weight=0` is required when using a readout).

## 3. Command line
```
nexcortex [--db DIR] [--event name:sev[:note]]... [--question TEXT]
          [--llm MODEL_DIR | --claude MODEL] [--max-new N] [--operator NAME]
          [--script approve|reject|defer] [--history] [--seed N] [--prior-weight X]
          [--train-readout FILE] [--readout FILE] [--readout-groups all|graph,reservoir,human] [--l2 X]
```
Typical flow: run with `--db` and approve/override over time → `--train-readout r.bin --prior-weight 0` (prints held-out
numbers; warns if it does not beat the untrained rule) → `--readout r.bin --prior-weight 0` for ranking.
Live Claude: `ANTHROPIC_API_KEY=... nexcortex --claude claude-sonnet-5-5`.

## 4. Tests
28 tests, all passing at last run: `tests/pipeline.rs` (16: graph validity, Dale's law/stability, determinism, decision
recording and restart persistence, refusal of bad input, grounding check, spectral clusters), `tests/readout.rs` (8: recovers
a hidden graph preference, learns a human habit, no leakage, deterministic save/load, refuses a different reservoir, too little
data, override pool and persistence, decisions do not perturb features), `tests/claude.rs` (4: grounded reply accepted and
key kept out of body, unverified/ungrounded handling, API-failure fallback, real curl path against a local HTTP server).

## 5. Bugs found and fixed during development
- Confirmed priors made every entity "evidence" on small graphs → `prior_weight=0` with a readout, added to the fingerprint.
- Decision hyperedges leaked labels through diffusion → properties-only decision log.
- Override pool artifact (target appended last) → override pool = entities not shown. Earlier ATT&CK report corrected.
- Naive permutation p-values were invalid under overlapping splits → full-pipeline label-permutation null (24 permutations).
- Several figures in earlier reports were mis-stated and corrected in place.

## 6. Evaluations and what they show
| Dataset / task | Result |
|---|---|
| MITRE ATT&CK (real): given observed techniques, predict others a group uses; group-split, bootstrap CIs | Small consistent gain of the learned readout over the untrained reservoir rule (precision@5 about +0.05; nDCG@10 +0.02 to +0.04). Operator is an oracle built from ground truth, so this is not a human-in-the-loop result. |
| `accounting_data.csv` (tabular anomaly labelling) | Chance level (AUC about 0.49) vs logistic regression 0.65. |
| `ai_accounting_dataset.csv` (8 frauds) | Nothing beats chance (permutation test). |
| Belize land registry (synthetic): portfolio expansion | All proprietors: no signal (learned hit@5 0.062 vs random 0.052, shuffled control 0.054). Companies/government only: learned beat random on 3 of 4 splits, small samples, wide CIs. Untrained reservoir rule at or below random. |
| Claude explainer | Integration verified (tests plus a simulated-API run on 40 ATT&CK briefs: rankings unchanged 40/40, bad outputs flagged, errors fell back). **No live Claude calls were made; no API key was available**, so real grounding rate and latency are unmeasured. |

Pattern: the system gives a modest benefit where the data has relational structure (ATT&CK) and nothing on tabular or random
data, as expected for a graph-diffusion system. The untrained reservoir rule alone is often no better than random.

## 7. Known limitations
- **Scalability:** edge-count blowup fixed (see SCALABILITY_REPORT.md); memory is now ~n^2 in entities, ceiling about 8-9k entities on 7 GB. Sparse storage needed beyond that.
- **No real operator data:** all learned-readout evidence uses a ground-truth oracle operator. Behaviour with real human
  decisions is untested.
- **Readout capacity/data:** a linear shared-weight model; it needs on the order of 100+ labelled decisions and gives no benefit
  without graph or reservoir structure to exploit.
- **Confidence** is a heuristic, not a probability.
- **LLM grounding check** is a name-level heuristic; Claude calls send entity names and notes to a third party.
- Most evaluations use my own task framings; results do not establish performance on a real deployment.

## 8. Repository map
- `src/` — `kb`, `features`, `memory`, `readout`, `brief`, `explain`, `claude`, `human`, `pipeline`, `bin/nexcortex`.
- `examples/` — `attack_live` (live ATT&CK and generic group-completion runner; `TIE_TOPK`, `ATTACK_SPLIT_SALT`),
  `tabular_eval`, `llm_eval` (Claude/simulated explainer evaluation).
- `scripts/` — data preparation and scoring for ATT&CK, the two accounting datasets and the land registry.
- Reports — `ATTACK_LIVE_RUN_REPORT.md`, `ACCOUNTING_DATASET_ACCURACY_REPORT.md`, `AI_ACCOUNTING_FRAUD_ACCURACY_REPORT.md`,
  `CLAUDE_LLM_INTEGRATION_REPORT.md`, `BELIZE_LAND_REGISTRY_ACCURACY_REPORT.md`, `CHANGELOG.md`, `README.md`.

## 9. Overall assessment
A sound, deterministic, leak-checked pipeline with careful safeguards around its LLM stage, and a readout that helps modestly on
relational data. It is not yet validated on real operator decisions, large graphs, or live Claude output. Those are the three
gaps to close before relying on it.
