# nexcortex

The architecture, implemented as one Rust crate that wires your two archives together:

```
NexGraph (persistent knowledge)            src/kb.rs        nexgraph v0.22
        |
        v
Spectral / diffusion features              src/features.rs  CALL physics.heat_from, spectral.embedding/cluster/eigenvalues, algo.pagerank
        |
        v
Connectome RNN (working memory + reasoning) src/memory.rs   human_connectome_rnn (hcrnn) built from the knowledge graph itself
        |
        v
LLM explanation layer                      src/explain.rs   nexgen-llm 0.4 (Burn 0.21), with grounding check + deterministic fallback
        |
        v
Human decision maker                       src/human.rs     approve / reject / defer / override; recorded back into NexGraph
```
`src/pipeline.rs` runs it; `src/bin/nexcortex.rs` is the CLI. Layout expected on disk (this is how the archive is packed):
`nexcortex/`, `nexgraph/{nexgraph,human_connectome_rnn}/`, `nexgen_llm/nexgen_llm/` side by side.

## What each stage does here

1. **Knowledge** — entities are `:Entity` nodes, ties are weighted 2-member hyperedges, groups are n-member hyperedges. It is a normal durable NexGraph database (`--db DIR`), so it survives restarts and is queryable with NQL.
2. **Features** — for each evidence item, heat diffusion `exp(-t(I-P))` from that entity at three scales (0.5 / 2 / 8), mixed 0.5 / 0.35 / 0.15, times severity. Also a 3-D spectral embedding, spectral clusters, the bottom of the Laplacian spectrum and PageRank. Spectral stages require a **connected** graph; the pipeline refuses otherwise.
3. **Working memory** — the knowledge graph becomes a `Connectome` (stronger tie = higher weight and shorter fibre = shorter conduction delay) and the `hcrnn` network is built on it: Dale's-law E/I units per entity, conduction delays, leaky tanh dynamics, spectral radius calibrated < 1. Evidence is injected as a time series of graph signals (one input channel per entity, identity input matrix), pulse by pulse; the network then runs silently and the **post-evidence window** (12 steps after the last pulse) is read out as each entity's salience.
4. **Explanation** — `TemplateExplainer` (deterministic) or `LlmExplainer` (any Llama/Mistral/Qwen2/Qwen3 safetensors directory). The model sees only the fact sheet; its output must name the top candidate and mention no entity outside the brief, otherwise it is shown as *unverified* behind the deterministic text. Prompt too long, injection-guard refusal or generation error -> template with a warning. The explanation never changes ranking or options.
5. **Human** — the pipeline renders facts + explanation + caveats and asks. It never executes anything. Approve must name an offered candidate; override must name an existing entity; closed stdin is an error, not an approval. Every answer is stored as a `:Decision` node (operator, note, confidence, plus the evidence shown, the entity chosen and the machine pick refused; since the readout, logged decisions are *not* wired into the graph with a `DECIDED_ON` edge, see below) and bumps `confirmed` / `rejected` counters. Confirmed entities come back in later runs as weak prior evidence (severity `0.3 * min(count,3)/3`).

## Run

```bash
cd nexcortex
cargo build --release                       # first build ~4 min (Burn)
./target/release/nexcortex                  # interactive, in-memory demo network
./target/release/nexcortex --db ./kb --operator ana --event web-01:0.9:"WAF alert" --event db-03:0.7
./target/release/nexcortex --db ./kb --history
./target/release/nexcortex --llm /path/to/Qwen3-0.6B --max-new 160
./target/release/nexcortex --script approve # non-interactive stand-in for the human (tests/demos), audited as "scripted"
cargo test --release                        # 16 pipeline tests + 8 readout tests
```
Library use: `Pipeline::new(kb, PipelineConfig::default())?`, `pipe.run(&query, &explainer, &mut human)?`; build your own graph with `KbBuilder`.

## Learned readout (`src/readout.rs`)

```
graph features (11)  +  reservoir trajectory (11)  +  human decisions (4)   ->  w . phi  ->  softmax over candidates
```
* **Graph:** PageRank, closeness of the spectral embedding to the evidence, same-cluster share, heat at short/mid/long scale, diffusion, degree, strongest-path and hop proximity, direct tie to the evidence.
* **Reservoir trajectory:** the entity's excitatory rate in five post-evidence time bins, at input end, its peak and peak time, its neighbours' post-window activity, and its salience rank. Features are divided by their maximum over *all* candidates.
* **Human decisions:** how often the entity was confirmed / refused before, and a precedent score (cosine similarity of the current evidence to earlier evidence where it was chosen / refused).
* One weight vector shared by all entities (26 parameters, independent of graph size), softmax cross-entropy + L2, solved exactly by damped Newton: deterministic, no learning rate or seed.
* Labels are `approve` / `override`. `reject` / `defer` are not supervised examples. A training pool is the tier the operator chose from: the shown candidates for an approve; for an override, every entity that was *not* shown (they refused all shown ones). An earlier version appended the override target to the shown set, which made it the one candidate always ranked below the rest, a construction artifact found when most decisions were overrides.
* Examples are rebuilt by **replaying** the decision log in order; each sees only the history before it (`examples_do_not_leak_later_decisions`).

```bash
./nexcortex --db ./kb --prior-weight 0 --train-readout kb.readout          # needs >= 8 logged, labelled decisions
./nexcortex --db ./kb --prior-weight 0 --readout kb.readout --event web-01:0.9
./nexcortex --db ./kb --train-readout x --readout-groups graph,reservoir    # ablate a source
```
`--train-readout` prints leave-one-out and forward-chained top-1 / MRR next to the untrained reservoir and diffusion rules on identical pools, and warns when the readout does not beat the reservoir rule. Library: `readout::train(&pipe, &cfg)` then `pipe.set_readout(model)`. A readout is bound to its reservoir by a fingerprint (config, entity names, adjacency, `top_k`, `prior_weight`) and refused otherwise.

**What this does and does not show.** Tests use *simulated* operators with a hidden preference (hub-seeking: 0.37 -> ~0.95 top-1 vs the reservoir rule; a name-level habit that only the human features can see). That proves the machinery recovers a learnable rule and does not leak labels. **No real operator data was available**, so there is no evidence yet that it improves real decisions; with few decisions 26 weights over-fit, which is why the held-out numbers are printed rather than trained-on numbers. Decisions logged before this change carry no evidence and are skipped. At inference the readout re-ranks the reservoir's best `rerank_pool` (10) candidates, wider than the 5 shown in training, a mild extrapolation. Learned scores are relative, not calibrated.

**Two behaviours changed.** (1) Logged decisions no longer add a `DECIDED_ON` hyperedge: NexGraph diffuses over every hyperedge, so those edges changed the heat features and leaked each decision's label into its own features (the leak test caught this). Chosen/refused entities are stored as `:Decision` properties instead. (2) Use `--prior-weight 0` with a readout: confirmed entities are otherwise re-injected as evidence *and removed from the candidates*, and on a small graph every entity eventually is, leaving nothing to rank ("every entity is evidence"); this was already true of the pipeline before.

## Findings from building it (read these)

* **A first readout was wrong and was replaced.** Reading the network's *end-of-run* state ranked the same hub (`ad-01`) first for DMZ-only, data-only and corp-only evidence: after the transient dies out only the connectome's dominant mode remains. The readout is now the post-evidence window, and `ranking_depends_on_the_evidence` is a regression test for exactly this.
* **The memory is fading and recency-weighted.** Half-life is ~6 steps and only ~10-15% of peak activity remains after the 12-step readout window. Later evidence counts more than earlier evidence. The pipeline replays the evidence in reverse order and, if the top candidate changes, flags it and cuts confidence by 20%. In the demo scenario it does flag.
* **The recurrent stage changes the answer relative to plain diffusion** (demo: diffusion favours `lb-01`/`waf-01`/`web-03`; the network favours the data zone `db-01`/`vault-01`/`backup-01`, pulled by the latest alert). **I have no ground truth that this is better.** It is a different, order-aware read of the same evidence, shown next to the diffusion rank so the operator can judge.
* **The reservoir is frozen; only the readout can learn** (next section). `hcrnn`'s own trainable task has 3 inputs / 2 outputs and does not apply.
* **Confidence is a heuristic** (rank margin 40%, diffusion agreement 35%, retention 25%, x0.8 if order-unstable). It is not calibrated.
* **Demo data is synthetic** (22-entity network). Demo candidates in one zone are near-tied by symmetry, and the brief says so.
* Within-zone near-ties are real, not a bug: a 4-clique of equal ties gives its members equal salience.

## Verified here (rustc 1.97, Linux, CPU)

* `cargo build --release` clean, no warnings; **16 integration tests pass**: KB validation, disconnected-graph refusal, diffusion monotonicity, Dale's-law/stability check, evidence dependence, determinism, decision validation (nothing written on refusal), persistence across restart, prior-seed feedback, stdin parsing, grounding check, zone recovery by spectral clustering (3 near-zero eigenvalues), and two LLM tests on `nexgen_llm`'s tiny fixture model (oversized prompt -> template fallback; real generation -> ungrounded text correctly marked unverified).
* CLI run interactively through stdin (invalid input re-prompted, decision recorded) and across two processes on a durable `--db`.

## Not verified / limits

* **No real LLM checkpoint was run** (none available offline). The LLM stage was exercised only with a 48-token-vocabulary random-weight fixture, which proves plumbing, guard and fallback, not explanation quality. With a real model, check that explanations stay inside the facts; the grounding check is a name check, not a faithfulness proof.
* The knowledge graph is dense in memory for the RNN (n x n adjacency, 2n units): fine to a few thousand entities, not millions. NexGraph's own spectral solver is dense up to ~1500 vertices (Lanczos above).
* `GROUP` hyperedges influence diffusion and spectral features (NexGraph includes every hyperedge) but not the RNN wiring or the explanation paths, which use `REL` ties only.
* Severity is user-supplied; the system does not extract evidence from logs.

## Change made to your code

`nexgraph/human_connectome_rnn/crates/spectral_hypergraph/Cargo.toml`: `indexmap = "=2.2.6"` -> `"2.2.6"`. The exact pin (a Burn-0.13-era leftover; `indexmap` is not used in any source file) made it impossible to resolve together with Burn 0.21, which needs `indexmap >= 2.11`. No source file in any of your crates was modified.
