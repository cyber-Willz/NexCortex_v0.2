# nexcortex on live MITRE ATT&CK data: end-to-end run and analysis

Date of run: 2026-11-24. Everything below was produced by `examples/attack_live.rs` on data fetched live from GitHub during the run.

> ## Correction (2026-11-24): numbers below are superseded
> After this report was written, running the readout on a second dataset exposed a flaw in how training pools were built for `override` decisions (the override target was appended to the shown candidates, so it was always the lowest-ranked one in its pool, a construction artifact). It is fixed (an override now competes against every entity that was *not* shown) and covered by a test. The effect was small here (only 6 to 8 of ~176 decisions were overrides), but I re-ran all three splits with the fixed code, once, without tuning. Updated paired differences, learned (all) minus reservoir rule, mean of the three splits, with the number of splits whose 95% interval excludes zero:
>
> | Metric | split 0 | split 1 | split 2 | splits significant |
> |---|---|---|---|---|
> | precision@5 | +0.049 | +0.058 | +0.049 | 3 of 3 |
> | nDCG@10 | +0.029 | +0.018 | +0.037 | 3 of 3 |
> | hit@5 | +0.035 | +0.025 | +0.023 | 0 of 3 (all lower bounds at 0.000) |
> | MRR | -0.001 | +0.063 | +0.044 | 1 of 3 |
>
> Versus the popularity prior, precision@5 is +0.021, +0.125, +0.107 (2 of 3 significant) and MRR +0.076, +0.123, +0.069 (2 of 3). So the conclusion in sections 1 and 6.3 should be read more favourably than written: after the fix the learned readout gives a small but consistent gain in precision@5 and nDCG@10 over the untrained reservoir rule, still on 40 to 43 held-out groups with the oracle-operator caveat unchanged. The tables in sections 4 and 5 show the pre-fix numbers.

## 1. Bottom line

* The full pipeline (knowledge graph, spectral and diffusion features, connectome reservoir, learned readout, brief, explanation, decision log) runs end to end on real data. Total wall time is about 3 minutes on 2 CPU cores for 454 reservoir units.
* **On this task the system is useful but not dramatically better than a popularity count.** The hard-to-beat baseline is "rank techniques by how many known groups use them". The untrained reservoir rule roughly matches it. The learned readout adds a small, consistent-in-sign improvement that is mostly inside the statistical noise of 40 to 43 held-out groups.
* **Learned readout vs the untrained reservoir rule (three group splits):** precision@5 +0.047, +0.032, +0.016; MRR +0.025, +0.055, +0.029; hit@5 +0.058, -0.013, -0.012. Only 3 of the 15 paired comparisons have a 95% interval that excludes zero.
* **Learned readout vs the popularity baseline:** better on precision@5 and MRR in all three splits, with intervals excluding zero in 2 of 3 for each metric. This is the most defensible positive result.
* **The heat-diffusion rule alone is useless here** (hit@5 of 0.02 against 0.34 for random).
* **Caveat that matters most:** the "human decisions" are an oracle derived from the real ground truth, not real operators. This run tests whether the readout can learn the real task from decision logs. It does not test whether real analysts would be helped.

## 2. Data

| Item | Value |
|---|---|
| Source | `mitre-attack/attack-stix-data`, `enterprise-attack/enterprise-attack.json` on GitHub, fetched live by `scripts/attack_prepare.py` |
| Version | ATT&CK Enterprise v20.1 |
| SHA-256 | `539a843a7b53ed9749740b2123dddae3726b1a4831841c32aca740b011b910b5` (55.3 MB) |
| Used | non-revoked, non-deprecated objects only |
| Entities | 227 parent techniques (sub-techniques rolled up to their parent); kind = first tactic in matrix order |
| Ground truth | 174 intrusion groups with a documented technique set (direct `uses` relationships only; software links are not followed) |
| Eligible for incidents | groups with at least 8 documented techniques (131 of 174) |

## 3. Task and protocol

**Task.** An adversary has been observed using a few techniques, in kill-chain order. Which other techniques is it documented to use, that is, what should be hunted for next? Success is a hidden, documented technique appearing in the top 5 or 10.

**Incident construction.** For each eligible group, the techniques are sorted by kill-chain order. Two incidents are made per group, revealing 35% and 60% of the techniques as evidence (clamped to 3 to 12, severity 0.8 each). The remaining techniques are the hidden ground truth. At least 3 must remain hidden.

**No-leakage design.**

* Groups are split 70/30 by a hash of the group name. The main run uses 119 training and 55 held-out groups.
* The knowledge graph is built from training groups only. Tie weight is the cosine of co-use between two techniques over training groups, plus 0.05 if they share a tactic. The graph has 227 entities and 12,374 ties, one connected component, and no bridging ties were needed.
* An oracle operator acts on training incidents through the real pipeline: it approves the first shown candidate the group really uses, otherwise it overrides to the most popular technique the group really uses. These decisions are logged with the real `Pipeline::record`.
* The readout is trained only on those logged decisions, replayed in order, so each example sees only earlier history.
* Test incidents are never recorded and never touch the graph.

**Configuration.** `top_k` = 10, `rerank_pool` = 40, `prior_weight` = 0 (see the README for why), readout L2 = 1.0, template explainer. The LLM stage was not run because no checkpoint was available.

**Metrics.** hit@k (at least one hidden technique in the top k), precision@5, MRR of the first hit, and nDCG@10. Scores are averaged per group (macro) with 95% bootstrap intervals over groups (2,000 resamples). The random baseline is computed exactly.

## 4. Main result (split 0: 43 held-out groups, 86 incidents)

| Method | hit@5 | precision@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| Random (exact expectation) | 0.341 | 0.090 | 0.220 | 0.094 |
| Popularity prior (training groups) | 0.895 | 0.474 | 0.625 | 0.473 |
| Diffusion rule | 0.023 | 0.005 | 0.061 | 0.007 |
| Reservoir rule (pipeline default) | 0.884 | 0.447 | 0.702 | 0.462 |
| **Learned readout, all features** | **0.942** | **0.493** | **0.727** | **0.497** |
| Learned, graph only | 0.930 | 0.498 | 0.701 | 0.491 |
| Learned, reservoir only | 0.907 | 0.486 | 0.704 | 0.490 |
| Learned, human only | 0.895 | 0.453 | 0.690 | 0.490 |
| Control: all features, shuffled labels | 0.756 | 0.372 | 0.513 | 0.391 |

Paired differences, learned (all) minus baseline, with 95% intervals over groups:

| Metric | vs reservoir rule | vs popularity prior |
|---|---|---|
| hit@5 | +0.058 [+0.012, +0.116] | +0.047 [-0.012, +0.128] |
| precision@5 | +0.047 [-0.007, +0.100] | +0.019 [-0.044, +0.079] |
| MRR | +0.025 [-0.031, +0.081] | +0.102 [+0.039, +0.172] |
| nDCG@10 | +0.035 [+0.004, +0.068] | +0.024 [-0.015, +0.064] |

hit@10 is 0.977 for the learned readout and the popularity prior, and 0.953 for the reservoir rule, so everything sensible finds at least one hit in the top 10.

By amount of evidence, precision@5 goes from the reservoir rule to the learned readout as follows: 3 to 5 techniques observed (15 incidents) 0.267 to 0.347; 6 to 8 (20 incidents) 0.220 to 0.340; 9 to 12 (51 incidents) 0.588 to 0.596. Per group, the learned readout is better on 18, worse on 7, and equal on 18.

## 5. Replication on two further group splits

These repeat the whole protocol (graph, decisions, training, evaluation) with a different hash salt. The splits draw from the same 174 groups, so they are **not independent**; they show sensitivity to the split, not three independent confirmations.

| Split | Held-out groups | Comparison | hit@5 | precision@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| 0 | 43 | learned minus reservoir | +0.058* | +0.047 | +0.025 | +0.035* |
| 1 | 40 | learned minus reservoir | -0.013 | +0.032 | +0.055* | +0.009 |
| 2 | 43 | learned minus reservoir | -0.012 | +0.016 | +0.029 | +0.013 |
| 0 | 43 | learned minus popularity | +0.047 | +0.019 | +0.102* | +0.024 |
| 1 | 40 | learned minus popularity | +0.062 | +0.100* | +0.114* | +0.055 |
| 2 | 43 | learned minus popularity | +0.035 | +0.074* | +0.054 | +0.030 |

`*` means the 95% bootstrap interval excludes zero. Untrained reservoir rule against popularity, hit@5: 0.884 vs 0.895 (split 0), 0.963 vs 0.887 (split 1), 0.930 vs 0.884 (split 2).

On the replication splits, the shuffled-label control stays far below everything else: hit@5 of 0.525 and 0.651 against 0.95 and 0.92 for the learned model.

## 6. Analysis

**6.1 The task is dominated by popularity.** A few techniques (command and scripting interpreters, ingress tool transfer, masquerading, and similar) appear in most groups. Ranking by popularity alone gets a hit in the top 5 for about 89% of held-out groups, against 34% for random. Any method has to be read against that bar, not against random. The reservoir rule clears it narrowly in two of three splits and ties it in the third. Its edge comes mainly from ordering (MRR 0.70 vs 0.63 in split 0), not from finding hits that popularity misses.

**6.2 The reservoir already carries the co-use structure.** The untrained reservoir rule, driven by evidence diffused through a graph of co-use ties, reproduces roughly popularity-level quality with no training. That is a reasonable result for a frozen network: the working-memory state encodes which techniques co-occur with the evidence. It is also why the learned readout has little headroom on hit@5.

**6.3 The learned readout helps a little, mainly on ordering and on small-evidence incidents.** Gains are most consistent in precision@5 (positive in all three splits) and MRR (positive in all three), and larger when only 3 to 8 techniques are observed (about +0.08 to +0.12 precision@5 in split 0). With 9 to 12 observed techniques the reservoir is already near ceiling. Gains in hit@5 are not consistent across splits. Taken together this is weak, directionally positive evidence, not a demonstrated improvement.

**6.4 The label-shuffled control is a useful sanity check.** The same model trained on shuffled labels falls well below the untrained rule (hit@5 0.756 vs 0.884 in split 0). So the readout is genuinely using the labels, and a learned reordering is not automatically better: a model fit to noise actively hurts.

**6.5 Feature-group ablations are not reliable at this sample size.** In split 0 the graph-only readout is close to the full one. In split 1 graph-only is best on hit@5, and in split 2 human-only has the best MRR (0.715). The ranking of ablations changes from split to split and the intervals overlap heavily, so no claim about which source matters most is supported. In the full model's weights, reservoir features carry the most total weight (6.37 vs 3.48 graph and 1.72 human), led by `direct_tie` and several reservoir time-bin features. Weight magnitude is not the same as held-out usefulness.

**6.6 The logged-decision evaluation is less favorable than the held-out one.** On the 176 logged decisions (pool = the 10 shown), leave-one-out top-1 is 0.59 for the learned readout against 0.60 for the reservoir rule in split 0. The CLI's own check (`--train-readout`) would therefore print its "does not beat the reservoir rule" warning for split 0. In splits 1 and 2 the learned readout is ahead (0.46 vs 0.38 and 0.54 vs 0.51). This is a within-pool metric on a different quantity from the held-out ranking metrics, which is why it can disagree. It is also a reminder that the readout is a marginal gain here.

**6.7 Diffusion alone fails, for a reason not yet isolated.** The heat-diffusion ranking has hit@5 of 0.02, below random. A plausible explanation is that heat concentrates on techniques in the same tactic as the evidence (through the weak same-tactic ties) and on the evidence's closest neighbours, whereas the hidden techniques are mostly from later tactics. This is a hypothesis; I did not test it, for example by removing the tactic ties. The reservoir stage, fed the same diffusion fields, does much better, so the recurrent stage is doing real work relative to one-shot diffusion on this graph.

**6.8 Example brief.** The sample brief (held-out group APT-C-36, 9 observed techniques) ranks T1059 first with learned score 0.81, and the template explanation correctly states the rank, path and that plain diffusion ranks it 78th. Ground truth: 3 of the 10 shown candidates (T1059, T1105, T1036) are documented for that group. The system flagged the diffusion/reservoir disagreement as a caveat, which is the right behavior. The reported confidence of 0.72 is the existing heuristic and is not calibrated.

**6.9 Cost.** Pipeline construction takes 3.6 s (spectral features and connectome). Each analysis takes about 0.15 s. Logging 176 decisions took 27 s, replaying them for training took 29 s, and evaluating 86 test incidents across 7 ranking variants took 93 s. The dense n by n design is comfortable at 227 entities; the README's limit of a few thousand entities still applies.

## 7. Limitations and threats to validity

1. **Oracle, not humans.** The decisions come from ground truth that the operator is given. They teach the readout the real task (which techniques a group uses), but real analysts would be noisier and have different preferences. Nothing here measures effect on real analysts.
2. **Small test sets.** 40 to 43 held-out groups per split; intervals are wide, and I report many comparisons without correction. Treat starred results as suggestive.
3. **Splits overlap.** All three draw from the same 174 groups.
4. **Reporting bias in ATT&CK.** The ground truth is what has been publicly reported and mapped, not what the adversaries actually did. A "hidden" technique that is not documented counts as a miss.
5. **Incident construction.** Evidence is always the earliest techniques in kill-chain order, with constant severity 0.8. Real alert streams are not ordered or labelled this way.
6. **Graph built from the same kind of data it is tested on.** Co-use ties come from training groups, which makes the graph strongly popularity-shaped. That is legitimate (no test leakage) but it explains why popularity is so competitive.
7. **Direct group-technique links only.** Techniques reached through malware or tool links are not included, so groups look smaller than their real behaviour.
8. **Not run:** the LLM explanation stage (no checkpoint), multi-operator or noisy-label settings, and any larger knowledge base (sub-techniques, campaigns, software).

## 8. What this suggests

* The architecture works on real data and is fast enough to iterate. The frozen reservoir is a competitive, training-free ranker on this task.
* Treat the learned readout as an optional refinement. Use `--train-readout` and trust its held-out printout; when it warns, do not deploy.
* To learn whether it helps, collect real analyst decisions (or at least noisy ones) and repeat this protocol. The pipeline already logs what is needed.
* Worthwhile next experiments, in order of likely value: (a) add technique kind (tactic) and a popularity feature to the readout and see whether the learned model then clearly beats the popularity baseline; (b) test the same-tactic-tie hypothesis for diffusion by building the graph from co-use ties only; (c) add sub-techniques and software links; (d) use many more random splits or repeated cross-validation over groups to tighten the intervals.

## 9. Reproduce

```bash
cd nexcortex
python3 scripts/attack_prepare.py ./attack_data               # live download + conversion; prints version and SHA-256
cargo build --release --example attack_live
./target/release/examples/attack_live ./attack_data           # split 0
ATTACK_SPLIT_SALT=s1 ./target/release/examples/attack_live ./attack_data   # split 1 (s2 for split 2)
```

Run-to-run output is deterministic given the same data file and splits (no random numbers are drawn in training; bootstrap uses a fixed seed).

Sources: [MITRE ATT&CK STIX data on GitHub](https://github.com/mitre-attack/attack-stix-data), [ATT&CK Enterprise matrix](https://attack.mitre.org/matrices/enterprise/).
