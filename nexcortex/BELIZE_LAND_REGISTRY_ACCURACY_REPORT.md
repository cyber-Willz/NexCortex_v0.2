# nexcortex on the Belize land-registry dataset — accuracy analysis

## Verdict
**On this dataset the system has little to find, and it mostly doesn't.** The dataset is *synthetic* (its README says so),
and the generator source shows its disputes, mortgages and acquisitions are drawn at random, so the only learnable structure
is in who owns land where. On that task:
- **All proprietors (4,834 portfolios):** no real signal. Learned readout hit@5 0.062 vs random 0.052 vs shuffled-label control 0.054; the CIs overlap.
- **Companies/government only (79 portfolios):** a small, positive but unstable edge over random. The learned readout beat random on 3 of 4 splits (precision@5 0.027 / 0.033 / 0.013, random 0.010); on the fourth it was indistinguishable.
- **The untrained pipeline default (reservoir rule)** scored at or below random everywhere (precision@5 0.000–0.007 on the clean splits).
- **Scalability limit found:** 92,151 tie edges made the pipeline try to allocate 68 GB and crash. I sparsified to the top 12 ties per entity (about 3.7–4.8k edges) to run at all.

These are weak results on an artificial task I chose; they are not an assessment of the system on a real registry.

## What I tested, and why this task
The archive holds registry tables (75k parcels, 175k ownership entries, 154k dealings, 2.2k disputes, 54 acquisitions). The system
ranks *other entities* given observed entities, so I used the one task with real structure to recover:
**portfolio expansion.** Entity = a "block cluster" (10 consecutive blocks within a registration section; 435 entities).
Group = a proprietor; its set = every cluster the party has ever held land in. Given a proprietor's first 3–12 clusters, rank the
rest. Same protocol as the ATT&CK run: proprietors split train/held-out by name hash; graph built from train proprietors only;
oracle operator logs decisions on train incidents; readout trained on those logs; held-out never recorded; macro-average over
groups with bootstrap CIs; exact-expectation random baseline.

Not tested, and why:
- **Dispute prediction** (the obvious registry task): `generate_dataset.py` picks disputed parcels uniformly at random among eligible
  parcels (non-government, transactable tenure). There is nothing to learn beyond eligibility. I read this in the source; I did not run it.
- **Late compensation on acquisitions:** one flagged case out of 54, so not measurable.

## Results (held-out proprietors; 95% bootstrap CI over groups)
**A. All proprietors — 120 held-out groups, 240 incidents**

| Method | hit@5 | precision@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| random (exact) | 0.052 | 0.011 | 0.050 | 0.017 |
| popularity prior | 0.033 | 0.008 | 0.044 | 0.015 |
| reservoir rule (default) | 0.033 | 0.007 | 0.036 | 0.015 |
| learned readout (all features) | 0.062 [0.033, 0.096] | 0.013 [0.007, 0.020] | 0.059 | 0.021 |
| control: shuffled labels | 0.054 | 0.012 | 0.042 | 0.016 |

Learned vs reservoir rule: precision@5 +0.007 [0.000, +0.013]; MRR +0.023 [+0.001, +0.047]. Learned vs popularity: all CIs include 0.
The shuffled-label control scores the same as the learned model on hit@5 and precision@5, so I do not credit the readout with signal here.
Per group: 12 better, 4 worse, 104 equal.

**B. Companies, government, community — four different train/test splits (each 16–26 held-out groups)**

| Split | random p@5 | reservoir p@5 | learned p@5 | shuffled control p@5 | learned − reservoir hit@5 |
|---|---|---|---|---|---|
| default | 0.010 | 0.000 | 0.027 | 0.008 | +0.135 [+0.058, +0.231] |
| salt a | 0.010 | 0.000 | 0.033 | 0.000 | +0.167 [+0.056, +0.278] |
| salt b | 0.010 | 0.000 | 0.013 | 0.004 | +0.065 [0.000, +0.152] |
| salt c | 0.069* | 0.069 | 0.075 | 0.050 | +0.031 [−0.062, +0.125] |

\*One very large holder in that test set (a government-scale portfolio) inflates the random baseline; there is nothing to separate there.
On the default split, learned hit@5 was 0.135 vs random 0.049 and vs popularity +0.115 [+0.019, +0.212]. Splits overlap, so I did not pool them into one CI.
The absolute levels are low: even the best case finds a true next cluster in the top 5 for about 1 in 7 incidents.

## Reading the results
- The graph-only readout was the weakest learned variant in A (hit@5 0.008) and mid-pack in B. The edge in B comes mostly from the reservoir and "human" feature
  groups (the oracle operator's habit of overriding to the most popular cluster), i.e. closer to a learned popularity prior than to structure found by diffusion.
  I did not isolate why; this is my reading, not a tested explanation.
- The untrained reservoir rule being at or below random is a real finding on this graph: salience ranking does not help when ties are
  near-uniform co-holding with no community structure.
- The synthetic generator draws individuals' purchases from a shared pool, so portfolios are mostly noise. Real registries may behave differently; I can't tell from this data.

## Caveats
1. **Oracle operator**: no real operator logs exist, so the "human decisions" are generated from ground truth (the same caveat as the ATT&CK report).
2. **Sparsified graph** (top-12 ties per entity) was needed to avoid the memory failure; results are for that graph, not the dense one. Another setting was not tried.
3. **Evidence choice**: evidence is each portfolio's lowest-indexed clusters, hidden = the rest. It is not temporal (not "earlier purchases predict later ones").
4. **Small test sets in B** (16–26 groups); CIs are wide and 4 splits is not a systematic robustness study.
5. Synthetic data only; the dataset's own README says its parameters are illustrative.

## Reproduce
```
python3 -I scripts/landreg_prepare.py <belize_land_registry_dir> <out_dir> [all|company]
TIE_TOPK=12 cargo run --release --example attack_live -- <out_dir> 120      # [ATTACK_SPLIT_SALT=a for other splits]
```
`TIE_TOPK` is a new optional env var in `examples/attack_live.rs` (default: off).
