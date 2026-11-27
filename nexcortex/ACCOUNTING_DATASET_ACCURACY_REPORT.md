# nexcortex on `accounting_data.csv`: accuracy report

Date of run: 2026-11-24. Produced by `examples/tabular_eval.rs`, `scripts/tabular_prepare.py` and `scripts/tabular_score.py`.

## 1. Bottom line

* **On this dataset the system shows no demonstrable predictive accuracy.** Ranking held-out transactions by how likely they are to have failed, the system's AUC is 0.49 (reservoir rule) and 0.48 (learned readout), against 0.50 for random ranking. The best system variant (learned readout restricted to graph features) reaches 0.57, which is within the noise of the test size and was one of four variants tried.
* **A plain logistic regression does better: AUC 0.65**, and it is the best method I tried. That is still a weak classifier. The dataset itself carries little signal about the outcome.
* **"Accuracy" in the plain sense is meaningless here.** 95.1% of transactions have outcome 1, so predicting "1" for everything scores 95.1% accuracy while finding zero failures. I report ranking and balanced measures instead.
* **This is a poor fit for the system.** The data is a flat table with no entities or relationships, and the system is built to reason over a knowledge graph. I had to construct a graph from feature similarity, and that graph barely carries the label (section 6). The result says little about the system on data it was designed for. For that, see the ATT&CK report.
* **I also found and fixed a real flaw in the readout** while running this dataset (section 7).

## 2. The dataset

| Item | Value |
|---|---|
| File | `accounting_data.csv`, 1,000 rows, 18 columns, no missing values, no duplicate rows |
| Target | `Transaction Outcome`: 951 are 1, 49 are 0. The file has no documentation, so I treated the rare class 0 as "failed" and as the thing to find |
| Other columns | account type (4 values), amounts, cash flow, net income, revenue, expenditure, profit margin, debt-to-equity, operating expenses, gross profit, volume (1 to 9), processing time, accuracy score, missing-data flag |
| Dates | one row per day from 2025-01-01 to 2027-09-27 |

Observations that suggest the data is synthetic: dates run daily for nearly three years, one transaction per day; net income does not equal revenue minus expenditure (mean absolute gap 2,126); the column named "Missing Data Indicator" is true for 48 rows even though no value is missing; almost every numeric column looks uniform.

**What signal exists** (all 1,000 rows, outcome 0 as the positive class):

* Missing Data Indicator = True: 8 of 48 rows failed (16.7%) against 41 of 952 (4.3%); Fisher exact p = 0.0016.
* Transaction Volume: AUC 0.63, Mann-Whitney p = 0.002 (higher volume, more failures).
* Profit Margin: AUC 0.59, p = 0.033.
* Everything else: p > 0.10.
* Standard models, 5-fold cross-validation repeated 5 times: logistic regression AUC 0.63 and average precision 0.098 (base rate 0.049); random forest AUC 0.57; gradient boosting AUC 0.60.

So the best achievable accuracy on this data is low, and any method should be judged against that.

## 3. How the system was applied

The system ranks entities given evidence; this data has no entities. I made each transaction a node and connected each to its 10 nearest neighbours in standardised feature space (6,759 ties, one connected component). Labels were not used to build the graph. The task: given 5 transactions known to have failed (the evidence), rank held-out transactions by how likely they are to have failed too.

**Protocol (repeated 5 times).**

* Stratified 70/30 split: 700 training transactions (34 failed) and 300 held-out (15 failed).
* 60 training incidents, each with 5 random known-failed training transactions as evidence. An oracle operator, which reads only training labels, approves the first shown candidate that is a known-failed training transaction, otherwise overrides to a random known-failed training transaction. 33 to 45 of the 60 decisions per split were overrides, so most were overrides. These are logged through the real pipeline and the readout is trained on them.
* Scoring: 20 test incidents per split, each with 5 fresh known-failed training transactions as evidence. The system's ranking is restricted to the 300 held-out transactions and scored against their true outcomes, which were never read before scoring.
* Reservoir of 2,000 units; `top_k` = 10, readout may re-rank all candidates; no LLM stage.
* Baselines are scored on the same 300 held-out transactions in the same splits. Logistic regression and gradient boosting use all 34 training failures and all features. A one-hop "similar to the evidence" rule uses the same graph and the same 5 evidence nodes as the system.

## 4. Results

Mean over the 5 splits (standard deviation across splits in brackets). Base rate of failures in each test set: 0.050 (15 of 300). Precision@10 and precision@30 are the share of the top 10 or 30 that really failed (random expectation 0.05); recall@30 is the share of all 15 failures found in the top 30 (random expectation 0.10); hit@10 is the chance of at least one failure in the top 10.

| Method | AUC | Average precision | Precision@10 | Precision@30 | Recall@30 | Hit@10 |
|---|---|---|---|---|---|---|
| Random (500 permutations) | 0.503 (0.004) | 0.067 | 0.050 | 0.050 | 0.101 | 0.410 |
| Single feature: Transaction Volume | 0.605 (0.067) | 0.102 | 0.060 | 0.073 | 0.147 | 0.400 |
| **Logistic regression (34 train failures)** | **0.652 (0.025)** | **0.156** | **0.160** | **0.133** | **0.267** | **1.000** |
| Gradient boosting | 0.584 (0.019) | 0.074 | 0.040 | 0.073 | 0.147 | 0.400 |
| One-hop similarity to the 5 evidence nodes | 0.523 (0.085) | 0.077 | 0.063 | 0.071 | 0.141 | 0.550 |
| System: diffusion rule | 0.540 (0.032) | 0.079 | 0.071 | 0.085 | 0.169 | 0.610 |
| System: reservoir rule | 0.493 (0.030) | 0.076 | 0.073 | 0.068 | 0.135 | 0.590 |
| System: learned readout, all features | 0.479 (0.054) | 0.081 | 0.104 | 0.064 | 0.128 | 0.670 |
| System: learned, graph features only | 0.573 (0.048) | 0.105 | 0.123 | 0.099 | 0.198 | 0.800 |
| System: learned, reservoir features only | 0.524 (0.021) | 0.079 | 0.083 | 0.070 | 0.139 | 0.590 |
| System: learned, human-history only | 0.481 (0.060) | 0.073 | 0.065 | 0.063 | 0.126 | 0.520 |

AUC per split:

| Method | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Logistic regression | 0.688 | 0.621 | 0.638 | 0.662 | 0.650 |
| Reservoir rule | 0.519 | 0.473 | 0.457 | 0.487 | 0.526 |
| Learned, all features | 0.458 | 0.492 | 0.443 | 0.433 | 0.568 |
| Learned, graph only | 0.607 | 0.526 | 0.514 | 0.600 | 0.616 |

Paired AUC differences across splits: learned (all) minus reservoir rule -0.014 (better in 2 of 5 splits); learned (all) minus logistic regression -0.173 (better in 0 of 5); reservoir rule minus random -0.010.

**Classification view.** Flagging the top 10% of held-out transactions (30 of 300) as "failed", mean over the splits:

| Method | Accuracy | Precision of flags | Recall of failures | Balanced accuracy |
|---|---|---|---|---|
| Everything succeeds (trivial) | 0.950 | n/a | 0.000 | 0.500 |
| Random flagging | 0.860 | 0.052 | 0.099 | 0.502 |
| Logistic regression | 0.877 | 0.133 | 0.267 | 0.588 |
| System: reservoir rule | 0.864 | 0.068 | 0.135 | 0.519 |
| System: learned, all features | 0.863 | 0.064 | 0.128 | 0.515 |
| System: learned, graph only | 0.870 | 0.099 | 0.198 | 0.552 |

Accuracy falls below 95% for every flagging method because flagging 30 transactions as failed necessarily mislabels about 28 good ones. The informative columns are precision and recall: the system's flags are right about 6 to 10% of the time against a 5% base rate.

## 5. Reading the numbers

* **The system as designed (reservoir rule, learned with all features) is indistinguishable from random.** AUCs of 0.48 and 0.49 are below 0.5 by less than their split-to-split spread.
* **The learned graph-only variant looks better (AUC 0.57, precision@10 0.12) but I would not rely on it.** It is one of four ablations I looked at after seeing the results. Each test set has only 15 failures, so a single split's AUC has a standard error of roughly 0.08, and the five splits overlap. It beat 0.5 in all five splits, which is mildly encouraging, not conclusive.
* **Logistic regression beats every system variant in every split**, but it uses 34 labelled failures and every feature directly, while the system sees only 5 failures per incident and reaches the features only through the similarity graph. That is the real comparison here: a graph of similarity is a poor way to use this data.
* **Gradient boosting (0.58) and a random forest (0.57) do worse than logistic regression**, as is typical when the signal is weak and linear.

## 6. Why the system cannot do better here

I checked whether the graph carries the label at all.

* Neighbours of failed transactions fail 8.0% of the time against 5.2% for other transactions (base rate 4.9%). The difference is real (permutation test p = 0.016) but small.
* Best case for any method that spreads evidence through this graph: use *all* 34 training failures as evidence and score held-out nodes by one-hop tie weight. AUC across the five splits is 0.547, 0.468, 0.522, 0.622, 0.524 (mean 0.537). The two-hop version is no better.

So the graph limits any evidence-propagation method to roughly AUC 0.54, which matches what the system achieved. The failures are not clustered in feature space; what predicts them is a few individual features (volume, the missing-data flag, profit margin). A graph of overall similarity dilutes those.

The learned readout's human-history features are also circular on this dataset. The oracle repeatedly picks from the same 34 training failures, so "previously confirmed" simply identifies training failures, and no held-out transaction has any history. These features cannot help on held-out nodes, and that matches the near-zero AUC of the human-only variant (0.48).

## 7. A flaw this dataset exposed, now fixed

On a tiny first run, the learned readout scored 0.75 leave-one-out top-1 against 0.00 for the reservoir rule on 8 decisions. That was too good. Most oracle decisions here were overrides (the operator rejects everything shown and names another entity), and I had built the training pool for an override as "the shown candidates plus the chosen entity". The chosen entity was therefore always ranked below all the others in its pool, and the model could learn "pick the lowest-ranked one", a construction artifact and not a real pattern.

The fix: an override now competes against every entity that was *not* shown. A test covers it, and all 24 tests pass. I re-ran the ATT&CK experiment with the fix, once and without tuning. The gains over the reservoir rule became more consistent (precision@5 up 0.05 to 0.06 in all three splits, all intervals excluding zero), and the ATT&CK report now carries a correction. The results in this report use the fixed code.

## 8. Cost

At 1,000 nodes (2,000 reservoir units): pipeline construction about 7 s, one analysis about 0.25 s, 60 decisions logged in about 17 s, replay for training about 27 s, 20 test incidents across 6 ranking variants about 29 s. Each split takes about 80 s on 2 CPU cores.

## 9. Limitations

1. **Few failures.** 15 failures per test set, 49 overall. All intervals are wide, and I did not apply any multiple-comparison correction across the variants I examined.
2. **Overlapping splits.** The 5 splits resample the same 1,000 rows. Their spread understates the true uncertainty.
3. **One graph construction.** I used 10 nearest neighbours on standardised features, an RBF-style tie weight, and a fixed evidence count of 5. I did not tune any of these. A graph that up-weights volume or the missing-data flag would likely do better, but choosing such weights from labels would need its own train/test discipline.
4. **Oracle decisions, not humans.** As in the ATT&CK run, the operator is derived from the labels.
5. **Unknown target semantics.** I assumed outcome 0 means failure.
6. **Apples and oranges in the baseline comparison.** Logistic regression uses all training labels; the system gets 5 evidence items per incident by design.

## 10. Verdict and recommendations

* **Accuracy of the system on this dataset: effectively chance** (AUC about 0.5, precision of flags about the base rate). I would not use it for failed-transaction detection on this data.
* **What would work better:** a conventional classifier. Logistic regression on all features reaches AUC 0.65, precision@10 of 0.16 (three times the base rate) and recall@30 of 0.27. Expect to find a quarter to a third of failures by reviewing 10% of transactions, not a reliable detector.
* **If you want to use the system on transaction data,** give it a graph with real structure (accounts, counterparties, payment channels, shared devices or merchants) and real labelled analyst decisions. It is built for that, and this table has none of it.
* **The ceiling is the data.** With 49 failures and weak, mostly synthetic-looking features, no method here exceeds AUC of about 0.65.

## 11. Reproduce

```bash
cd nexcortex
python3 scripts/tabular_prepare.py accounting_data.csv ./tab 10 5
cp accounting_data.csv ./tab/
cargo build --release --example tabular_eval
for s in 0 1 2 3 4; do ./target/release/examples/tabular_eval ./tab $s ./tab/out$s.tsv 60 20 5; done
python3 scripts/tabular_score.py ./tab ./tab 5
```
