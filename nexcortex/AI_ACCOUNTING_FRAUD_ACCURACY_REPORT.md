# nexcortex on `ai_accounting_dataset.csv`: accuracy report

Date of run: 2026-11-24. Produced by `examples/tabular_eval.rs`, `scripts/fraud_prepare.py` and `scripts/fraud_score.py`.

## 1. Bottom line

* **No method I tried, including the system, can be distinguished from chance on this dataset.** There are only 8 fraud cases in 500 transactions, and the data shows no detectable fraud signal. Any accuracy figure you see below is within what random labels would produce.
* **The system's best result** (learned readout using reservoir features) has AUC 0.62 on the first 5 splits. Against a null built by randomly re-assigning which 8 transactions are fraud and re-running the whole pipeline 24 times, that is p = 0.12. The untrained reservoir rule scores 0.37, which is not significantly below chance either (p = 0.20 on the low side).
* **No baseline is significant either.** The best, a vendor fraud-rate prior, reaches AUC 0.74 with p = 0.08.
* **Plain accuracy is meaningless.** 98.4% of transactions are not fraud, so "nothing is fraud" scores 98.4% accuracy (98.7% on the held-out sets). The system flagging 10% of transactions scores about 89% accuracy and finds roughly as much fraud as random flagging.
* **With 8 positives, this evaluation can only detect very large effects.** The permutation null has a standard deviation of 0.08 to 0.16 in AUC, so a method would need an AUC of roughly 0.65 to 0.75 to be distinguishable from chance. A modest real effect would be invisible here.
* **I corrected an error of my own along the way:** my first significance test treated the 15 overlapping splits as independent and was far too optimistic. Section 5 explains it, and the final numbers use the correct test.

## 2. The dataset

| Item | Value |
|---|---|
| File | `ai_accounting_dataset.csv`, 500 rows, 10 columns, no missing values, no duplicate rows |
| Target | `is_fraud`: 8 fraud, 492 not (1.6%) |
| Columns | transaction id, date (all 2024), account (4), vendor (8), category (8), amount, currency (all USD), payment method (4), description, `is_fraud` |
| Amount | median 55.30, mean 92.76, max 2,572.75; fraud mean 90.97 against 92.78 for the rest (AUC 0.48, Mann-Whitney p = 0.83) |

**Signal checks.** The 8 frauds are spread across all four accounts and payment methods, six of the eight categories and four of the eight vendors. Six of the 8 frauds sit in two vendors (FoodExpress and Global Tech), but those two were picked after looking at the data. Under random labels the top two vendors hold at least 6 of 8 frauds with probability 0.089, before allowing for the four columns I looked at. None of the other columns shows concentration (p between 0.93 and 0.96).

**Signs the data is synthetic.** The description text ("Payment to X for Y") names the same vendor as the vendor column in only 15% of rows and the same category in only 12%, about what random pairing would give. Every transaction is in USD.

## 3. How the system was applied

The system ranks entities given evidence, so I built a graph with real structure: one entity per transaction plus one per attribute value (8 vendors, 8 categories, 4 accounts, 4 payment methods, 64 descriptions, 5 amount bands, 12 months). Each transaction is tied to its attribute entities (605 entities, 3,500 ties). No labels were used to build the graph.

**Task.** Given 3 known fraudulent transactions as evidence, rank the held-out transactions by how likely they are to be fraudulent too.

**Protocol.**

* 15 random stratified 70/30 splits of the transactions: 350 for training (6 frauds), 150 held-out (2 frauds). Attribute entities are never scored.
* In each split, 40 training incidents with 3 random known-fraud training transactions as evidence. An oracle operator that reads only training labels approves a shown candidate if it is a known training fraud, otherwise overrides to a random known training fraud. In every split all 40 decisions were overrides: the system's top-10 never contained a known training fraud.
* The readout trains on those logged decisions. Scoring uses 10 incidents per split, with the ranking restricted to the 150 held-out transactions and scored against their true labels.
* Baselines use the same training rows and the same test rows. Logistic regression (one-hot attributes plus log amount, balanced weights), naive Bayes on vendor, category, account and payment method with smoothed rates, a vendor fraud-rate prior, and a "shares the most attributes with the evidence frauds" rule that uses the same graph and evidence as the system.

## 4. Descriptive results (15 splits)

Mean over splits, standard deviation across splits in brackets. Each test set has 2 frauds among 150 transactions (base rate 0.013). The random expectations are: AUC 0.50, average precision 0.044, hit in the top 15 (10% of the test set) 0.19, mean percentile rank of a fraud 0.50.

| Method | AUC | Average precision | Hit in top 15 | Mean percentile rank (0 = top) |
|---|---|---|---|---|
| Logistic regression | 0.259 (0.118) | 0.014 | 0.000 | 0.741 |
| Naive Bayes on categories | 0.478 (0.174) | 0.028 | 0.067 | 0.525 |
| Vendor fraud-rate prior | 0.712 (0.080) | 0.038 | 0.133 | 0.294 |
| Shared attributes with the evidence | 0.297 (0.092) | 0.015 | 0.000 | 0.703 |
| System: diffusion rule | 0.328 (0.085) | 0.016 | 0.000 | 0.674 |
| System: reservoir rule | 0.334 (0.080) | 0.016 | 0.000 | 0.667 |
| System: learned, all features | 0.577 (0.101) | 0.045 | 0.207 | 0.428 |
| System: learned, graph only | 0.609 (0.119) | 0.046 | 0.153 | 0.396 |
| System: learned, reservoir only | 0.649 (0.087) | 0.059 | 0.260 | 0.356 |
| System: learned, human-history only | 0.334 (0.080) | 0.016 | 0.000 | 0.667 |

**Do not read these as evidence of skill or anti-skill.** The 15 splits resample the same 8 frauds, so the spread across splits understates the uncertainty. In particular, several methods score well *below* 0.5 (logistic regression 0.26, shared attributes 0.30, reservoir rule 0.33). Section 5 shows that this is also within chance.

Two details are informative. The human-history-only readout is identical to the reservoir rule (AUC 0.334) because no held-out transaction has any decision history, so its features are constant on the test set; the history features are circular here, as in the previous dataset. And the system's learned readout scores above 0.5 while the reservoir rule it is built on scores below 0.5. A plausible reading is that the readout learns to invert an unhelpful ordering that is itself a quirk of these 8 points, not that it detects fraud; the permutation test below shows the learned numbers are not distinguishable from noise either way.

**Classification view.** Flagging the top 10% of held-out transactions (15 of 150) as fraud, mean over the 15 splits:

| Method | Accuracy | Precision of flags | Recall of fraud | Balanced accuracy |
|---|---|---|---|---|
| "Nothing is fraud" (trivial) | 0.987 | n/a | 0.000 | 0.500 |
| Random flagging | about 0.89 | about 0.013 | 0.101 | 0.50 |
| Logistic regression | 0.887 | 0.000 | 0.000 | 0.449 |
| Vendor fraud-rate prior | 0.888 | 0.009 | 0.067 | 0.483 |
| System: reservoir rule | 0.887 | 0.000 | 0.000 | 0.449 |
| System: learned, all features | 0.889 | 0.014 | 0.103 | 0.502 |
| System: learned, reservoir only | 0.891 | 0.020 | 0.153 | 0.527 |

Every flagging method has accuracy near 89%, below the trivial 98.7%, because flagging 15 of 150 transactions necessarily mislabels about 13 good ones.

## 5. Significance: a permutation test of the whole pipeline

My first significance test drew random rankings independently for each split. That is wrong here: the splits overlap heavily and all contain the same 8 frauds, so the average across splits varies far more than that null allows. It gave p-values as low as 0.003, which I discard.

**The valid test.** Re-assign which 8 of the 500 transactions are fraud at random, then re-run the entire pipeline (graph kept, new stratified splits, new oracle decisions, new training, new scoring) and record each method's mean AUC over 5 splits. I did this 24 times and compared the real result, computed on the same first 5 splits.

| Method | Real AUC | Null mean AUC | Null sd | p (real at least this high) | p (real at least this low) |
|---|---|---|---|---|---|
| Logistic regression | 0.296 | 0.486 | 0.137 | 0.88 | 0.16 |
| Naive Bayes on categories | 0.624 | 0.490 | 0.140 | 0.16 | 0.88 |
| Vendor fraud-rate prior | 0.745 | 0.479 | 0.157 | 0.08 | 0.96 |
| Shared attributes with evidence | 0.302 | 0.451 | 0.093 | 0.92 | 0.12 |
| System: diffusion rule | 0.366 | 0.475 | 0.115 | 0.84 | 0.20 |
| System: reservoir rule | 0.370 | 0.473 | 0.105 | 0.84 | 0.20 |
| System: learned, all features | 0.537 | 0.530 | 0.084 | 0.44 | 0.60 |
| System: learned, graph only | 0.550 | 0.542 | 0.112 | 0.44 | 0.60 |
| System: learned, reservoir only | 0.622 | 0.493 | 0.078 | 0.12 | 0.92 |
| System: learned, human-history only | 0.370 | 0.473 | 0.105 | 0.84 | 0.20 |

No method reaches p < 0.05 in either direction. The p-values are from 24 permutations, so the smallest p reachable is about 0.04, and none were close to it. The learned readout's null mean AUC is 0.53 to 0.54, slightly above 0.5, which hints at a small built-in tendency for the readout to land above chance on label noise; this is a point to check with more permutations.

## 6. What this says about the system

* **It cannot find signal that is not there.** With 8 frauds and no association between fraud and any attribute, no model could. The system's results match what random labels produce.
* **It is not shown to be worse than the baselines.** The baselines are not distinguishable from chance either.
* **It does work mechanically on this graph.** 605 entities, about 16 s per split end to end, 0.25 s per analysis.
* **The oracle never approved anything.** The system's top-10 candidates contained no known training fraud in any of the 600 decision incidents, so every training decision was an override. That is a sign the graph ranking carries no fraud information, consistent with the data.

## 7. Limitations

1. **Eight positives.** This is the dominant limitation. Even a perfectly informative method cannot be shown to work with such a sample unless its effect is very large.
2. **The permutation test uses 24 permutations and 5 splits each,** so p-values are coarse (steps of about 0.04).
3. **The graph is untuned.** One construction, with all ties weight 1. Different weights might help, but tuning them against 8 labels would overfit.
4. **Oracle decisions, not analysts.**
5. **I used the real labels only for scoring and for training-set evidence;** the graph and the splits' structure do not depend on test labels.

## 8. Verdict and recommendations

* **Accuracy of the system on this dataset: not distinguishable from chance. No claim of fraud-detection skill is supported,** and the same is true of every baseline.
* **To evaluate any fraud detector you need far more fraud cases.** As a rough guide, you would want at least a few hundred positives, or a much stronger effect, before the evaluation can separate a working method from noise.
* **If real fraud in your data is tied to structure** such as shared accounts, repeated counterparties or devices, the system is built to use it, but this file has none of that: every attribute is shared by dozens of transactions and none of them concentrates fraud.
* **Treat any single-split accuracy on a dataset this size with suspicion,** including the 15-split table above.

## 9. Reproduce

```bash
cd nexcortex
python3 scripts/fraud_prepare.py ai_accounting_dataset.csv ./fraud 15
cargo build --release --example tabular_eval
for s in $(seq 0 14); do ./target/release/examples/tabular_eval ./fraud $s ./fraud/out$s.tsv 40 10 3; done
python3 scripts/fraud_score.py ./fraud ./fraud ai_accounting_dataset.csv 15
```

The permutation test repeats the same steps on a copy of the CSV with the 8 fraud labels moved to randomly chosen rows, using 5 splits each; `fraud_score.py` accepts the number of simulations and a quiet flag for this.
