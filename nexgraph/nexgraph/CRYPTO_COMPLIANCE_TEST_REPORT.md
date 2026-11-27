# nexgraph on the crypto-compliance datasets: test report (v0.24 tested, v0.25 delivered)

**Input:** `crypto_compliance_improved_datasets.tar.gz`: ten cleaned fraud/risk CSVs (343,216 rows, 35 MB) plus the author's `REPORT.md`, `process.py`, `dataset_test.rs` and baseline scores. The files were treated as untrusted data: extracted into their own folder, read but not executed. **System:** nexgraph 0.24 (v0.23 + performance workspace + Belize fixes), release build, directory database (file WAL, fsync off), 2 vCPU / 8 GB sandbox.

## 1. Result in brief
* **All ten files load and query correctly.** 79 of 79 figures returned by nexgraph (counts, group sums and means, quantiles, correlations, histogram bins, robust outliers, time buckets, upsert merges, graph components) match an independent recomputation in pandas/networkx.
* **Four problems found and fixed in v0.25:** high memory at checkpoint and at open, a profiler that labelled counts and flags as street addresses, silent row merging on keyed imports, and an unchecked 4 GiB limit on the checkpoint frame.
* **One problem found and not fixed:** about 2.9 KB of resident memory per node with 14 properties. At that rate the 2.9M-row BitcoinHeist file the dataset author did not import (see section 5) would need roughly 8.6 GB to hold and 12 GB to import. That is an extrapolation from a 742k-row run, not a measurement.
* The datasets are mostly flat feature tables. nexgraph's graph features apply to four of them (wallet/transaction, user/transaction/location, shared Ethereum addresses, daily time series); the rest are exercised as property tables.

## 2. How each file was modelled and loaded

| File | Rows | Modelling | Load |
|---|---|---|---|
| bitcoin_heist_ransomware | 185,605 | one `Heist` node per row (and a second copy keyed on `address`, to test upserts) | 1.25 s |
| bitcoin_scam_messages | 10,000 | `Msg` nodes | 0.18 s |
| bitcoin_daily_network | 4,937 | `Daily` nodes (ISO `date` becomes a timestamp) | 0.04 s |
| crypto_scam_tx | 20,000 | `ScamTx` nodes | 0.26 s |
| etfd_ethereum_fraud | 83,470 | `Etfd` nodes | 0.64 s |
| ethereum_fraud_demo | 1,000 | `EthDemo` nodes | 0.05 s |
| eth_fraud_cleaned + eth_illicit_features | 9,816 + 11,880 | one `Account` node per address, keyed upsert from both files | 0.20 s + 0.28 s |
| blockchain_tx_risk | 9,000 | hyperedge `TxRisk` joining `Transaction` and `Wallet` nodes | 0.22 s |
| digital_payment_fraud | 7,500 | hyperedge `Payment` joining `PayTxn`, `User`, `Location` nodes | 0.26 s |

Total 508,189 nodes and 16,500 hyperedges; 141 MB on disk. All 343,216 rows were read with 0 skipped. Import plus checkpoint took 8.9 s on v0.25 (10.1 s on v0.24).

## 3. What the data shows (from nexgraph queries, all verified)
* **Overlap of the two Ethereum files:** 1,631 addresses are in both; the merged `Account` set has 20,065 nodes (9,816 + 11,880 - 1,631). The fraud label agrees on all 1,631.
* **Heist:** 38,328 ransomware rows in 28 families and 147,277 `white` rows. Largest families: paduaCryptoWall 11,996, montrealCryptoLocker 8,755, princetonCerber 8,386. `income_btc` is very skewed (median 1.89, mean 44.9, maximum 129,708.8); 43,123 rows (23%) exceed the robust z of 3.5, so that screen is not useful here without a log transform.
* **Scam transactions:** 7.25% scams overall; by chain 7.2% (BSC), 7.6% (Ethereum), 6.8% (Polygon), 6.9% (Solana). No chain stands out.
* **Wallet/transaction graph:** 9,000 transactions over 8,593 wallets; 400 wallets appear more than once (at most 3 times); 15.5% of transactions are labelled risky.
* **Payments:** 7,500 payments, 5,106 users (1,827 with more than one), 489 fraud labels (6.5%). The `device_location` column has only 5 distinct values, so those five location nodes become hubs; the largest degree in the combined graph is 3,097. The largest connected component has 12,611 nodes, then 4, 4 (the rest are small).
* **Time series:** `stats.timeseries` accepts the ISO `date` column directly (5,188 daily buckets spanning the date range of 4,937 rows, gaps reported with count 0).
* The author's own caveats stand and were **not** re-tested here: the labels in `blockchain_tx_risk` and `digital_payment_fraud` appear unrelated to the features, and `etfd_ethereum_fraud` is probably synthetic. No model was trained in this run.

## 4. Verification
Recomputed in pandas/networkx from the raw CSVs, compared to nexgraph output (relative tolerance 1e-9, 1e-6 for quantiles and Spearman): row counts and distinct keys (6); the merged-account count and label agreement (3); per-family heist counts, income sums and mean length (18); split counts (6); per-chain scam counts and means (12); per-type message counts and means (12); hyperedge counts and means (6); node counts for wallets and users (2); nine `stats.describe` figures; Pearson and Spearman; the 8-bin histogram; the robust-outlier count; time-series total; daily-network median and mean; graph components. **79 of 79 match.** Not independently checked: PageRank values, `algo.degree` beyond the maximum shown, and `vector.search`.

## 5. Findings about the system, and what v0.25 changed

| # | Finding | Evidence | v0.25 |
|---|---|---|---|
| 1 | Checkpoint needed several copies of the data | peak RSS 4,423 MB loading 508k nodes; 4,310 MB checkpointing 742k heist rows (import peak was 3,063 MB) | encodes from the store one entity at a time, drops the old store before the rebuild, writes the frame without a copy. Load peak 1,849 MB; heist x4 checkpoint adds nothing over the import peak (3,063 MB). Checkpoint files are byte-identical to v0.24's. |
| 2 | Opening a database used about twice the data size | 742k rows: 4,027 MB, 4.9 s | 2,143 MB, 3.0 s (file and decrypted buffers dropped; ops moved into the store) |
| 3 | Profiler: 10 columns classed as High-risk street `Address`; 7 were counts or 0/1 flags (`unique_sent_to_addresses`, `numUniqRecAddress`, `btc_address_present` and others) | `prof.tsv` | street rule needs non-numeric content; the 3 real wallet columns are the new `WalletAddress` kind; 0 `Address` columns remain in these files, the 7 numeric ones are `Number` |
| 4 | Keyed node import merges rows that share a key, visible only as `nodes_updated` | heist `address`: 185,605 rows, 152,908 nodes, 32,697 merged, each overwriting an earlier row's features | `ImportReport::duplicate_keys` (32,697 for this file); the upsert behaviour itself is unchanged |
| 5 | Checkpoint frame length is u32 | code review (`frame`); not hit by this data | refused with an error at 4 GiB |
| 6 | **Not fixed:** about 2.9 KB resident per node with 14 properties, against about 220 bytes per row on disk | 742,420 rows: 2,143 MB after reopen, 161 MB on disk | each node owns a `BTreeMap<String, Value>`; interning keys changes the public `Props` type, so deferred |
| 7 | `import_csv_nodes` is one atomic transaction (import peak 3,063 MB for 742k rows, about 1 GB above the resident size) | scale run | unchanged; `import_csv_hyperedges_with` already has `chunk_rows` |
| 8 | Profiler "false precision" Info finding fires on nearly every float feature column (10 times on `eth_illicit_features`) | `prof.tsv` | unchanged; it is aimed at de-identification and is noise for ML feature tables |
| 9 | `template_id` (10 distinct values over 10,000 rows) is classed as a record identifier because its name ends in `id` | `prof.tsv` | unchanged |

## 6. Scale test (synthetic, derived from the real heist file)
`bitcoin_heist_ransomware.csv` repeated four times with the last three characters of each address changed: 742,420 rows, 14 columns, 78 MB. The raw 2.9M-row file was not supplied. Import 5.2 s, checkpoint 4.1 s (v0.24: 5.3 s), reopen 3.0 s (v0.24: 4.9 s). Peak RSS: v0.24 4,310 MB, v0.25 3,063 MB. Linear extrapolation to 2.97M rows (not run): import about 12 GB, resident about 8.6 GB.

## 7. Not tested
HTTPS server and roles, PostgreSQL, the web UI and the WASM build; `vector.search`; any model training or the author's baseline AUCs; query timings beyond a single cold run each (all analytic statements here took under 1.4 s, the graph components over 508k nodes took 1.3 s).

## 8. Reproducing
`bench/crypto_compliance/` in the delivered system (README inside): `prof`, `load`, `run`, `scale`, `rssopen`, the two NQL files and `cross.py`.
