# nexgraph v0.18: GAAP ledger, bulk loading, ratios and aging

Prompted by running the v0.17 ledger on a public transaction-level dataset (Superstore Sales, 8,399 orders loaded from GitHub; 25,198 entries; net income reconciled to the
file's Profit column to the cent; see `examples/gaap_live.rs`). That run showed what a real load needs and what the ledger could not yet answer.

## Added
* **`Ledger::post_many`**: atomic batch posting (all or none, gapless numbers, error names the failing entry). The live load had no way to be all-or-nothing before.
* **`Ledger::import_journal_csv`** (feature `ingest`): journal-per-CSV loader built on the crate's RFC-4180 parser (handles bare `\r`, CRLF, LF, BOM, quoted fields). Atomic; errors name entry and line.
* **`gaap.ratios(company, from, to)`**: 17 profitability, liquidity, solvency and efficiency ratios; averages of opening and ending balances; `NULL` when undefined.
* **`gaap.open_items` / `gaap.aging`**: unsettled items per entry `reference` (invoice/bill number) with 0-30/31-60/61-90/91+ aging for any account, reversal-aware, tied to the ledger balance.
  `reference` was already stored on every entry but nothing read it.
* Procedure lists (error text in `algo.rs`) updated. `gaap.` was already on the read-only allow-list, so the new procedures are available to the HTTP server's read role unchanged.
* Version 0.18.0.

## Verification
* `tests/gaap_v018.rs` (8) and `tests/gaap_csv.rs` (2), plus the 14 v0.17 tests unchanged and passing.
* Ratios are checked against hand arithmetic on the v0.17 demonstration year (current ratio 128,100 / 35,200, ROE 8,700 / 104,700, DSO 366 / (48,500 / 8,500), ...); a two-year case checks the average-balance rule.
* Aging is checked on a hand-built receivable set (partial payment, full settlement, a reversed invoice, an unreferenced entry) and on 1,500 random invoices with random partial payments: every open
  amount and age, and the bucket totals, equal an independent tally and the ledger balance.
* Whole suite: `cargo test --release` (default features, Rust 1.91) passes in every test binary, 0 failures; the PostgreSQL-backed `pg` tests run 0 tests without a database. The GAAP tests also pass on Rust 1.85 with `--no-default-features`.
* Live data (`gaap_live_results_v0.18.md`): the 25,198-entry Superstore ledger loads through one `post_many` call (0.5 s); the statements are identical to the v0.17 run; receivable aging at each year-end
  equals an independent tally from the CSV (13 open orders / $37,853.07 at 2010-12-31, all 0-30 days, difference 0.00); FY2010 ratios were recomputed by hand from the statements.

## Limits (stated plainly)
* Ratio roles are identified by account **name and category** (see `GAAP.md`); a chart with unconventional names needs those names, or the ratio will read 0/`NULL`.
* Aging is reference-matching on the general ledger, not a subledger: no customer/vendor master, terms, dunning or write-offs. Entries without a reference are separate items.
* Not added: interest coverage, tax-rate ratios, multi-currency, drafts/approval, inventory costing, payroll, leases, bonds, fixed-asset registers. Statements still reload all entries per call (fine at the 25k-entry scale tested; no index yet).
* `post_many` validates every entry against the books as committed *before* the batch, which is exact because postings do not change accounts or periods; it does not let an entry in the batch depend on a period closed within the same batch (closing is a separate operation).
