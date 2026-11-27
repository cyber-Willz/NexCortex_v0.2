# Claude API as the supporting LLM — integration and test report

## Bottom line
Claude is integrated as stage 4 (`ClaudeExplainer`, CLI `--claude <model>`). **I could not make live Claude calls: this
environment has no `ANTHROPIC_API_KEY`, so no measurement below is of Claude itself.** What is verified is the
integration (request, parsing, safeguards, fallback, HTTP path); what is not is Claude's real grounding rate and latency.
One command below produces those numbers with your key.

## What was built
- `src/claude.rs`: `ClaudeExplainer` implements `Explainer`. Anthropic Messages API (`/v1/messages`, `anthropic-version: 2023-06-01`),
  temperature 0, same facts-only system prompt as the local LLM, input = `Brief::facts()` only.
- Transport is a trait. Production = `curl` with the key supplied on stdin (never in argv or the body); tests use a mock.
  Base URL honours `ANTHROPIC_BASE_URL`.
- Safeguards identical to the local explainer: grounding check (must name the top candidate; no KB entity outside the brief),
  failed check -> template leads and the model text is shown as `[unverified model text]`; any API error, bad JSON, empty
  text -> deterministic template with a warning. Explanation never touches ranking.
- `Stats`: calls, API errors, grounded, unverified, latency, tokens.

## Verification (all pass; full suite 28 tests)
| Test | Result |
|---|---|
| Grounded reply accepted; request body is facts-only, key absent from body | pass |
| Hallucinated entity / missing top candidate -> unverified, template leads | pass |
| 401 / non-JSON / API error object / empty text -> template fallback | pass |
| Real `curl` path against a local HTTP server: headers, path, parse | pass |

## Harness run on ATT&CK held-out briefs (SIMULATED transport, not Claude)
40 held-out briefs from the learned-readout pipeline; a scripted fake API cycles faithful / hallucinating / error replies
(3:1:1) to prove the checks fire:
- ranking identical before vs after explanation: **40/40** (by construction)
- faithful replies accepted: 24; hallucinated (`T9999`) flagged unverified: 8; API errors -> template: 8
- template explanation grounded: 40/40
These counts reflect the script I wrote, not Claude's behaviour. Latency shows 0 ms for the same reason.

## What this does NOT change
The LLM never alters ranking, so accuracy results are unchanged: ATT&CK small learned-readout gain; `accounting_data.csv`
and `ai_accounting_dataset.csv` at chance. Claude adds readable explanation, not predictive power.

## Run it live
```
export ANTHROPIC_API_KEY=...           # your key
cargo run --release --example llm_eval -- /path/to/attack_data 40 claude-sonnet-5-5
cargo run --release -- --claude claude-sonnet-5-5 --script approve     # single brief via CLI
```
Read the output for: grounded vs unverified counts, API errors, p50/p95 latency, tokens. Caveats: the grounding check is
a name-based heuristic (it does not verify numbers or causal claims), and briefs send entity names/notes to a third party.
