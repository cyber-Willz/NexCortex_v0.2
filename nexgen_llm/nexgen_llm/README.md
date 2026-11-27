# nexgen-llm 0.4.0

Decoder-only LLM inference in Rust on [Burn](https://burn.dev) 0.21 (CPU/NdArray).
Families: **Llama, Mistral, Qwen2/2.5, Qwen3** (`config.json` `model_type`); anything else is
rejected at load time.

This crate merges two uploaded code bases: `burn-llm` 0.3 (the base) and `llama` 0.0.1 (a Candle
port of Meta's Llama-2 reference code).

```bash
cargo run --release --bin generate -- /path/to/Qwen3-0.6B --chat --no-think "Explain RoPE."
cargo run --release --bin generate -- /path/to/Llama-2-7b-chat-hf --chat-format llama2 --system "Be brief." "Hi"
cargo run --release --bin generate -- /path/to/model --logprobs --temp 0 "The capital of France is"
cargo run --release --bin generate -- /path/to/model --check "some prompt"      # numerical health probe
```
Needs rustc **1.92+** (Burn 0.21's real MSRV). The model directory must hold `config.json`,
safetensors weights (HF layout) and either `tokenizer.json` or `tokenizer.model`.

## Analysis of the two inputs

| | `burn-llm` 0.3 | `llama` 0.0.1 |
|---|---|---|
| Backend | Burn / NdArray | Candle (CUDA/Metal/MKL features) |
| Architectures | Llama, Mistral, Qwen2, Qwen3 | Llama-2 only (no RoPE scaling, no QK-norm, no biases) |
| Weights | HF safetensors, sharded, f32/f16/bf16 | Meta `consolidated.NN.pth`, fairscale shards merged |
| Tokenizer | HF `tokenizer.json` | pure-Rust SentencePiece `tokenizer.model` |
| Chat | ChatML, Llama-3, **no** injection guard | Llama-2 `[INST]`, special-tag guard |
| Sampling | temp, top-k, top-p, min-p, repetition penalty | temp, top-p |
| Logprobs | none | yes (+ echo) |
| Verification | independent float64 numpy reference, 3 architectures | synthetic tiny model, no external reference |

**What `burn-llm` was missing** (each of these is now covered): it could not read a Meta-style
checkpoint folder that only has `tokenizer.model`; it had no Llama-2 chat layout; its chat
renderer let user text forge turn boundaries (HF tokenizers parse `<|im_end|>` out of plain
text, and a test below shows the forged boundary really becomes a control token); it could not
report token log-probabilities; and, from reading the sampler, all-NaN logits would reach
`WeightedIndex::new(..).expect(..)` and panic with an unhelpful message.

**What `llama` does that was *not* carried over**, and why: Candle backend and GPU features
(different tensor stack, would be a second implementation of everything); `.pth` shard loading
(needs a pickle/torch reader — convert to safetensors with the usual HF script first); fixed-batch
`generate` with padding (the CLI is B=1); its flagged-unsafe dialogs are still fully generated and
then replaced with an error string — here they are refused *before* any compute instead.

## What changed from burn-llm 0.3

| Area | Change | Evidence |
|---|---|---|
| Tokenizer | `src/spm.rs`: SentencePiece BPE read straight from `tokenizer.model` (byte fallback, rejects charsmap models rather than mis-tokenize). `load_tokenizer` uses `tokenizer.json` if present, else `tokenizer.model`. Works with `TokenStream`. | 10 unit tests (ported from `llama`, adapted) |
| Chat | `ChatFormat::Llama2`, built at token level exactly like Meta's reference (system folded into first user turn *before* strip, BOS/EOS per exchange, no EOS on the last turn). Never auto-detected; select with `--chat-format llama2`. | layout + malformed-dialog tests |
| Injection | `chat::encode_chat` refuses content containing the format's control strings **or any special token the tokenizer defines**, and unknown roles. Override: `--allow-special` / `ChatOptions::allow_special`. | forged-turn test |
| Logprobs | `SamplingConfig::logprobs` → `GenerationOutput::logprobs`: ln p of each generated token under the **raw** distribution (before temperature, penalty, top-k/p, min-p). `--logprobs` prints them. | matches numpy reference to <2e-3, also under temperature+top-k+penalty |
| Robustness | Clear error (with step number) instead of a panic when logits are all non-finite or contain +inf. | by construction; not triggered by a real model |
| Housekeeping | crate renamed `nexgen-llm` (lib `nexgen_llm`); unused `thiserror` dependency removed. | |

Unchanged from 0.3 (see its notes): GQA without K/V expansion, last-position-only logits,
RoPE `linear`/`llama3`/`yarn`, Qwen3 QK-norm and explicit `head_dim`, stop-id union, chunked prefill.

## Verification actually run

Built with rustc 1.100 on CPU. `cargo test --release`: **58 tests pass** (50 unit + 8 numerical
integration). Baseline `burn-llm` 0.3 in the same environment: 40 pass. The 3 numpy cross-checks
(Qwen3/YaRN/bf16, Qwen2/Llama-3-RoPE, Llama MHA) still pass unchanged. Also run by hand: the CLI
on the Llama fixture with a hand-built 36-piece `tokenizer.model` — plain prompt, `--chat-format
llama2 --system`, `--logprobs`, injection refusal, and `--allow-special`.
Tests ran with `lto = off` for build speed (the manifest keeps `lto = true` for release).

## Not verified / limitations

- **No real checkpoint was run** (no weights available offline). The SentencePiece tokenizer and
  Llama-2 template were tested on synthetic vocabularies, not against Meta's real
  `tokenizer.model` or against `transformers`/`sentencepiece`; compare token ids on a few strings
  before trusting it with a real model.
- Llama-2 support assumes HF-converted weights (`model.layers.N...`), not Meta's native names.
- The Llama-2 template is Meta's; Mistral's `[INST]` variant differs and is not implemented.
- CPU only, no quantization, KV cache grows by concatenation, B=1, no stop *strings*.
- The injection guard rejects rather than escapes; legitimate text that happens to contain e.g.
  `[INST]` needs `--allow-special`.
- Licensing: `src/spm.rs` derives from Meta's Llama-2 code — see `NOTICE`.
