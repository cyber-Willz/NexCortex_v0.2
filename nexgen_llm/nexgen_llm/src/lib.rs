//! `nexgen-llm` — decoder-only LLM inference on Burn 0.21.
//!
//! Supported families (`config.json` `model_type`): `llama`, `mistral`, `qwen2`, `qwen3`.
//! Anything else is rejected at load time instead of being approximated.
//!
//! Tokenizers: Hugging Face `tokenizer.json` or Meta's SentencePiece `tokenizer.model`
//! ([`spm`], pure Rust). Chat: ChatML, Llama-3 and Llama-2 `[INST]`, with control-token
//! injection protection ([`chat`]).

pub mod attention;
pub mod block;
pub mod chat;
pub mod config;
pub mod diagnostics;
pub mod generate;
pub mod mlp;
pub mod model;
pub mod rms_norm;
pub mod rope;
pub mod spm;
pub mod tokenizer;
pub mod weights;
