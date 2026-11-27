//! Stage 4 — **LLM explanation layer**.
//!
//! [`TemplateExplainer`] is deterministic and always available. [`LlmExplainer`] (feature `llm`)
//! runs a local model through `nexgen-llm`. The model only sees the fact sheet in
//! [`Brief::facts`]; its output is checked for grounding and always shown beside the facts, and
//! any failure (injection guard, generation error, ungrounded text) falls back to the template.
//! The explanation never changes the ranking or the decision options.

use crate::brief::Brief;
use anyhow::Result;

#[derive(Clone, Debug)]
pub struct Explanation {
    pub text: String,
    /// `"template"` or `"llm"`.
    pub source: &'static str,
    /// For the LLM: the text names the top candidate and no entity outside the brief's evidence
    /// and candidates. Always `true` for the template.
    pub grounded: bool,
    pub warnings: Vec<String>,
}

pub trait Explainer {
    fn explain(&self, brief: &Brief) -> Result<Explanation>;
}

pub struct TemplateExplainer;

impl Explainer for TemplateExplainer {
    fn explain(&self, b: &Brief) -> Result<Explanation> {
        let ev: Vec<String> = b.evidence.iter().map(|e| format!("{} ({:.2})", e.entity, e.severity)).collect();
        let mut t = format!("Given evidence on {}, ", ev.join(", "));
        match b.candidates.first() {
            None => t.push_str("no other entity stands out in working memory."),
            Some(c) => {
                match c.learned {
                    None => t.push_str(&format!(
                        "{} ({}) is the strongest candidate: it keeps the most activity in the recurrent network after the evidence stops (salience 1.00; the next is {:.2}). ",
                        c.entity,
                        c.kind,
                        b.candidates.get(1).map_or(0.0, |x| x.salience)
                    )),
                    Some(p) => t.push_str(&format!(
                        "{} ({}) is the top candidate of a readout trained on earlier human decisions (score {:.2}; reservoir-only rank {}, salience {:.2}). ",
                        c.entity, c.kind, p, c.baseline_rank, c.salience
                    )),
                }
                if c.path.len() > 1 {
                    t.push_str(&format!("It is {} hop(s) from the evidence via {}. ", c.path.len() - 1, c.path.join(" -> ")));
                }
                match c.diffusion_rank.cmp(&c.rank) {
                    _ if c.learned.is_some() => t.push_str(&format!("Plain diffusion ranks it {}.", c.diffusion_rank)),
                    std::cmp::Ordering::Greater => t.push_str(&format!(
                        "Plain diffusion would have ranked it {}; recurrent propagation along delayed excitatory/inhibitory paths moved it up.",
                        c.diffusion_rank
                    )),
                    std::cmp::Ordering::Less => t.push_str(&format!("Plain diffusion ranks it {}; the recurrent stage moved it down.", c.diffusion_rank)),
                    std::cmp::Ordering::Equal => t.push_str("Plain diffusion agrees with this ranking."),
                }
                if let Some(c2) = b.candidates.get(1) {
                    t.push_str(&format!(" Runner-up: {} ({}).", c2.entity, c2.kind));
                }
            }
        }
        t.push_str(&format!(" Confidence {:.2} is a heuristic, not a probability; verify against the caveats before acting.", b.confidence));
        Ok(Explanation { text: t, source: "template", grounded: true, warnings: vec![] })
    }
}

/// Names the narrative mentions that are not in the brief. `known` = every entity in the knowledge
/// base; `allowed` = the ones the brief covers.
pub fn ungrounded_mentions(text: &str, known: &[String], allowed: &[String]) -> Vec<String> {
    let lower = text.to_lowercase();
    known
        .iter()
        .filter(|k| lower.contains(&k.to_lowercase()) && !allowed.iter().any(|a| a.eq_ignore_ascii_case(k)))
        .cloned()
        .collect()
}

pub fn brief_entities(b: &Brief) -> Vec<String> {
    b.evidence.iter().map(|e| e.entity.clone()).chain(b.candidates.iter().map(|c| c.entity.clone())).collect()
}

#[cfg(feature = "llm")]
pub use llm::LlmExplainer;

#[cfg(feature = "llm")]
mod llm {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use nexgen_llm::chat::{self, ChatOptions, Message};
    use nexgen_llm::config::ModelConfig;
    use nexgen_llm::generate::{generate, SamplingConfig};
    use nexgen_llm::model::LlmModel;
    use nexgen_llm::tokenizer::{load_tokenizer, resolve_stop_ids, Tokenizer, TokenStream};
    use nexgen_llm::weights::load_model;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::path::Path;

    type B = NdArray<f32>;

    pub struct LlmExplainer {
        model: LlmModel<B>,
        tok: Tokenizer,
        stop: Vec<u32>,
        device: NdArrayDevice,
        pub sampling: SamplingConfig,
        /// System prompt (default instructs the model to use only the supplied facts).
        pub system: String,
        /// Final instruction appended to the fact sheet.
        pub instruction: String,
        /// Every entity name in the knowledge base (for the grounding check).
        known: Vec<String>,
        fallback: TemplateExplainer,
    }

    const SYSTEM: &str = "You explain a decision-support result to a human operator. Use ONLY the facts provided. \
Do not invent entities, numbers or causes. In at most four sentences say why the first candidate ranked first, \
what the diffusion-only rank adds, and what the operator should verify.";

    impl LlmExplainer {
        /// `dir` holds `config.json`, safetensors weights and a tokenizer (Llama/Mistral/Qwen2/Qwen3).
        pub fn load(dir: impl AsRef<Path>, known_entities: Vec<String>, max_new_tokens: usize) -> Result<Self> {
            let dir = dir.as_ref();
            let device = NdArrayDevice::default();
            let cfg = ModelConfig::from_json_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
            let tok = load_tokenizer(dir)?;
            let stop = resolve_stop_ids(dir, &cfg, &tok);
            let model = load_model::<B>(dir, None, &device)?;
            let sampling = SamplingConfig { max_new_tokens, ..SamplingConfig::greedy() };
            Ok(LlmExplainer { model, tok, stop, device, sampling, system: SYSTEM.to_string(), instruction: "Explain this result.".to_string(), known: known_entities, fallback: TemplateExplainer })
        }

        fn run(&self, brief: &Brief) -> Result<String> {
            let user = format!("{}\n{}", brief.facts(), self.instruction);
            let ids = match chat::detect_format(&self.tok) {
                // The injection guard rejects entity names/notes containing chat control tokens.
                Some(fmt) => chat::encode_chat(
                    fmt,
                    &self.tok,
                    &[Message::new("system", &self.system), Message::new("user", &user)],
                    &ChatOptions { no_think: true, allow_special: false },
                )?,
                None => self.tok.encode(&format!("{}\n\n{user}\n\nExplanation:", self.system), true)?,
            };
            let mut rng = StdRng::seed_from_u64(0);
            let mut stream = TokenStream::new(&self.tok).with_context(&ids);
            let mut text = String::new();
            generate(&self.model, &ids, &self.stop, &self.sampling, &self.device, &mut rng, |id| {
                if let Ok(Some(p)) = stream.push(id) {
                    text.push_str(&p);
                }
            })?;
            if let Ok(Some(rest)) = stream.flush() {
                text.push_str(&rest);
            }
            Ok(text.trim().to_string())
        }
    }

    impl Explainer for LlmExplainer {
        fn explain(&self, brief: &Brief) -> Result<Explanation> {
            let fallback = |why: String| -> Result<Explanation> {
                let mut e = self.fallback.explain(brief)?;
                e.warnings.push(why);
                Ok(e)
            };
            let text = match self.run(brief) {
                Ok(t) => t,
                Err(e) => return fallback(format!("LLM explanation unavailable ({e}); showing the deterministic template")),
            };
            let top = brief.candidates.first().map(|c| c.entity.clone());
            let allowed = brief_entities(brief);
            let stray = ungrounded_mentions(&text, &self.known, &allowed);
            let names_top = top.as_ref().map_or(true, |t| text.to_lowercase().contains(&t.to_lowercase()));
            let mut warnings = Vec::new();
            if text.is_empty() {
                return fallback("LLM produced no text; showing the deterministic template".into());
            }
            if !names_top {
                warnings.push("LLM text does not mention the top candidate".to_string());
            }
            if !stray.is_empty() {
                warnings.push(format!("LLM text mentions entities outside the brief: {}", stray.join(", ")));
            }
            let grounded = warnings.is_empty();
            if !grounded {
                // Keep the model text visible for audit, but lead with the deterministic explanation.
                let mut t = self.fallback.explain(brief)?;
                warnings.push("model text failed the grounding check and is shown only as unverified".to_string());
                t.text = format!("{}\n\n[unverified model text] {}", t.text, text);
                t.source = "llm";
                t.grounded = false;
                t.warnings = warnings;
                return Ok(t);
            }
            Ok(Explanation { text, source: "llm", grounded, warnings })
        }
    }
}
