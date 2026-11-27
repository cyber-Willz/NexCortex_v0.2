use anyhow::{bail, Result};
use burn::tensor::{backend::Backend, Int, Tensor};
// burn 0.21 pulls in rand 0.10 (API as compiled in v0.2: `rand::distr::weighted`).
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::Rng;

use crate::model::{GenerationState, LlmModel};

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// `<= 0` means greedy (argmax).
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    /// Drop tokens whose probability is below `min_p × p(top token)`.
    pub min_p: Option<f32>,
    /// HF/CTRL-style multiplicative penalty on logits of tokens already in the sequence
    /// (`1.0` = off; typical 1.05–1.2).
    pub repetition_penalty: f32,
    pub max_new_tokens: usize,
    /// Prompt tokens per forward pass during prefill. Attention scores are
    /// `O(chunk × kv_len)` per head, so this bounds peak memory for long prompts.
    pub prefill_chunk: usize,
    /// Record the log-probability of every generated token (see [`GenerationOutput::logprobs`]).
    pub logprobs: bool,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: Some(50),
            top_p: Some(0.9),
            min_p: None,
            repetition_penalty: 1.0,
            max_new_tokens: 256,
            prefill_chunk: 512,
            logprobs: false,
        }
    }
}

impl SamplingConfig {
    pub fn greedy() -> Self {
        Self { temperature: 0.0, top_k: None, top_p: None, min_p: None, ..Self::default() }
    }
}

fn logits_to_vec<B: Backend>(logits: Tensor<B, 2>) -> Vec<f32> {
    logits
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("logits tensor should convert to Vec<f32>")
}

/// Index of the maximum finite logit (NaN-safe: `total_cmp`, NaNs sort below everything real).
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| {
            let (x, y) = (if a.1.is_nan() { f32::NEG_INFINITY } else { *a.1 }, if b.1.is_nan() { f32::NEG_INFINITY } else { *b.1 });
            x.total_cmp(&y)
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// `ln softmax(logits)[id]`, computed stably (log-sum-exp). NaNs are treated as `-inf`.
pub fn log_prob_of(logits: &[f32], id: u32) -> f32 {
    let clean = |x: f32| if x.is_nan() { f32::NEG_INFINITY } else { x };
    let max = logits.iter().copied().map(clean).fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&x| ((clean(x) - max) as f64).exp()).sum();
    clean(logits[id as usize]) - max - sum.ln() as f32
}

/// Applies the repetition penalty in place: positive logits are divided, negative multiplied.
pub fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < f32::EPSILON {
        return;
    }
    let mut seen = std::collections::HashSet::new();
    for &t in history {
        if seen.insert(t) {
            if let Some(l) = logits.get_mut(t as usize) {
                *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
            }
        }
    }
}

/// Samples a token id from raw `logits` (already penalised by the caller).
///
/// Order: temperature → softmax → sort → top-k → min-p → top-p → renormalise → draw.
pub fn sample_from_logits(logits: &[f32], cfg: &SamplingConfig, rng: &mut impl Rng) -> u32 {
    if cfg.temperature <= 0.0 {
        return argmax(logits);
    }
    let inv_t = 1.0 / cfg.temperature;
    let max_logit = logits.iter().cloned().filter(|x| !x.is_nan()).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, if l.is_nan() { 0.0 } else { ((l - max_logit) * inv_t).exp() }))
        .collect();
    let sum: f32 = probs.iter().map(|p| p.1).sum();
    for p in probs.iter_mut() {
        p.1 /= sum;
    }
    probs.sort_by(|a, b| b.1.total_cmp(&a.1));

    if let Some(k) = cfg.top_k {
        probs.truncate(k.max(1));
    }
    if let Some(m) = cfg.min_p {
        let floor = m * probs[0].1;
        let keep = probs.iter().take_while(|p| p.1 >= floor).count().max(1);
        probs.truncate(keep);
    }
    if let Some(top_p) = cfg.top_p {
        let mut cum = 0.0;
        let mut cut = probs.len();
        for (i, p) in probs.iter().enumerate() {
            cum += p.1;
            if cum >= top_p {
                cut = i + 1;
                break;
            }
        }
        probs.truncate(cut.max(1));
    }

    let weights: Vec<f32> = probs.iter().map(|p| p.1).collect();
    let dist = WeightedIndex::new(&weights).expect("non-empty, non-negative weights");
    probs[dist.sample(rng)].0 as u32
}

fn ids_tensor<B: Backend>(ids: &[u32], device: &B::Device) -> Tensor<B, 2, Int> {
    let v: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
    Tensor::<B, 1, Int>::from_ints(v.as_slice(), device).reshape([1, ids.len()])
}

/// Prefills `prompt` in chunks and returns the logits of its final position.
pub fn prefill<B: Backend>(
    model: &LlmModel<B>,
    state: &mut GenerationState<B>,
    prompt: &[u32],
    chunk: usize,
    device: &B::Device,
) -> Vec<f32> {
    let chunk = chunk.max(1);
    let mut last = Vec::new();
    for c in prompt.chunks(chunk) {
        last = logits_to_vec(model.forward_last(ids_tensor::<B>(c, device), state));
    }
    last
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

pub struct GenerationOutput {
    /// Prompt followed by everything generated (including the stop token, if one was hit).
    pub sequence: Vec<u32>,
    /// With [`SamplingConfig::logprobs`]: `ln p(token)` for each generated token (stop token
    /// included), i.e. `sequence[prompt_len + i]`, under the model's **raw** next-token
    /// distribution (temperature 1, before repetition penalty / top-k / top-p / min-p). Empty
    /// otherwise.
    pub logprobs: Vec<f32>,
    pub new_tokens: usize,
    pub finish: FinishReason,
}

/// Prefill + autoregressive decode.
///
/// * `stop_ids` — see [`crate::tokenizer::resolve_stop_ids`]; generation ends on any of them.
/// * `on_token` — called with each newly sampled id (excluding the stop token), for streaming.
pub fn generate<B: Backend>(
    model: &LlmModel<B>,
    prompt_ids: &[u32],
    stop_ids: &[u32],
    cfg: &SamplingConfig,
    device: &B::Device,
    rng: &mut impl Rng,
    mut on_token: impl FnMut(u32),
) -> Result<GenerationOutput> {
    if prompt_ids.is_empty() {
        bail!("empty prompt (tokenizer returned no ids)");
    }
    let vocab = model.config.vocab_size;
    if let Some(bad) = prompt_ids.iter().find(|&&t| t as usize >= vocab) {
        bail!("prompt contains token id {bad} but the model's vocab_size is {vocab}");
    }
    let max_pos = model.config.max_position_embeddings;
    if prompt_ids.len() >= max_pos {
        bail!("prompt ({} tokens) does not fit the model's {max_pos}-position context", prompt_ids.len());
    }
    // Reserve exactly what this run can use (clamped to the trained context).
    let budget = cfg.max_new_tokens.min(max_pos - prompt_ids.len());
    let mut state = model.init_generation(device, prompt_ids.len() + budget);

    let mut sequence = prompt_ids.to_vec();
    let mut logprobs = Vec::new();
    let mut logits = prefill(model, &mut state, prompt_ids, cfg.prefill_chunk, device);
    let mut finish = FinishReason::Length;

    for step in 0..budget {
        // All-NaN/inf logits would make the sampler divide 0 by 0 and panic deep inside
        // `WeightedIndex`; fail with a message that names the step instead.
        if !logits.iter().any(|x| x.is_finite()) || logits.contains(&f32::INFINITY) {
            bail!("model produced no finite logits (or a +inf logit) at generation step {step} (numerical failure; try `--check`)");
        }
        let raw = cfg.logprobs.then(|| logits.clone());
        apply_repetition_penalty(&mut logits, &sequence, cfg.repetition_penalty);
        let next = sample_from_logits(&logits, cfg, rng);
        if let Some(raw) = raw {
            logprobs.push(log_prob_of(&raw, next));
        }
        sequence.push(next);
        if stop_ids.contains(&next) {
            finish = FinishReason::Stop;
            break;
        }
        on_token(next);
        // Skip the forward pass that would produce logits we'd never sample from.
        if state.remaining() == 0 {
            break;
        }
        logits = logits_to_vec(model.forward_last(ids_tensor::<B>(&[next], device), &mut state));
    }

    Ok(GenerationOutput { new_tokens: sequence.len() - prompt_ids.len(), sequence, logprobs, finish })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(1234)
    }

    #[test]
    fn greedy_is_argmax_and_nan_safe() {
        assert_eq!(argmax(&[0.1, 2.0, -1.0]), 1);
        assert_eq!(argmax(&[f32::NAN, -5.0, -6.0]), 1);
        assert_eq!(sample_from_logits(&[0.0, 3.0, 1.0], &SamplingConfig::greedy(), &mut rng()), 1);
    }

    #[test]
    fn top_k_one_always_returns_the_top_token() {
        let cfg = SamplingConfig { temperature: 1.0, top_k: Some(1), top_p: None, ..Default::default() };
        let mut r = rng();
        for _ in 0..50 {
            assert_eq!(sample_from_logits(&[0.0, 0.5, 2.0, 1.9], &cfg, &mut r), 2);
        }
    }

    #[test]
    fn min_p_excludes_unlikely_tokens() {
        // probs ≈ [0.9, 0.09, 0.01]; min_p 0.5 keeps only tokens with p >= 0.45.
        let l = [(0.9f32).ln(), (0.09f32).ln(), (0.01f32).ln()];
        let cfg = SamplingConfig { temperature: 1.0, top_k: None, top_p: None, min_p: Some(0.5), ..Default::default() };
        let mut r = rng();
        for _ in 0..200 {
            assert_eq!(sample_from_logits(&l, &cfg, &mut r), 0);
        }
    }

    #[test]
    fn top_p_keeps_smallest_nucleus() {
        let l = [(0.6f32).ln(), (0.3f32).ln(), (0.1f32).ln()];
        let cfg = SamplingConfig { temperature: 1.0, top_k: None, top_p: Some(0.85), min_p: None, ..Default::default() };
        let mut r = rng();
        let mut seen = [false; 3];
        for _ in 0..500 {
            seen[sample_from_logits(&l, &cfg, &mut r) as usize] = true;
        }
        assert!(seen[0] && seen[1] && !seen[2], "nucleus is {{0,1}}: {seen:?}");
    }

    #[test]
    fn sampling_matches_distribution_roughly() {
        let l = [(0.75f32).ln(), (0.25f32).ln()];
        let cfg = SamplingConfig { temperature: 1.0, top_k: None, top_p: None, min_p: None, ..Default::default() };
        let mut r = rng();
        let n = 4000;
        let ones = (0..n).filter(|_| sample_from_logits(&l, &cfg, &mut r) == 1).count() as f32 / n as f32;
        assert!((ones - 0.25).abs() < 0.03, "{ones}");
    }

    #[test]
    fn log_prob_matches_softmax_and_is_stable() {
        let l = [(0.75f32).ln(), (0.25f32).ln()];
        assert!((log_prob_of(&l, 0) - (0.75f32).ln()).abs() < 1e-6);
        // huge logits must not overflow
        let big = [1000.0f32, 1000.0];
        assert!((log_prob_of(&big, 1) - (0.5f32).ln()).abs() < 1e-5);
        // NaN counts as impossible, finite entries still normalise
        let nan = [f32::NAN, 0.0];
        assert!(log_prob_of(&nan, 1).abs() < 1e-6);
        assert_eq!(log_prob_of(&nan, 0), f32::NEG_INFINITY);
    }

    #[test]
    fn repetition_penalty_directions_and_dedup() {
        let mut l = vec![2.0, -2.0, 1.0];
        apply_repetition_penalty(&mut l, &[0, 1, 0, 0], 2.0);
        assert_eq!(l, vec![1.0, -4.0, 1.0]); // token 0 penalised once despite 3 occurrences
        let mut same = vec![2.0, -2.0];
        apply_repetition_penalty(&mut same, &[0, 1], 1.0);
        assert_eq!(same, vec![2.0, -2.0]);
    }
}
