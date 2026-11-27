//! Learned readout on top of the connectome reservoir.
//!
//! Until now the reservoir was read out with a fixed rule (mean excitatory rate in the window after
//! the last pulse). This module replaces that rule by a **trained** one, fitted on what human
//! operators actually decided. The reservoir itself stays frozen (a calibrated echo-state-style
//! network); only the readout is learned, which is the standard reservoir-computing split.
//!
//! ```text
//!   graph features        (structure + diffusion + proximity to the evidence)   \
//!   reservoir trajectory  (the entity's activity over time, plus its neighbours) >-> w . phi -> softmax over candidates
//!   human decisions       (earlier choices, refusals and similar precedents)    /
//! ```
//!
//! ## Model
//! A conditional-logit (listwise softmax) model: for one query with candidate pool `P`,
//! `P(chosen = i) = softmax_i(w . standardise(phi_i))`. The weights are shared by all entities, so
//! the model has `N_FEATURES` parameters however big the knowledge graph is. The loss is convex
//! (softmax cross-entropy + L2), solved exactly by damped Newton with a Cholesky solve: no learning
//! rate, no random initialisation, fully deterministic.
//!
//! ## Labels and the selection-bias rule
//! * `approve` / `override` give a positive (the chosen entity). `reject` / `defer` carry no
//!   positive and are **not** used as supervised examples (they still feed the human features).
//! * The operator only ever saw the machine's top-`top_k`. For an `approve` the pool is those shown
//!   candidates. For an `override` (every shown candidate refused, something else named) it is all
//!   the entities that were *not* shown: the target competes only with its own tier, never with the
//!   shown ones, so no candidate is distinguishable by construction. At inference the readout re-ranks a larger pool
//!   ([`PipelineConfig::rerank_pool`](crate::PipelineConfig)), which is a mild extrapolation; the
//!   features are normalised over all entities, not over the pool, to keep both consistent.
//!
//! ## No leakage
//! Training examples are rebuilt by **replaying** the decision log in order. Example `k` sees only
//! the human history and the prior seeds that existed *before* decision `k`; its own label never
//! enters its own features.
//!
//! ## Honest limits
//! Few decisions mean a weak, over-fitted model; [`evaluate`] reports leave-one-out and
//! forward-chaining (train on the past, predict the next decision) numbers next to the two
//! untrained baselines, and a readout that does not beat them should not be used.

use crate::features::{EventField, Structure};
use crate::kb::{DecisionRecord, KnowledgeBase};
use crate::memory::MemoryTrace;
use crate::pipeline::{Pipeline, Query, EventSpec};
use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    Graph,
    Reservoir,
    Human,
}

pub const N_FEATURES: usize = 26;

/// Feature names, in column order, with the source each one belongs to.
pub const FEATURES: [(&str, Group); N_FEATURES] = [
    // --- graph ---
    ("pagerank", Group::Graph),
    ("embed_closeness", Group::Graph),
    ("cluster_share", Group::Graph),
    ("heat_short", Group::Graph),
    ("heat_mid", Group::Graph),
    ("heat_long", Group::Graph),
    ("diffusion", Group::Graph),
    ("degree", Group::Graph),
    ("path_proximity", Group::Graph),
    ("hop_proximity", Group::Graph),
    ("direct_tie", Group::Graph),
    // --- reservoir trajectory ---
    ("res_post_window", Group::Reservoir),
    ("res_bin_0_3", Group::Reservoir),
    ("res_bin_3_7", Group::Reservoir),
    ("res_bin_7_12", Group::Reservoir),
    ("res_bin_12_20", Group::Reservoir),
    ("res_bin_20_end", Group::Reservoir),
    ("res_at_input_end", Group::Reservoir),
    ("res_peak", Group::Reservoir),
    ("res_peak_time", Group::Reservoir),
    ("res_neighbour_post", Group::Reservoir),
    ("res_post_rank_pct", Group::Reservoir),
    // --- human decisions ---
    ("human_confirmed", Group::Human),
    ("human_refused", Group::Human),
    ("human_precedent_for", Group::Human),
    ("human_precedent_against", Group::Human),
];

const DIFFUSION_COL: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Groups {
    pub graph: bool,
    pub reservoir: bool,
    pub human: bool,
}

impl Groups {
    pub const ALL: Groups = Groups { graph: true, reservoir: true, human: true };
    pub fn has(&self, g: Group) -> bool {
        match g {
            Group::Graph => self.graph,
            Group::Reservoir => self.reservoir,
            Group::Human => self.human,
        }
    }
    /// `"all"` or a comma list of `graph`, `reservoir`, `human`.
    pub fn parse(s: &str) -> Result<Groups> {
        let mut g = Groups { graph: false, reservoir: false, human: false };
        for p in s.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()) {
            match p {
                "all" => g = Groups::ALL,
                "graph" => g.graph = true,
                "reservoir" => g.reservoir = true,
                "human" => g.human = true,
                o => bail!("unknown feature group {o:?} (use graph, reservoir, human or all)"),
            }
        }
        if !(g.graph || g.reservoir || g.human) {
            bail!("no feature group selected");
        }
        Ok(g)
    }
    pub fn label(&self) -> String {
        let mut v = Vec::new();
        if self.graph {
            v.push("graph");
        }
        if self.reservoir {
            v.push("reservoir");
        }
        if self.human {
            v.push("human");
        }
        v.join(",")
    }
}

// ---------------------------------------------------------------------------------------------
// Human decisions as features
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Past {
    evidence: Vec<(usize, f64)>,
    chosen: Option<usize>,
    refused: Option<usize>,
}

/// What operators decided before "now", resolved to entity indices.
#[derive(Clone, Debug, Default)]
pub struct HumanHistory {
    past: Vec<Past>,
    /// Records dropped because they name entities that no longer exist.
    pub skipped: usize,
}

impl HumanHistory {
    pub fn from_records(kb: &KnowledgeBase, recs: &[DecisionRecord]) -> HumanHistory {
        let idx = |name: &str| kb.entity(name).and_then(|e| kb.index_of(e.id));
        let mut h = HumanHistory::default();
        for r in recs {
            let evidence: Option<Vec<(usize, f64)>> = r.evidence.iter().map(|(n, s)| idx(n).map(|i| (i, *s))).collect();
            let Some(evidence) = evidence else {
                h.skipped += 1;
                continue;
            };
            let picked = matches!(r.kind.as_str(), "approve" | "override");
            let chosen = if picked { r.chosen.as_deref().and_then(idx) } else { None };
            let refused = r.refused.as_deref().and_then(idx);
            h.past.push(Past { evidence, chosen, refused });
        }
        h
    }
    pub fn len(&self) -> usize {
        self.past.len()
    }
    pub fn is_empty(&self) -> bool {
        self.past.is_empty()
    }
}

fn cosine(a: &[(usize, f64)], b: &[(usize, f64)]) -> f64 {
    let dot: f64 = a.iter().flat_map(|x| b.iter().filter(move |y| y.0 == x.0).map(move |y| x.1 * y.1)).sum();
    let na = a.iter().map(|x| x.1 * x.1).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x.1 * x.1).sum::<f64>().sqrt();
    if na * nb > 0.0 {
        (dot / (na * nb)).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------------------------

/// Everything the feature extractor reads. Built by the pipeline; also used when replaying history.
pub struct Inputs<'a> {
    pub structure: &'a Structure,
    /// Dense `REL` adjacency, `n x n`.
    pub adj: &'a [f64],
    pub n: usize,
    /// All evidence the reservoir was driven with: operator evidence plus prior seeds.
    pub evidence: &'a [(usize, f64)],
    /// Operator-supplied evidence only (used to match precedents).
    pub user_evidence: &'a [(usize, f64)],
    pub fields: &'a [EventField],
    pub trace: &'a MemoryTrace,
    /// Per-entity severity-weighted one-shot diffusion score.
    pub diffusion: &'a [f64],
    pub history: &'a HumanHistory,
}

/// One feature row for every non-evidence entity.
#[derive(Clone, Debug)]
pub struct FeatureTable {
    /// Entity indices (ascending), evidence excluded.
    pub entities: Vec<usize>,
    pub rows: Vec<[f64; N_FEATURES]>,
}

impl FeatureTable {
    pub fn row_of(&self, entity: usize) -> Option<&[f64; N_FEATURES]> {
        self.entities.binary_search(&entity).ok().map(|p| &self.rows[p])
    }
}

/// Multi-source shortest "cost" (sum of 1/weight, so strong ties are short) and hop count.
fn proximity(adj: &[f64], n: usize, src: &[usize]) -> (Vec<f64>, Vec<usize>) {
    let mut dist = vec![f64::INFINITY; n];
    let mut done = vec![false; n];
    for &s in src {
        dist[s] = 0.0;
    }
    for _ in 0..n {
        let Some(u) = (0..n).filter(|&i| !done[i] && dist[i].is_finite()).min_by(|&a, &b| dist[a].total_cmp(&dist[b])) else { break };
        done[u] = true;
        for v in 0..n {
            let w = adj[u * n + v];
            if w > 0.0 && dist[u] + 1.0 / w < dist[v] {
                dist[v] = dist[u] + 1.0 / w;
            }
        }
    }
    let mut hops = vec![usize::MAX; n];
    let mut queue: std::collections::VecDeque<usize> = src.iter().copied().collect();
    for &s in src {
        hops[s] = 0;
    }
    while let Some(u) = queue.pop_front() {
        for v in 0..n {
            if adj[u * n + v] > 0.0 && hops[v] == usize::MAX {
                hops[v] = hops[u] + 1;
                queue.push_back(v);
            }
        }
    }
    (dist, hops)
}

pub fn extract(x: &Inputs) -> FeatureTable {
    let n = x.n;
    let mut is_ev = vec![false; n];
    for &(e, _) in x.evidence {
        is_ev[e] = true;
    }
    let cands: Vec<usize> = (0..n).filter(|&i| !is_ev[i]).collect();
    let m = cands.len();
    let sev_sum = x.evidence.iter().map(|e| e.1).sum::<f64>().max(1e-12);
    // Scale-free: every magnitude is divided by its maximum over *all* candidates, never over a pool.
    let cmax = |v: &[f64]| cands.iter().map(|&i| v[i].abs()).fold(0.0f64, f64::max).max(1e-12);
    let st = x.structure;
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(N_FEATURES);

    // ---- graph ----
    cols.push(st.pagerank.iter().map(|p| p * n as f64).collect());
    let mut cen = [0.0f64; 3];
    for &(e, s) in x.evidence {
        for k in 0..3 {
            cen[k] += s * st.embedding[e][k];
        }
    }
    cen.iter_mut().for_each(|c| *c /= sev_sum);
    let dist_emb: Vec<f64> = (0..n).map(|i| (0..3).map(|k| (st.embedding[i][k] - cen[k]).powi(2)).sum::<f64>().sqrt()).collect();
    let dm = cmax(&dist_emb);
    cols.push(dist_emb.iter().map(|d| -d / dm).collect());
    cols.push((0..n).map(|i| x.evidence.iter().filter(|e| st.cluster[e.0] == st.cluster[i]).map(|e| e.1).sum::<f64>() / sev_sum).collect());
    let n_scales = x.fields.first().map_or(0, |f| f.heat.len());
    for s_idx in [0, n_scales / 2, n_scales.saturating_sub(1)] {
        let mut h = vec![0.0; n];
        for f in x.fields {
            if let Some(hv) = f.heat.get(s_idx) {
                for i in 0..n {
                    h[i] += f.severity * hv[i];
                }
            }
        }
        let mx = cmax(&h);
        h.iter_mut().for_each(|v| *v /= mx);
        cols.push(h);
    }
    let dmax = cmax(x.diffusion);
    cols.push(x.diffusion.iter().map(|v| v / dmax).collect());
    let deg: Vec<f64> = (0..n).map(|i| x.adj[i * n..(i + 1) * n].iter().sum()).collect();
    let gm = cmax(&deg);
    cols.push(deg.iter().map(|d| d / gm).collect());
    let ev_idx: Vec<usize> = x.evidence.iter().map(|e| e.0).collect();
    let (cost, hops) = proximity(x.adj, n, &ev_idx);
    cols.push(cost.iter().map(|&d| if d.is_finite() { 1.0 / (1.0 + d) } else { 0.0 }).collect());
    cols.push(hops.iter().map(|&h| if h == usize::MAX { 0.0 } else { 1.0 / (1.0 + h as f64) }).collect());
    let mut tie = vec![0.0; n];
    for &(e, s) in x.evidence {
        for i in 0..n {
            tie[i] += s * x.adj[e * n + i];
        }
    }
    let tm = cmax(&tie);
    cols.push(tie.iter().map(|v| v / tm).collect());

    // ---- reservoir trajectory ----
    let tr = x.trace;
    let sal = &tr.salience;
    let big_m = cmax(sal);
    cols.push(sal.iter().map(|v| v / big_m).collect());
    let ie = tr.input_end;
    let settle = (tr.steps - ie).max(1);
    for (a, b) in [(0usize, 3usize), (3, 7), (7, 12), (12, 20), (20, usize::MAX)] {
        let a = a.min(settle - 1);
        let b = b.min(settle).max(a + 1);
        cols.push(tr.mean_between(ie + 1 + a, ie + 1 + b).iter().map(|v| v / big_m).collect());
    }
    cols.push(tr.at_input_end.iter().map(|v| v / big_m).collect());
    let mut peak = vec![f64::MIN; n];
    let mut t_peak = vec![0.0; n];
    for s in 0..=tr.steps {
        for i in 0..n {
            let r = tr.e_rates[s * n + i];
            if r > peak[i] {
                peak[i] = r;
                t_peak[i] = s as f64 / tr.steps.max(1) as f64;
            }
        }
    }
    cols.push(peak.iter().map(|v| v / big_m).collect());
    cols.push(t_peak);
    cols.push(
        (0..n)
            .map(|i| {
                let row = &x.adj[i * n..(i + 1) * n];
                let tot: f64 = row.iter().sum();
                if tot > 0.0 {
                    row.iter().zip(sal).map(|(w, s)| w * s).sum::<f64>() / tot / big_m
                } else {
                    0.0
                }
            })
            .collect(),
    );
    let mut by_sal = cands.clone();
    by_sal.sort_by(|&a, &b| sal[b].total_cmp(&sal[a]).then(a.cmp(&b)));
    let mut pct = vec![0.0; n];
    for (pos, &i) in by_sal.iter().enumerate() {
        pct[i] = if m > 1 { 1.0 - pos as f64 / (m - 1) as f64 } else { 1.0 };
    }
    cols.push(pct);

    // ---- human decisions ----
    let (mut conf, mut refu) = (vec![0.0f64; n], vec![0.0f64; n]);
    let (mut prec_for, mut prec_against) = (vec![0.0f64; n], vec![0.0f64; n]);
    for p in &x.history.past {
        let sim = cosine(x.user_evidence, &p.evidence);
        if let Some(c) = p.chosen {
            conf[c] += 1.0;
            prec_for[c] += sim;
        }
        if let Some(r) = p.refused {
            refu[r] += 1.0;
            prec_against[r] += sim;
        }
    }
    cols.push(conf.iter().map(|c| c.ln_1p()).collect());
    cols.push(refu.iter().map(|c| c.ln_1p()).collect());
    cols.push(prec_for.iter().map(|s| 1.0 - (-s).exp()).collect());
    cols.push(prec_against.iter().map(|s| 1.0 - (-s).exp()).collect());

    debug_assert_eq!(cols.len(), N_FEATURES);
    let rows = cands
        .iter()
        .map(|&i| {
            let mut r = [0.0; N_FEATURES];
            for (k, c) in cols.iter().enumerate() {
                r[k] = if c[i].is_finite() { c[i] } else { 0.0 };
            }
            r
        })
        .collect();
    FeatureTable { entities: cands, rows }
}

// ---------------------------------------------------------------------------------------------
// Training data: replay of the decision log
// ---------------------------------------------------------------------------------------------

/// One supervised example: the pool the operator chose from and which entry they chose.
#[derive(Clone, Debug)]
pub struct Example {
    pub question: String,
    /// Pool entity indices in baseline order: the shown candidates (approve) or, for an override,
    /// every candidate that was not shown.
    pub entities: Vec<usize>,
    pub rows: Vec<[f64; N_FEATURES]>,
    pub positive: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Skipped {
    /// Decisions made before evidence was logged.
    pub legacy: usize,
    /// `reject` / `defer`: no entity was chosen.
    pub no_label: usize,
    /// Logged entities that no longer exist in the knowledge base.
    pub unknown_entity: usize,
    /// The chosen entity was itself evidence (cannot be a candidate).
    pub chose_evidence: usize,
}

#[derive(Clone, Debug)]
pub struct TrainingSet {
    pub examples: Vec<Example>,
    pub skipped: Skipped,
    /// Candidates shown to the operator when the examples were rebuilt (`PipelineConfig::top_k`).
    pub shown: usize,
    pub fingerprint: u64,
}

impl TrainingSet {
    /// Rebuild one example per labelled decision by replaying the log in order against the
    /// pipeline's current (frozen) reservoir.
    pub fn from_pipeline(pipe: &Pipeline) -> Result<TrainingSet> {
        let (log, legacy) = pipe.kb.decision_log()?;
        let n = pipe.kb.entities().len();
        let mut skipped = Skipped { legacy, ..Skipped::default() };
        let mut examples = Vec::new();
        let mut confirmed: HashMap<String, i64> = HashMap::new();
        for (k, rec) in log.iter().enumerate() {
            // State of the world *before* this decision.
            let history = HumanHistory::from_records(&pipe.kb, &log[..k]);
            let mut priors: Vec<(String, i64)> = confirmed.iter().map(|(a, b)| (a.clone(), *b)).collect();
            priors.sort();
            let label = matches!(rec.kind.as_str(), "approve" | "override").then(|| rec.chosen.clone()).flatten();
            if let Some(name) = &rec.chosen {
                if matches!(rec.kind.as_str(), "approve" | "override") {
                    *confirmed.entry(name.clone()).or_insert(0) += 1;
                }
            }
            let Some(chosen_name) = label else {
                skipped.no_label += 1;
                continue;
            };
            let known = |nm: &str| pipe.kb.entity(nm).is_some();
            if !known(&chosen_name) || !rec.evidence.iter().all(|e| known(&e.0)) {
                skipped.unknown_entity += 1;
                continue;
            }
            let q = Query {
                question: rec.question.clone(),
                events: rec.evidence.iter().map(|(e, s)| EventSpec { entity: e.clone(), severity: *s, note: String::new() }).collect(),
            };
            let a = pipe.analyze_with(&q, &priors, &history, None)?;
            let chosen = pipe.kb.index_of(pipe.kb.entity(&chosen_name).unwrap().id).unwrap();
            if a.table.row_of(chosen).is_none() {
                skipped.chose_evidence += 1;
                continue;
            }
            // Pool = the tier the operator actually chose from. Approve: they picked one of the shown
            // candidates. Override: they turned every shown candidate down and named something
            // else, so the alternatives are all entities that were NOT shown. (Appending the
            // override target to the shown set would make it the one candidate that is always
            // ranked below the others, a construction artifact the model would learn.)
            let shown: Vec<usize> = a.baseline_order.iter().take(pipe.cfg.top_k).copied().collect();
            let entities: Vec<usize> = if shown.contains(&chosen) { shown } else { a.baseline_order.iter().skip(pipe.cfg.top_k).copied().collect() };
            if entities.len() < 2 || n < 2 {
                skipped.no_label += 1;
                continue;
            }
            let rows = entities.iter().map(|&e| *a.table.row_of(e).expect("pool entities are candidates")).collect();
            let positive = entities.iter().position(|&e| e == chosen).unwrap();
            examples.push(Example { question: rec.question.clone(), entities, rows, positive });
        }
        Ok(TrainingSet { examples, skipped, shown: pipe.cfg.top_k, fingerprint: pipe.fingerprint() })
    }
}

// ---------------------------------------------------------------------------------------------
// The readout model
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ReadoutConfig {
    pub groups: Groups,
    /// L2 strength on the standardised weights (loss is a sum over examples, not a mean).
    pub l2: f64,
    /// Refuse to train on fewer labelled examples than this.
    pub min_examples: usize,
    pub max_iter: usize,
}

impl Default for ReadoutConfig {
    fn default() -> Self {
        ReadoutConfig { groups: Groups::ALL, l2: 1.0, min_examples: 8, max_iter: 100 }
    }
}

#[derive(Clone, Debug)]
pub struct Readout {
    pub groups: Groups,
    pub l2: f64,
    pub n_train: usize,
    /// Must equal [`Pipeline::fingerprint`] for the readout to be attached.
    pub fingerprint: u64,
    mean: [f64; N_FEATURES],
    std: [f64; N_FEATURES],
    weight: [f64; N_FEATURES],
}

type Xs = Vec<Vec<[f64; N_FEATURES]>>;

fn standardise(rows: &[[f64; N_FEATURES]], mean: &[f64; N_FEATURES], std: &[f64; N_FEATURES]) -> Vec<[f64; N_FEATURES]> {
    rows.iter()
        .map(|r| {
            let mut z = [0.0; N_FEATURES];
            for k in 0..N_FEATURES {
                z[k] = (r[k] - mean[k]) / std[k];
            }
            z
        })
        .collect()
}

fn objective(xs: &Xs, pos: &[usize], w: &[f64; N_FEATURES], l2: f64) -> f64 {
    let mut f = 0.5 * l2 * w.iter().map(|x| x * x).sum::<f64>();
    for (rows, &p) in xs.iter().zip(pos) {
        let z: Vec<f64> = rows.iter().map(|r| r.iter().zip(w).map(|(a, b)| a * b).sum()).collect();
        let mx = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = mx + z.iter().map(|v| (v - mx).exp()).sum::<f64>().ln();
        f += lse - z[p];
    }
    f
}

fn grad_hess(xs: &Xs, pos: &[usize], w: &[f64; N_FEATURES], l2: f64) -> ([f64; N_FEATURES], Vec<f64>) {
    let d = N_FEATURES;
    let mut g = [0.0; N_FEATURES];
    let mut h = vec![0.0; d * d];
    for k in 0..d {
        g[k] = l2 * w[k];
        h[k * d + k] = l2;
    }
    for (rows, &p) in xs.iter().zip(pos) {
        let z: Vec<f64> = rows.iter().map(|r| r.iter().zip(w).map(|(a, b)| a * b).sum()).collect();
        let mx = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = z.iter().map(|v| (v - mx).exp()).collect();
        let s: f64 = e.iter().sum();
        let pr: Vec<f64> = e.iter().map(|v| v / s).collect();
        let mut mean = [0.0; N_FEATURES];
        for (r, &pi) in rows.iter().zip(&pr) {
            for k in 0..d {
                mean[k] += pi * r[k];
            }
        }
        for k in 0..d {
            g[k] += mean[k] - rows[p][k];
        }
        for (r, &pi) in rows.iter().zip(&pr) {
            for a in 0..d {
                for b in 0..d {
                    h[a * d + b] += pi * r[a] * r[b];
                }
            }
        }
        for a in 0..d {
            for b in 0..d {
                h[a * d + b] -= mean[a] * mean[b];
            }
        }
    }
    (g, h)
}

/// Solve `H x = b` for symmetric positive definite `H` by Cholesky; `None` if it is not SPD.
fn solve_spd(h: &[f64], b: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = h[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if s <= 1e-14 {
                    return None;
                }
                l[i * d + i] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    let mut y = vec![0.0; d];
    for i in 0..d {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * d + k] * y[k];
        }
        y[i] = s / l[i * d + i];
    }
    let mut x = vec![0.0; d];
    for i in (0..d).rev() {
        let mut s = y[i];
        for k in i + 1..d {
            s -= l[k * d + i] * x[k];
        }
        x[i] = s / l[i * d + i];
    }
    Some(x)
}

impl Readout {
    /// Fit on labelled examples. Deterministic; no random numbers are drawn.
    pub fn fit(examples: &[&Example], cfg: &ReadoutConfig, fingerprint: u64) -> Result<Readout> {
        if examples.is_empty() {
            bail!("no training examples");
        }
        if !(cfg.l2 > 0.0 && cfg.l2.is_finite()) {
            bail!("l2 must be positive and finite, got {}", cfg.l2);
        }
        let all: Vec<&[f64; N_FEATURES]> = examples.iter().flat_map(|e| e.rows.iter()).collect();
        let cnt = all.len() as f64;
        let mut mean = [0.0; N_FEATURES];
        let mut std = [1.0; N_FEATURES];
        for k in 0..N_FEATURES {
            if !cfg.groups.has(FEATURES[k].1) {
                continue; // masked: stays 0 after standardisation
            }
            let mu = all.iter().map(|r| r[k]).sum::<f64>() / cnt;
            let var = all.iter().map(|r| (r[k] - mu).powi(2)).sum::<f64>() / cnt;
            mean[k] = mu;
            std[k] = if var.sqrt() > 1e-9 { var.sqrt() } else { 1.0 };
        }
        let mut xs: Xs = examples.iter().map(|e| standardise(&e.rows, &mean, &std)).collect();
        for rows in xs.iter_mut() {
            for r in rows.iter_mut() {
                for k in 0..N_FEATURES {
                    if !cfg.groups.has(FEATURES[k].1) {
                        r[k] = 0.0;
                    }
                }
            }
        }
        let pos: Vec<usize> = examples.iter().map(|e| e.positive).collect();
        let mut w = [0.0; N_FEATURES];
        let mut f = objective(&xs, &pos, &w, cfg.l2);
        for _ in 0..cfg.max_iter {
            let (g, h) = grad_hess(&xs, &pos, &w, cfg.l2);
            if g.iter().fold(0.0f64, |a, b| a.max(b.abs())) < 1e-9 {
                break;
            }
            let neg_g: Vec<f64> = g.iter().map(|v| -v).collect();
            let step = solve_spd(&h, &neg_g, N_FEATURES).ok_or_else(|| anyhow!("Hessian is not positive definite (should not happen with l2 > 0)"))?;
            let slope: f64 = g.iter().zip(&step).map(|(a, b)| a * b).sum();
            let mut t = 1.0;
            let mut accepted = false;
            for _ in 0..40 {
                let mut w2 = w;
                for k in 0..N_FEATURES {
                    w2[k] += t * step[k];
                }
                let f2 = objective(&xs, &pos, &w2, cfg.l2);
                if f2.is_finite() && f2 <= f + 1e-4 * t * slope {
                    let done = (f - f2).abs() < 1e-13 * (1.0 + f.abs());
                    w = w2;
                    f = f2;
                    accepted = true;
                    if done {
                        t = 0.0;
                    }
                    break;
                }
                t *= 0.5;
            }
            if !accepted || t == 0.0 {
                break;
            }
        }
        if !w.iter().all(|x| x.is_finite()) {
            bail!("training diverged (non-finite weights)");
        }
        Ok(Readout { groups: cfg.groups, l2: cfg.l2, n_train: examples.len(), fingerprint, mean, std, weight: w })
    }

    pub fn score(&self, row: &[f64; N_FEATURES]) -> f64 {
        (0..N_FEATURES).map(|k| self.weight[k] * (row[k] - self.mean[k]) / self.std[k]).sum()
    }

    /// Softmax over the given rows (a candidate pool).
    pub fn probs(&self, rows: &[&[f64; N_FEATURES]]) -> Vec<f64> {
        let z: Vec<f64> = rows.iter().map(|r| self.score(r)).collect();
        let mx = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = z.iter().map(|v| (v - mx).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|v| v / s).collect()
    }

    /// Weight per feature on the standardised scale (comparable across features), largest first.
    pub fn weights(&self) -> Vec<(&'static str, Group, f64)> {
        let mut v: Vec<_> = FEATURES.iter().enumerate().filter(|(k, f)| self.groups.has(f.1) && self.weight[*k] != 0.0).map(|(k, f)| (f.0, f.1, self.weight[k])).collect();
        v.sort_by(|a, b| b.2.abs().total_cmp(&a.2.abs()));
        v
    }

    /// Total absolute standardised weight per source: a rough view of what the model leans on.
    pub fn group_weight(&self) -> [(Group, f64); 3] {
        let mut t = [(Group::Graph, 0.0), (Group::Reservoir, 0.0), (Group::Human, 0.0)];
        for (k, f) in FEATURES.iter().enumerate() {
            let i = match f.1 {
                Group::Graph => 0,
                Group::Reservoir => 1,
                Group::Human => 2,
            };
            t[i].1 += self.weight[k].abs();
        }
        t
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut s = String::from("nexcortex-readout 1\n");
        s.push_str(&format!("fingerprint {:016x}\ngroups {}\nl2 {:?}\nn_train {}\n", self.fingerprint, self.groups.label(), self.l2, self.n_train));
        for (k, f) in FEATURES.iter().enumerate() {
            s.push_str(&format!("feature {} {:?} {:?} {:?}\n", f.0, self.mean[k], self.std[k], self.weight[k]));
        }
        std::fs::write(path.as_ref(), s).with_context(|| format!("writing {:?}", path.as_ref()))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Readout> {
        let text = std::fs::read_to_string(path.as_ref()).with_context(|| format!("reading {:?}", path.as_ref()))?;
        let mut lines = text.lines();
        if lines.next() != Some("nexcortex-readout 1") {
            bail!("{:?} is not a nexcortex readout (bad header)", path.as_ref());
        }
        let (mut fp, mut groups, mut l2, mut n_train) = (None, None, None, None);
        let mut feat: HashMap<String, (f64, f64, f64)> = HashMap::new();
        for line in lines {
            let mut p = line.split_whitespace();
            match p.next() {
                Some("fingerprint") => fp = p.next().and_then(|x| u64::from_str_radix(x, 16).ok()),
                Some("groups") => groups = p.next().map(Groups::parse).transpose()?,
                Some("l2") => l2 = p.next().and_then(|x| x.parse::<f64>().ok()),
                Some("n_train") => n_train = p.next().and_then(|x| x.parse::<usize>().ok()),
                Some("feature") => {
                    let name = p.next().context("feature line without a name")?.to_string();
                    let v: Vec<f64> = p.map(|x| x.parse::<f64>()).collect::<std::result::Result<_, _>>().context("bad number in feature line")?;
                    if v.len() != 3 {
                        bail!("feature {name} needs mean, std, weight");
                    }
                    feat.insert(name, (v[0], v[1], v[2]));
                }
                Some(o) => bail!("unknown line {o:?}"),
                None => {}
            }
        }
        let mut r = Readout {
            groups: groups.context("missing groups")?,
            l2: l2.context("missing l2")?,
            n_train: n_train.context("missing n_train")?,
            fingerprint: fp.context("missing fingerprint")?,
            mean: [0.0; N_FEATURES],
            std: [1.0; N_FEATURES],
            weight: [0.0; N_FEATURES],
        };
        for (k, f) in FEATURES.iter().enumerate() {
            let (m, s, w) = *feat.get(f.0).with_context(|| format!("feature {} missing: readout was written by a different feature set", f.0))?;
            if !(s.is_finite() && s > 0.0 && m.is_finite() && w.is_finite()) {
                bail!("feature {} has non-finite or non-positive statistics", f.0);
            }
            r.mean[k] = m;
            r.std[k] = s;
            r.weight[k] = w;
        }
        if feat.len() != N_FEATURES {
            bail!("readout has {} features, this build has {N_FEATURES}", feat.len());
        }
        Ok(r)
    }
}

// ---------------------------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub top1: f64,
    pub mrr: f64,
    /// Mean log-probability of the chosen entity (learned model only).
    pub mean_logp: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct EvalReport {
    pub n_examples: usize,
    pub mean_pool: f64,
    /// Chance top-1 for the same pools.
    pub chance_top1: f64,
    /// Untrained reservoir readout (the pre-existing rule): rank by post-window salience.
    pub reservoir_rule: Metrics,
    /// Untrained one-shot diffusion.
    pub diffusion_rule: Metrics,
    /// Leave-one-out.
    pub learned_loo: Metrics,
    /// Forward chaining: train on decisions `0..k`, predict decision `k`. `None` if too few examples.
    pub learned_forward: Option<(Metrics, usize)>,
}

fn rank_of(scores: &[f64], pos: usize) -> usize {
    1 + (0..scores.len()).filter(|&j| scores[j] > scores[pos] || (scores[j] == scores[pos] && j < pos)).count()
}

fn metrics<'a>(it: impl Iterator<Item = (usize, Option<f64>)>) -> Metrics {
    let (mut n, mut top1, mut rr, mut lp, mut nlp) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (rank, logp) in it {
        n += 1.0;
        top1 += (rank == 1) as u8 as f64;
        rr += 1.0 / rank as f64;
        if let Some(l) = logp {
            lp += l;
            nlp += 1.0;
        }
    }
    if n == 0.0 {
        return Metrics::default();
    }
    Metrics { top1: top1 / n, mrr: rr / n, mean_logp: (nlp > 0.0).then(|| lp / nlp) }
}

fn learned_rank(model: &Readout, ex: &Example) -> (usize, f64) {
    let rows: Vec<&[f64; N_FEATURES]> = ex.rows.iter().collect();
    let p = model.probs(&rows);
    (rank_of(&p, ex.positive), p[ex.positive].max(1e-300).ln())
}

/// Honest numbers for a training set: the learned readout (leave-one-out and forward-chaining)
/// next to the two untrained rules, on exactly the same pools.
pub fn evaluate(set: &TrainingSet, cfg: &ReadoutConfig) -> Result<EvalReport> {
    let ex = &set.examples;
    if ex.len() < 3 {
        bail!("need at least 3 labelled decisions to evaluate, have {}", ex.len());
    }
    // Pools are in baseline (reservoir-salience) order; an override target outside the shown set is last.
    let reservoir_rule = metrics(ex.iter().map(|e| (e.positive + 1, None)));
    let diffusion_rule = metrics(ex.iter().map(|e| {
        let s: Vec<f64> = e.rows.iter().map(|r| r[DIFFUSION_COL]).collect();
        (rank_of(&s, e.positive), None)
    }));
    let chance = ex.iter().map(|e| 1.0 / e.rows.len() as f64).sum::<f64>() / ex.len() as f64;
    let mut loo = Vec::new();
    for j in 0..ex.len() {
        let train: Vec<&Example> = ex.iter().enumerate().filter(|(i, _)| *i != j).map(|(_, e)| e).collect();
        let m = Readout::fit(&train, cfg, set.fingerprint)?;
        let (r, lp) = learned_rank(&m, &ex[j]);
        loo.push((r, Some(lp)));
    }
    let start = cfg.min_examples.min(ex.len() / 2).max(3);
    let mut fwd = Vec::new();
    for k in start..ex.len() {
        let train: Vec<&Example> = ex[..k].iter().collect();
        let m = Readout::fit(&train, cfg, set.fingerprint)?;
        let (r, lp) = learned_rank(&m, &ex[k]);
        fwd.push((r, Some(lp)));
    }
    let n_fwd = fwd.len();
    Ok(EvalReport {
        n_examples: ex.len(),
        mean_pool: ex.iter().map(|e| e.rows.len() as f64).sum::<f64>() / ex.len() as f64,
        chance_top1: chance,
        reservoir_rule,
        diffusion_rule,
        learned_loo: metrics(loo.into_iter()),
        learned_forward: (n_fwd > 0).then(|| (metrics(fwd.into_iter()), n_fwd)),
    })
}

impl fmt::Display for EvalReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} labelled decisions, mean pool {:.1} candidates (chance top-1 {:.2})", self.n_examples, self.mean_pool, self.chance_top1)?;
        let row = |f: &mut fmt::Formatter<'_>, name: &str, m: &Metrics, note: &str| {
            writeln!(f, "  {name:<34} top-1 {:.2}  MRR {:.2}{}{note}", m.top1, m.mrr, m.mean_logp.map_or(String::new(), |l| format!("  mean log p {l:.2}")))
        };
        row(f, "reservoir rule (untrained)", &self.reservoir_rule, "")?;
        row(f, "diffusion rule (untrained)", &self.diffusion_rule, "")?;
        row(f, "learned readout, leave-one-out", &self.learned_loo, "")?;
        if let Some((m, n)) = &self.learned_forward {
            row(f, "learned readout, forward-chained", m, &format!("  (on the last {n})"))?;
        }
        Ok(())
    }
}

/// Convenience: rebuild the training set from the pipeline's knowledge base, fit and evaluate.
pub fn train(pipe: &Pipeline, cfg: &ReadoutConfig) -> Result<(Readout, EvalReport, TrainingSet)> {
    let set = TrainingSet::from_pipeline(pipe)?;
    if set.examples.len() < cfg.min_examples {
        bail!(
            "only {} labelled decisions can be used for training (need at least {}); skipped: {} without logged evidence, {} reject/defer, {} naming unknown entities, {} choosing evidence",
            set.examples.len(),
            cfg.min_examples,
            set.skipped.legacy,
            set.skipped.no_label,
            set.skipped.unknown_entity,
            set.skipped.chose_evidence
        );
    }
    let all: Vec<&Example> = set.examples.iter().collect();
    let model = Readout::fit(&all, cfg, set.fingerprint)?;
    let report = evaluate(&set, cfg)?;
    Ok((model, report, set))
}

/// FNV-1a over arbitrary bytes (stable across Rust versions, unlike `DefaultHasher`).
pub(crate) fn fnv1a(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}
