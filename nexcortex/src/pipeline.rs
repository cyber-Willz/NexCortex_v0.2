//! The full architecture, wired end to end:
//!
//! ```text
//! NexGraph (persistent knowledge)
//!   -> spectral / diffusion features
//!   -> connectome RNN (working memory + reasoning)
//!   -> LLM explanation layer
//!   -> human decision maker  --(decision recorded back into NexGraph)
//! ```

use crate::brief::{Brief, Candidate, Evidence, MemorySummary};
use crate::explain::{Explainer, Explanation};
use crate::features::{self, EventField, FeatureConfig, Structure};
use crate::human::{Decision, DecisionMaker};
use crate::kb::KnowledgeBase;
use crate::memory::{MemoryConfig, MemoryTrace, WorkingMemory};
use crate::readout::{self, FeatureTable, HumanHistory, Inputs, Readout};
use anyhow::{anyhow, bail, Result};
use nexgraph::Id;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub features: FeatureConfig,
    pub memory: MemoryConfig,
    /// How many candidates to put in front of the human.
    pub top_k: usize,
    /// Severity given to an entity an operator confirmed 3+ times (scaled down for fewer).
    pub prior_weight: f64,
    /// With a learned readout attached: how many of the reservoir's best candidates it may re-rank
    /// (>= `top_k`). Entities outside this pool are never shown.
    pub rerank_pool: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig { features: FeatureConfig::default(), memory: MemoryConfig::default(), top_k: 5, prior_weight: 0.3, rerank_pool: 10 }
    }
}

#[derive(Clone, Debug)]
pub struct EventSpec {
    pub entity: String,
    /// In (0, 1].
    pub severity: f64,
    pub note: String,
}

#[derive(Clone, Debug)]
pub struct Query {
    pub question: String,
    /// Evidence in the order it was observed (earliest first).
    pub events: Vec<EventSpec>,
}

pub struct Analysis {
    pub brief: Brief,
    pub structure: Structure,
    pub trace: MemoryTrace,
    pub fields: Vec<EventField>,
    /// Per-entity one-shot diffusion score (sum of severity-weighted scale-mixed heat).
    pub diffusion: Vec<f64>,
    /// Readout features (graph + reservoir trajectory + human history) for every non-evidence entity.
    pub table: FeatureTable,
    /// Non-evidence entities by the fixed reservoir rule (post-window salience), best first.
    pub baseline_order: Vec<usize>,
    /// Final ranking of every non-evidence entity (equals `baseline_order` without a readout).
    pub final_order: Vec<usize>,
}

pub struct Outcome {
    pub analysis: Analysis,
    pub explanation: Explanation,
    pub decision: Decision,
    pub decision_node: Id,
}

pub struct Pipeline {
    pub kb: KnowledgeBase,
    pub cfg: PipelineConfig,
    structure: Structure,
    memory: WorkingMemory,
    /// Dense `REL` adjacency (n x n), used for explanation paths.
    adj: Vec<f64>,
    readout: Option<Readout>,
}

impl Pipeline {
    pub fn new(kb: KnowledgeBase, cfg: PipelineConfig) -> Result<Pipeline> {
        let structure = features::structure(&kb, &cfg.features)?;
        let systems: Vec<String> = structure.cluster.iter().map(|c| format!("C{c}")).collect();
        let pos: Vec<[f64; 3]> = structure.embedding.iter().map(|p| [p[0] * 100.0, p[1] * 100.0, p[2] * 100.0]).collect();
        let conn = kb.to_connectome(&pos, &systems)?;
        let memory = WorkingMemory::new(&conn, cfg.memory.clone())?;
        let adj = kb.adjacency()?;
        if cfg.rerank_pool < cfg.top_k {
            bail!("rerank_pool ({}) must be at least top_k ({})", cfg.rerank_pool, cfg.top_k);
        }
        Ok(Pipeline { kb, cfg, structure, memory, adj, readout: None })
    }

    /// Identifies the frozen reservoir + feature setup a readout was trained against: memory and
    /// feature config, entity names and the weighted adjacency. A readout trained for another
    /// fingerprint is refused, because its weights mean nothing on a different reservoir.
    pub fn fingerprint(&self) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        readout::fnv1a(&mut h, format!("{:?}|{:?}|prior={:?}|top_k={}", self.cfg.memory, self.cfg.features, self.cfg.prior_weight, self.cfg.top_k).as_bytes());
        for e in self.kb.entities() {
            readout::fnv1a(&mut h, e.name.as_bytes());
            readout::fnv1a(&mut h, &[0]);
        }
        for w in &self.adj {
            readout::fnv1a(&mut h, &w.to_bits().to_le_bytes());
        }
        h
    }

    /// Attach a trained readout: from now on [`Self::analyze`] ranks with it.
    pub fn set_readout(&mut self, r: Readout) -> Result<()> {
        if r.fingerprint != self.fingerprint() {
            bail!("this readout was trained for a different reservoir / knowledge graph / feature configuration (fingerprint mismatch); retrain it");
        }
        self.readout = Some(r);
        Ok(())
    }
    pub fn clear_readout(&mut self) {
        self.readout = None;
    }
    pub fn readout(&self) -> Option<&Readout> {
        self.readout.as_ref()
    }
    /// Dense `REL` adjacency (n x n).
    pub fn adjacency(&self) -> &[f64] {
        &self.adj
    }
    /// Decisions made so far, resolved to entity indices (what the human features look at).
    pub fn human_history(&self) -> Result<HumanHistory> {
        let (log, _) = self.kb.decision_log()?;
        Ok(HumanHistory::from_records(&self.kb, &log))
    }

    pub fn structure(&self) -> &Structure {
        &self.structure
    }
    pub fn memory(&self) -> &WorkingMemory {
        &self.memory
    }

    /// Stages 1-3 and the brief: knowledge -> features -> working memory -> ranked candidates.
    /// Uses the attached readout if there is one, otherwise the fixed reservoir rule.
    pub fn analyze(&self, q: &Query) -> Result<Analysis> {
        let history = if self.readout.is_some() { self.human_history()? } else { HumanHistory::default() };
        self.analyze_with(q, &self.kb.confirmed()?, &history, self.readout.as_ref())
    }

    /// Order the non-evidence entities: by reservoir salience, then (with a readout) re-rank the
    /// best `rerank_pool` of them by the learned score. Returns (baseline order, final order, learned prob by entity).
    fn rank(&self, inp: &Inputs, readout: Option<&Readout>) -> (Vec<usize>, Vec<usize>, HashMap<usize, f64>, FeatureTable) {
        let table = readout::extract(inp);
        let sal = &inp.trace.salience;
        let mut baseline = table.entities.clone();
        baseline.sort_by(|&a, &b| sal[b].total_cmp(&sal[a]).then(a.cmp(&b)));
        let mut order = baseline.clone();
        let mut prob = HashMap::new();
        if let Some(r) = readout {
            let pool = self.cfg.rerank_pool.min(baseline.len());
            let rows: Vec<&[f64; readout::N_FEATURES]> = baseline[..pool].iter().map(|&e| table.row_of(e).expect("candidate row")).collect();
            let p = r.probs(&rows);
            let mut idx: Vec<usize> = (0..pool).collect();
            idx.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
            order = idx.iter().map(|&k| baseline[k]).chain(baseline[pool..].iter().copied()).collect();
            for (k, &e) in baseline[..pool].iter().enumerate() {
                prob.insert(e, p[k]);
            }
        }
        (baseline, order, prob, table)
    }

    /// [`Self::analyze`] with the "past" given explicitly: `priors` are the confirmation counts to
    /// seed as weak evidence and `history` is what the human features may look at. Training replays
    /// each logged decision with only the state that existed before it, so a label never leaks into
    /// its own features.
    pub fn analyze_with(&self, q: &Query, priors: &[(String, i64)], history: &HumanHistory, readout: Option<&Readout>) -> Result<Analysis> {
        if q.events.is_empty() {
            bail!("a query needs at least one evidence item");
        }
        let n = self.kb.entities().len();
        let mut seeds: Vec<(usize, f64, String, bool)> = Vec::new();
        // Persistent knowledge first: what operators confirmed before is a weak standing prior.
        let mut user_idx = Vec::new();
        for ev in &q.events {
            let e = self.kb.entity(&ev.entity).ok_or_else(|| anyhow!("unknown entity {:?}", ev.entity))?;
            if !(ev.severity > 0.0 && ev.severity <= 1.0) {
                bail!("severity of {} must be in (0, 1], got {}", ev.entity, ev.severity);
            }
            user_idx.push(self.kb.index_of(e.id).unwrap());
        }
        for (name, count) in priors {
            let Some(e) = self.kb.entity(name) else { continue };
            let i = self.kb.index_of(e.id).unwrap();
            if !user_idx.contains(&i) {
                let sev = (self.cfg.prior_weight * ((*count).min(3) as f64) / 3.0).clamp(0.0, 1.0);
                if sev > 0.0 {
                    seeds.push((i, sev, format!("confirmed {count}x in earlier decisions"), true));
                }
            }
        }
        for (ev, &i) in q.events.iter().zip(&user_idx) {
            seeds.push((i, ev.severity, ev.note.clone(), false));
        }

        let fields = features::event_fields(&self.kb, &self.cfg.features, &seeds.iter().map(|s| (s.0, s.1)).collect::<Vec<_>>())?;
        let trace = self.memory.run(&fields)?;
        let mut diffusion = vec![0.0; n];
        for f in &fields {
            for i in 0..n {
                diffusion[i] += f.signal[i];
            }
        }
        let all_ev: Vec<(usize, f64)> = seeds.iter().map(|s| (s.0, s.1)).collect();
        let user_ev: Vec<(usize, f64)> = user_idx.iter().zip(&q.events).map(|(&i, e)| (i, e.severity)).collect();
        let (baseline, order, prob, table) = self.rank(
            &Inputs { structure: &self.structure, adj: &self.adj, n, evidence: &all_ev, user_evidence: &user_ev, fields: &fields, trace: &trace, diffusion: &diffusion, history },
            readout,
        );
        // The memory fades, so evidence order matters. Replay it reversed and see if the answer moves.
        let order_stable = if fields.len() > 1 {
            let rev: Vec<EventField> = fields.iter().rev().cloned().collect();
            let t2 = self.memory.run(&rev)?;
            let (_, order2, _, _) = self.rank(
                &Inputs { structure: &self.structure, adj: &self.adj, n, evidence: &all_ev, user_evidence: &user_ev, fields: &rev, trace: &t2, diffusion: &diffusion, history },
                readout,
            );
            Some(order.first() == order2.first())
        } else {
            None
        };

        let evidence: Vec<Evidence> = seeds
            .iter()
            .map(|s| Evidence { entity: self.kb.entities()[s.0].name.clone(), kind: self.kb.entities()[s.0].kind.clone(), severity: s.1, note: s.2.clone(), from_prior: s.3 })
            .collect();
        let cand = &order;
        if cand.is_empty() {
            bail!("every entity is evidence; nothing to rank");
        }
        let mut by_diff = baseline.clone();
        by_diff.sort_by(|&a, &b| diffusion[b].total_cmp(&diffusion[a]).then(a.cmp(&b)));
        // Salience is relative to the reservoir's own best candidate, whatever the final order is.
        let s_max = trace.salience[baseline[0]];
        let d_max = diffusion[by_diff[0]];
        let confirmed: HashMap<String, i64> = priors.iter().cloned().collect();

        let ev_idx: Vec<usize> = seeds.iter().map(|s| s.0).collect();
        let mut candidates = Vec::new();
        for (r, &i) in cand.iter().take(self.cfg.top_k).enumerate() {
            let path = self.strongest_path(&ev_idx, i);
            let name = self.kb.entities()[i].name.clone();
            candidates.push(Candidate {
                rank: r + 1,
                entity: name.clone(),
                kind: self.kb.entities()[i].kind.clone(),
                cluster: self.structure.cluster[i],
                salience: if s_max > 0.0 { (trace.salience[i] / s_max).max(0.0) } else { 0.0 },
                diffusion: if d_max > 0.0 { diffusion[i] / d_max } else { 0.0 },
                diffusion_rank: by_diff.iter().position(|&x| x == i).unwrap() + 1,
                pagerank: self.structure.pagerank[i],
                path,
                times_confirmed_before: confirmed.get(&name).copied().unwrap_or(0),
                learned: prob.get(&i).copied(),
                baseline_rank: baseline.iter().position(|&x| x == i).unwrap() + 1,
            });
        }

        let mut flags = Vec::new();
        let top = &candidates[0];
        // Margin between the top two: learned probability when a readout ranked them, else salience.
        let margin = match (candidates.get(1), top.learned) {
            (Some(c2), Some(p1)) => (p1 - c2.learned.unwrap_or(0.0)).clamp(0.0, 1.0),
            (Some(c2), None) => 1.0 - c2.salience,
            (None, _) => 1.0,
        };
        let agree = match top.diffusion_rank {
            1 | 2 => 1.0,
            3 => 0.5,
            _ => 0.2,
        };
        let retention = trace.retention.clamp(0.0, 1.0);
        let mut confidence = (0.40 * (margin / 0.5).min(1.0) + 0.35 * agree + 0.25 * (retention / 0.2).min(1.0)).clamp(0.0, 1.0);
        if order_stable == Some(false) {
            confidence *= 0.8;
            flags.push("the top candidate changes if the same evidence is presented in reverse order (working memory favours the most recent evidence)".into());
        }
        if q.events.len() == 1 {
            flags.push("only one evidence item: the ranking mostly reflects graph proximity".into());
        }
        if margin < 0.10 {
            flags.push(format!("the top two candidates are nearly tied ({} vs {})", candidates[0].entity, candidates[1].entity));
        }
        if top.diffusion_rank > 3 {
            flags.push(format!("working memory and plain diffusion disagree strongly on {} (diffusion rank {})", top.entity, top.diffusion_rank));
        }
        if retention < 0.05 {
            flags.push("working memory kept under 5% of its peak activity over the readout window; the ranking rests on a weak signal".into());
        }
        if trace.spectral_radius >= 1.0 {
            flags.push(format!("recurrent spectral radius {:.2} >= 1: dynamics may not be stable", trace.spectral_radius));
        }
        if top.path.is_empty() {
            flags.push(format!("no knowledge-graph path links {} to any evidence entity", top.entity));
        }
        if s_max <= 0.0 {
            flags.push("no entity has positive working-memory activity".into());
        }
        if let Some(r) = readout {
            flags.push(format!(
                "ranked by a learned readout ({}; trained on {} human decisions), not the fixed reservoir rule; its probabilities are relative within the top {} reservoir candidates and not calibrated",
                r.groups.label(),
                r.n_train,
                self.cfg.rerank_pool.min(baseline.len())
            ));
            if top.baseline_rank != 1 {
                flags.push(format!("the learned readout promoted {} (reservoir rank {}) over the reservoir's own pick {}", top.entity, top.baseline_rank, self.kb.entities()[baseline[0]].name));
            }
        }

        let brief = Brief {
            question: q.question.clone(),
            evidence,
            candidates,
            confidence,
            flags,
            memory: MemorySummary {
                retention: trace.retention,
                half_life: trace.half_life,
                spectral_radius: trace.spectral_radius,
                steps: trace.steps,
                n_units: trace.n_units,
                n_connections: trace.n_connections,
            },
            n_entities: n,
            kb_timestamp: self.kb.db.last_ts(),
        };
        Ok(Analysis { brief, structure: self.structure.clone(), trace, fields, diffusion, table, baseline_order: baseline, final_order: order })
    }

    /// Chain of `REL` ties from the nearest evidence entity to `to`, where "nearest" minimises the
    /// sum of `1/weight` (so strong ties are short). Empty if unreachable.
    fn strongest_path(&self, from: &[usize], to: usize) -> Vec<String> {
        let n = self.kb.entities().len();
        let (mut dist, mut prev, mut done) = (vec![f64::INFINITY; n], vec![usize::MAX; n], vec![false; n]);
        dist[to] = 0.0;
        for _ in 0..n {
            let Some(u) = (0..n).filter(|&i| !done[i] && dist[i].is_finite()).min_by(|&a, &b| dist[a].total_cmp(&dist[b])) else { break };
            done[u] = true;
            for v in 0..n {
                let w = self.adj[u * n + v];
                if w > 0.0 && dist[u] + 1.0 / w < dist[v] {
                    dist[v] = dist[u] + 1.0 / w;
                    prev[v] = u;
                }
            }
        }
        let Some(&src) = from.iter().filter(|&&f| dist[f].is_finite()).min_by(|&&a, &&b| dist[a].total_cmp(&dist[b])) else { return vec![] };
        let mut path = vec![src];
        let mut cur = src;
        while cur != to {
            cur = prev[cur];
            path.push(cur);
        }
        path.iter().map(|&i| self.kb.entities()[i].name.clone()).collect()
    }

    /// The whole architecture: analyse, explain, ask the human, record the decision.
    pub fn run(&mut self, q: &Query, explainer: &dyn Explainer, human: &mut dyn DecisionMaker) -> Result<Outcome> {
        let analysis = self.analyze(q)?;
        let explanation = explainer.explain(&analysis.brief)?;
        let decision = human.decide(&analysis.brief, &explanation)?;
        let node = self.record(&analysis.brief, &decision, &human.operator())?;
        Ok(Outcome { analysis, explanation, decision, decision_node: node })
    }

    /// Validate and persist a human decision (also usable without `run`).
    pub fn record(&mut self, brief: &Brief, d: &Decision, operator: &str) -> Result<Id> {
        let top = brief.candidates.first().map(|c| c.entity.clone());
        let (entities, confirm, reject): (Vec<String>, Vec<String>, Vec<String>) = match d {
            Decision::Approve { entity, .. } => {
                if !brief.candidates.iter().any(|c| &c.entity == entity) {
                    bail!("{entity:?} is not one of the candidates in this brief; use override to name another entity");
                }
                (vec![entity.clone()], vec![entity.clone()], vec![])
            }
            Decision::Override { entity, .. } => {
                if self.kb.entity(entity).is_none() {
                    bail!("override names unknown entity {entity:?}");
                }
                (vec![entity.clone()], vec![entity.clone()], top.into_iter().collect())
            }
            Decision::Reject { .. } => (vec![], vec![], top.into_iter().collect()),
            Decision::Defer { .. } => (vec![], vec![], vec![]),
        };
        let evidence: Vec<(String, f64)> = brief.evidence.iter().filter(|e| !e.from_prior).map(|e| (e.entity.clone(), e.severity)).collect();
        self.kb.record_decision_logged(d.kind(), &brief.question, operator, d.note(), brief.confidence, &entities, &confirm, &reject, &evidence)
    }
}
