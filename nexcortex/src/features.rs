//! Stage 2 — **spectral / diffusion features** from NexGraph.
//!
//! * heat diffusion `exp(-t(I-P)) 1_seed` at several scales (`CALL physics.heat_from`) — how far
//!   and how strongly each piece of evidence reaches every other entity;
//! * a spectral embedding (`CALL spectral.embedding`), spectral clusters (`spectral.cluster`) and
//!   the low end of the Laplacian spectrum (`spectral.eigenvalues`);
//! * PageRank as a structural-importance prior.
//!
//! Heat diffusion is linear in its source, so one query per (evidence node, scale) is enough to
//! superpose any severity weighting.

use crate::kb::KnowledgeBase;
use anyhow::{bail, Result};
use nexgraph::{algo, Id, Value};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct FeatureConfig {
    /// Diffusion times (short = local neighbourhood, long = whole zone).
    pub scales: Vec<f64>,
    /// Mixing weights of the scales in the signal fed to the network (normalised internally).
    pub scale_weights: Vec<f64>,
    pub embed_dim: usize,
    /// Number of spectral clusters (0 = automatic: about one per 6 entities, 2..=8).
    pub clusters: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        FeatureConfig { scales: vec![0.5, 2.0, 8.0], scale_weights: vec![0.5, 0.35, 0.15], embed_dim: 3, clusters: 0 }
    }
}

/// Static, evidence-independent structure features (indexed like `KnowledgeBase::entities`).
#[derive(Clone, Debug)]
pub struct Structure {
    pub embedding: Vec<[f64; 3]>,
    pub cluster: Vec<usize>,
    pub pagerank: Vec<f64>,
    pub eigenvalues: Vec<f64>,
}

/// Diffusion of one evidence item over the graph.
#[derive(Clone, Debug)]
pub struct EventField {
    pub entity: usize,
    pub severity: f64,
    /// `scales x n` raw heat values for a unit source.
    pub heat: Vec<Vec<f64>>,
    /// Scale-mixed field, already multiplied by `severity` (what the network receives), length `n`.
    pub signal: Vec<f64>,
}

fn node_index(kb: &KnowledgeBase) -> HashMap<Id, usize> {
    kb.entities().iter().enumerate().map(|(i, e)| (e.id, i)).collect()
}

pub fn structure(kb: &KnowledgeBase, cfg: &FeatureConfig) -> Result<Structure> {
    let n = kb.entities().len();
    if n < 4 {
        bail!("need at least 4 entities for spectral features, have {n}");
    }
    let comps = kb.components()?;
    if comps != 1 {
        bail!("the knowledge graph has {comps} connected components; spectral features need exactly 1 (link the pieces or analyse one at a time)");
    }
    let idx = node_index(kb);
    let d = cfg.embed_dim.clamp(1, 3);
    let mut embedding = vec![[0.0; 3]; n];
    let r = kb.db.query(&format!("CALL spectral.embedding({d}) YIELD node, embedding RETURN node, embedding"))?;
    for row in &r.rows {
        if let (Value::Node(id), Value::Vector(v)) = (&row[0], &row[1]) {
            if let Some(&i) = idx.get(id) {
                for (k, x) in v.iter().take(3).enumerate() {
                    embedding[i][k] = *x as f64;
                }
            }
        }
    }
    let k = if cfg.clusters == 0 { (n / 6).clamp(2, 8) } else { cfg.clusters };
    let mut cluster = vec![0usize; n];
    let r = kb.db.query(&format!("CALL spectral.cluster({k}) YIELD node, cluster RETURN node, cluster"))?;
    for row in &r.rows {
        if let (Value::Node(id), Some(c)) = (&row[0], row[1].as_i64()) {
            if let Some(&i) = idx.get(id) {
                cluster[i] = c as usize;
            }
        }
    }
    let mut pagerank = vec![0.0; n];
    let (ids, pr) = kb.db.read(|v| algo::pagerank(v, 0.85, 60));
    for (id, s) in ids.iter().zip(pr) {
        if let Some(&i) = idx.get(id) {
            pagerank[i] = s;
        }
    }
    let r = kb.db.query("CALL spectral.eigenvalues(6) YIELD index, eigenvalue RETURN index, eigenvalue")?;
    let eigenvalues = r.rows.iter().filter_map(|x| x[1].as_f64()).collect();
    Ok(Structure { embedding, cluster, pagerank, eigenvalues })
}

/// Diffuse each evidence item `(entity index, severity)` over the graph.
pub fn event_fields(kb: &KnowledgeBase, cfg: &FeatureConfig, events: &[(usize, f64)]) -> Result<Vec<EventField>> {
    if cfg.scales.is_empty() || cfg.scales.len() != cfg.scale_weights.len() {
        bail!("scales and scale_weights must be non-empty and of equal length");
    }
    let n = kb.entities().len();
    let idx = node_index(kb);
    let wsum: f64 = cfg.scale_weights.iter().sum();
    let mut out = Vec::new();
    for &(e, sev) in events {
        let id = kb.entities().get(e).ok_or_else(|| anyhow::anyhow!("entity index {e} out of range"))?.id;
        let mut heat = Vec::new();
        for &t in &cfg.scales {
            let mut p = nexgraph::query::Params::new();
            p.insert("s".into(), Value::List(vec![Value::Node(id)]));
            let r = kb.db.query_with(&format!("CALL physics.heat_from($s, {t}) YIELD node, value RETURN node, value"), &p)?;
            let mut h = vec![0.0; n];
            for row in &r.rows {
                if let (Value::Node(nid), Some(v)) = (&row[0], row[1].as_f64()) {
                    if let Some(&i) = idx.get(nid) {
                        h[i] = v;
                    }
                }
            }
            heat.push(h);
        }
        let mut signal = vec![0.0; n];
        for (h, w) in heat.iter().zip(&cfg.scale_weights) {
            for i in 0..n {
                signal[i] += sev * (w / wsum) * h[i];
            }
        }
        out.push(EventField { entity: e, severity: sev, heat, signal });
    }
    Ok(out)
}
