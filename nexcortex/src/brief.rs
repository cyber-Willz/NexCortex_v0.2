//! The structured hand-off between the machine stages and the explanation / human stages.
//! Everything the LLM may say and everything the human sees is derived from this struct.

use std::fmt::Write;

#[derive(Clone, Debug)]
pub struct Evidence {
    pub entity: String,
    pub kind: String,
    pub severity: f64,
    pub note: String,
    /// `true` when this seed came from an earlier human confirmation stored in the knowledge base.
    pub from_prior: bool,
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub rank: usize,
    pub entity: String,
    pub kind: String,
    pub cluster: usize,
    /// Working-memory state, scaled so the top candidate is 1.0.
    pub salience: f64,
    /// One-shot multi-scale heat diffusion score, scaled so the top candidate is 1.0.
    pub diffusion: f64,
    /// Rank by diffusion alone among the same (non-evidence) entities, 1-based.
    pub diffusion_rank: usize,
    pub pagerank: f64,
    /// Shortest path in the knowledge graph from the nearest evidence entity (inclusive).
    pub path: Vec<String>,
    pub times_confirmed_before: i64,
    /// Probability the learned readout gives this candidate among the re-ranked pool (None = fixed rule).
    pub learned: Option<f64>,
    /// Rank under the fixed reservoir rule alone (1 = the reservoir's own top pick).
    pub baseline_rank: usize,
}

#[derive(Clone, Debug)]
pub struct MemorySummary {
    pub retention: f64,
    pub half_life: Option<usize>,
    pub spectral_radius: f64,
    pub steps: usize,
    pub n_units: usize,
    pub n_connections: usize,
}

#[derive(Clone, Debug)]
pub struct Brief {
    pub question: String,
    pub evidence: Vec<Evidence>,
    pub candidates: Vec<Candidate>,
    /// Heuristic in [0,1] built from rank margin, agreement between diffusion and working memory
    /// and retention. It is **not** a calibrated probability.
    pub confidence: f64,
    pub flags: Vec<String>,
    pub memory: MemorySummary,
    pub n_entities: usize,
    pub kb_timestamp: u64,
}

impl Brief {
    /// Plain-text fact sheet: the only context the LLM gets, and what the human reads next to the narrative.
    pub fn facts(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "QUESTION: {}", self.question);
        let _ = writeln!(s, "EVIDENCE:");
        for (i, e) in self.evidence.iter().enumerate() {
            let _ = writeln!(
                s,
                "  {}. {} ({}) severity {:.2}{}{}",
                i + 1,
                e.entity,
                e.kind,
                e.severity,
                if e.from_prior { " [previously confirmed by an operator]" } else { "" },
                if e.note.is_empty() { String::new() } else { format!(" - {}", e.note) }
            );
        }
        let learned = self.candidates.iter().any(|c| c.learned.is_some());
        let _ = writeln!(
            s,
            "CANDIDATES (ranked by {}, evidence entities excluded):",
            if learned { "a learned readout over graph features, reservoir trajectory and human decisions" } else { "working-memory salience" }
        );
        for c in &self.candidates {
            let path = if c.path.is_empty() { "no path".to_string() } else { c.path.join(" -> ") };
            let _ = writeln!(
                s,
                "  {}. {} ({}, community {}) salience {:.2}, diffusion {:.2} (diffusion-only rank {}), pagerank {:.3}, path: {}{}",
                c.rank,
                c.entity,
                c.kind,
                c.cluster,
                c.salience,
                c.diffusion,
                c.diffusion_rank,
                c.pagerank,
                path,
                if c.times_confirmed_before > 0 { format!(", confirmed before {}x", c.times_confirmed_before) } else { String::new() }
            );
            if let Some(p) = c.learned {
                let _ = writeln!(s, "       learned score {:.2} (reservoir-only rank {})", p, c.baseline_rank);
            }
        }
        let _ = writeln!(
            s,
            "WORKING MEMORY: retention {:.2}, half-life {}, spectral radius {:.2}, {} steps",
            self.memory.retention,
            self.memory.half_life.map_or("beyond window".into(), |h| format!("{h} steps")),
            self.memory.spectral_radius,
            self.memory.steps
        );
        let _ = writeln!(s, "CONFIDENCE (heuristic, not a probability): {:.2}", self.confidence);
        if !self.flags.is_empty() {
            let _ = writeln!(s, "CAVEATS:");
            for f in &self.flags {
                let _ = writeln!(s, "  - {f}");
            }
        }
        s
    }
}
