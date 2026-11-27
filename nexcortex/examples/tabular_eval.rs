//! Run nexcortex on a transaction-similarity graph (see scripts/tabular_prepare.py) for one split.
//!
//! Task: given a few transactions known to have FAILED (evidence), rank held-out transactions by how
//! likely they are to have failed too. The graph uses features only (no labels); training-time
//! decisions come from an oracle over TRAIN labels only; test labels are never read until scoring.
//! Writes the ranked list of TEST nodes per incident and method to the output file; metrics are
//! computed by the Python scorer so every method (including sklearn baselines) is scored identically.
//!
//! usage: tabular_eval <graph_dir> <split> <out.tsv> [train_incidents=60] [test_incidents=20] [evidence=5]

use anyhow::{Context, Result};
use nexcortex::human::Decision;
use nexcortex::readout::{self, Groups, Readout, ReadoutConfig, TrainingSet};
use nexcortex::*;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn pick(&mut self, from: &[usize], k: usize) -> Vec<usize> {
        let mut pool = from.to_vec();
        let mut out = Vec::new();
        for _ in 0..k.min(pool.len()) {
            let j = (self.next() % pool.len() as u64) as usize;
            out.push(pool.swap_remove(j));
        }
        out
    }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let dir = a.get(1).context("graph_dir")?;
    let split: usize = a.get(2).context("split")?.parse()?;
    let outp = a.get(3).context("out file")?;
    let n_train: usize = a.get(4).map_or(Ok(60), |x| x.parse())?;
    let n_test: usize = a.get(5).map_or(Ok(20), |x| x.parse())?;
    let m_ev: usize = a.get(6).map_or(Ok(5), |x| x.parse())?;
    let t0 = Instant::now();

    let nodes: Vec<(String, String, bool)> = std::fs::read_to_string(format!("{dir}/nodes.tsv"))?
        .lines()
        .map(|l| {
            let p: Vec<&str> = l.split('\t').collect();
            (p[0].to_string(), p[1].to_string(), p[2] == "1")
        })
        .collect();
    let n = nodes.len();
    let train: Vec<bool> = std::fs::read_to_string(format!("{dir}/splits.tsv"))?.lines().map(|l| l.split('\t').nth(1 + split).unwrap() == "1").collect();
    let mut ties: Vec<(usize, usize, f64)> = std::fs::read_to_string(format!("{dir}/edges.tsv"))?
        .lines()
        .map(|l| {
            let p: Vec<&str> = l.split('\t').collect();
            (p[0].parse().unwrap(), p[1].parse().unwrap(), p[2].parse().unwrap())
        })
        .collect();
    // spectral stages need one connected component
    let mut uf: Vec<usize> = (0..n).collect();
    fn find(u: &mut Vec<usize>, x: usize) -> usize {
        let mut r = x;
        while u[r] != r {
            r = u[r];
        }
        let mut c = x;
        while u[c] != r {
            let nx = u[c];
            u[c] = r;
            c = nx;
        }
        r
    }
    for &(i, j, _) in &ties {
        let (x, y) = (find(&mut uf, i), find(&mut uf, j));
        uf[x] = y;
    }
    let mut bridged = 0;
    for i in 1..n {
        let (r, r0) = (find(&mut uf, i), find(&mut uf, 0));
        if r != r0 {
            ties.push((0, i, 0.02));
            uf[r] = r0;
            bridged += 1;
        }
    }
    let mut b = KbBuilder::new();
    for nd in &nodes {
        b = b.entity(&nd.0, &nd.1);
    }
    for &(i, j, w) in &ties {
        b = b.relate(&nodes[i].0, &nodes[j].0, "similar", w);
    }
    let kb = b.build()?;
    let cfg = PipelineConfig { top_k: 10, rerank_pool: n, prior_weight: 0.0, ..PipelineConfig::default() };
    let t = Instant::now();
    let mut pipe = Pipeline::new(kb, cfg)?;
    eprintln!("split {split}: {n} nodes, {} ties (+{bridged} bridging), pipeline built in {:.1}s", ties.len(), t.elapsed().as_secs_f64());
    let ent: Vec<usize> = nodes.iter().map(|x| pipe.kb.index_of(pipe.kb.entity(&x.0).unwrap().id).unwrap()).collect();
    let tech_of: HashMap<usize, usize> = ent.iter().enumerate().map(|(i, &e)| (e, i)).collect();

    let train_failed: Vec<usize> = (0..n).filter(|&i| train[i] && nodes[i].2).collect();
    let test_nodes: HashSet<usize> = (0..n).filter(|&i| !train[i]).collect();
    let test_failed = test_nodes.iter().filter(|&&i| nodes[i].2).count();
    eprintln!("train failed {}, test nodes {} (failed {})", train_failed.len(), test_nodes.len(), test_failed);
    let query = |ev: &[usize]| Query {
        question: "Which other transactions are likely to have failed too?".into(),
        events: ev.iter().map(|&i| EventSpec { entity: nodes[i].0.clone(), severity: 0.8, note: "known failed".into() }).collect(),
    };

    // ---- oracle decisions over TRAIN labels only ----
    let t = Instant::now();
    let mut rng = Lcg(1000 + split as u64);
    let (mut approve, mut over) = (0, 0);
    for _ in 0..n_train {
        let ev = rng.pick(&train_failed, m_ev);
        let hidden: Vec<usize> = train_failed.iter().copied().filter(|x| !ev.contains(x)).collect();
        let an = pipe.analyze(&query(&ev))?;
        let hs: HashSet<&str> = hidden.iter().map(|&i| nodes[i].0.as_str()).collect();
        let d = match an.brief.candidates.iter().find(|c| hs.contains(c.entity.as_str())) {
            Some(c) => {
                approve += 1;
                Decision::Approve { entity: c.entity.clone(), note: "oracle".into() }
            }
            None => {
                over += 1;
                let h = hidden[(rng.next() % hidden.len() as u64) as usize];
                Decision::Override { entity: nodes[h].0.clone(), note: "oracle".into() }
            }
        };
        pipe.record(&an.brief, &d, "oracle")?;
    }
    eprintln!("{n_train} decisions logged ({approve} approve, {over} override) in {:.0}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let set = TrainingSet::from_pipeline(&pipe)?;
    let rc = ReadoutConfig::default();
    let rep = readout::evaluate(&set, &rc)?;
    eprintln!("replayed {} examples in {:.0}s\n{rep}", set.examples.len(), t.elapsed().as_secs_f64());
    let mut models: Vec<(&str, Readout)> = Vec::new();
    for (label, g) in [("learned_all", "all"), ("learned_graph", "graph"), ("learned_reservoir", "reservoir"), ("learned_human", "human")] {
        let c = ReadoutConfig { groups: Groups::parse(g)?, ..ReadoutConfig::default() };
        let all: Vec<&readout::Example> = set.examples.iter().collect();
        models.push((label, Readout::fit(&all, &c, set.fingerprint)?));
    }
    eprintln!("group weight (sum |w|, all): {:?}", models[0].1.group_weight().iter().map(|(g, w)| format!("{g:?}={w:.2}")).collect::<Vec<_>>());

    // ---- held-out scoring: rank TEST nodes only ----
    let mut out = std::fs::File::create(outp)?;
    writeln!(out, "# split {split}; failed in test {test_failed}; rep {}", rep.to_string().replace('\n', " | "))?;
    let mut rng = Lcg(5000 + split as u64);
    let t = Instant::now();
    for inc in 0..n_test {
        let ev = rng.pick(&train_failed, m_ev);
        writeln!(out, "E\t{split}\t{inc}\t{}", ev.iter().map(|i| nodes[*i].0.clone()).collect::<Vec<_>>().join(" "))?;
        let q = query(&ev);
        let emit = |out: &mut std::fs::File, name: &str, order: &[usize]| -> Result<()> {
            let ids: Vec<&str> = order.iter().filter_map(|e| tech_of.get(e)).filter(|&&i| !train[i]).map(|&i| nodes[i].0.as_str()).collect();
            writeln!(out, "R\t{split}\t{inc}\t{name}\t{}", ids.join(" "))?;
            Ok(())
        };
        pipe.clear_readout();
        let base = pipe.analyze(&q)?;
        emit(&mut out, "reservoir_rule", &base.baseline_order)?;
        let mut by_diff = base.baseline_order.clone();
        by_diff.sort_by(|&a, &b| base.diffusion[b].total_cmp(&base.diffusion[a]).then(a.cmp(&b)));
        emit(&mut out, "diffusion_rule", &by_diff)?;
        for (name, m) in &models {
            pipe.set_readout(m.clone())?;
            let an = pipe.analyze(&q)?;
            emit(&mut out, name, &an.final_order)?;
        }
    }
    eprintln!("scored {n_test} test incidents in {:.0}s; total {:.0}s", t.elapsed().as_secs_f64(), t0.elapsed().as_secs_f64());
    Ok(())
}
