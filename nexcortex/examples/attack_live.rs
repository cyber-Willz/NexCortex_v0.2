//! End-to-end run on real data: MITRE ATT&CK Enterprise (fetched live by scripts/attack_prepare.py).
//!
//! Task: an adversary has been observed using a few techniques (evidence, in kill-chain order).
//! Which *other* techniques is it documented to use, i.e. what should be hunted for next?
//! Ground truth = the technique set a real intrusion group is documented to use.
//!
//! Protocol (no leakage):
//!  * groups are split train / test by a hash of their name;
//!  * the knowledge graph (technique-technique ties) is built from TRAIN groups only;
//!  * the operator is an ORACLE built from the real ground truth (there are no real operator logs):
//!    it approves the first shown candidate that the group really uses, else overrides to the most
//!    popular technique the group really uses. Decisions are logged through the real pipeline;
//!  * the readout is trained on those logged decisions only;
//!  * test incidents (held-out groups) are never recorded and never touch the graph.
//!
//! usage: attack_live <data_dir> [max_groups_per_side]

use anyhow::{bail, Context, Result};
use nexcortex::human::{self, Decision};
use nexcortex::readout::{self, Groups, Readout, ReadoutConfig, TrainingSet};
use nexcortex::*;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

struct Tech {
    id: String,
    name: String,
    order: usize,
    tactics: String,
}

struct Incident {
    group: String,
    evidence: Vec<String>,
    hidden: Vec<String>,
}

fn fnv(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn load(dir: &str) -> Result<(Vec<Tech>, Vec<(String, Vec<String>)>)> {
    let t = std::fs::read_to_string(format!("{dir}/techniques.tsv")).context("techniques.tsv (run scripts/attack_prepare.py)")?;
    let techs = t
        .lines()
        .map(|l| {
            let p: Vec<&str> = l.split('\t').collect();
            Tech { id: p[0].into(), name: p[1].into(), order: p[2].parse().unwrap(), tactics: p[3].into() }
        })
        .collect();
    let g = std::fs::read_to_string(format!("{dir}/groups.tsv")).context("groups.tsv")?;
    let groups = g
        .lines()
        .map(|l| {
            let (n, ts) = l.split_once('\t').unwrap();
            (n.to_string(), ts.split(',').map(String::from).collect())
        })
        .collect();
    Ok((techs, groups))
}

#[derive(Clone, Copy, Default)]
struct M {
    hit5: f64,
    p5: f64,
    hit10: f64,
    mrr: f64,
    ndcg10: f64,
}
const NM: usize = 5;
impl M {
    fn arr(&self) -> [f64; NM] {
        [self.hit5, self.p5, self.hit10, self.mrr, self.ndcg10]
    }
}
const MNAMES: [&str; NM] = ["hit@5", "precision@5", "hit@10", "MRR", "nDCG@10"];

fn metrics(order: &[usize], hidden: &HashSet<usize>) -> M {
    let rel = |i: usize| hidden.contains(&order[i]) as u8 as f64;
    let k = |k: usize| (0..k.min(order.len())).map(rel).sum::<f64>();
    let first = (0..order.len()).find(|&i| rel(i) > 0.0);
    let dcg: f64 = (0..10.min(order.len())).map(|i| rel(i) / (i as f64 + 2.0).log2()).sum();
    let ideal: f64 = (0..10.min(hidden.len())).map(|i| 1.0 / (i as f64 + 2.0).log2()).sum();
    M { hit5: (k(5) > 0.0) as u8 as f64, p5: k(5) / 5.0, hit10: (k(10) > 0.0) as u8 as f64, mrr: first.map_or(0.0, |i| 1.0 / (i as f64 + 1.0)), ndcg10: dcg / ideal }
}

/// Exact expected metrics of a uniformly random ranking of `m` candidates containing `h` relevant ones.
fn random_metrics(m: usize, h: usize) -> M {
    let ln_c = |n: usize, k: usize| -> f64 { (0..k).map(|i| ((n - i) as f64).ln() - ((i + 1) as f64).ln()).sum() };
    let miss = |k: usize| if m - h >= k { (ln_c(m - h, k) - ln_c(m, k)).exp() } else { 0.0 };
    let mut mrr = 0.0;
    for r in 1..=(m - h + 1) {
        let p_prev_all_miss = miss(r - 1);
        mrr += (1.0 / r as f64) * p_prev_all_miss * h as f64 / (m - r + 1) as f64;
    }
    let ideal: f64 = (0..10.min(h)).map(|i| 1.0 / (i as f64 + 2.0).log2()).sum();
    let disc: f64 = (0..10).map(|i| 1.0 / (i as f64 + 2.0).log2()).sum();
    M { hit5: 1.0 - miss(5), p5: h as f64 / m as f64, hit10: 1.0 - miss(10), mrr, ndcg10: (h as f64 / m as f64) * disc / ideal }
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).context("usage: attack_live <data_dir> [max_groups_per_side]")?;
    let cap: usize = std::env::args().nth(2).map(|x| x.parse()).transpose()?.unwrap_or(usize::MAX);
    let t0 = Instant::now();
    let (techs, groups) = load(&dir)?;
    let idx: HashMap<&str, usize> = techs.iter().enumerate().map(|(i, t)| (t.id.as_str(), i)).collect();
    println!("data: {} techniques, {} groups", techs.len(), groups.len());

    // ---- split by group ----
    let salt = std::env::var("ATTACK_SPLIT_SALT").unwrap_or_default();
    println!("split salt: {:?}", salt);
    let is_train = |g: &str| fnv(&format!("{salt}{g}")) % 10 < 7;
    let train_g: Vec<&(String, Vec<String>)> = groups.iter().filter(|g| is_train(&g.0)).collect();
    let test_g: Vec<&(String, Vec<String>)> = groups.iter().filter(|g| !is_train(&g.0)).collect();
    println!("split: {} train groups, {} held-out groups", train_g.len(), test_g.len());

    // ---- knowledge graph from TRAIN groups only ----
    let n = techs.len();
    let mut users: Vec<HashSet<usize>> = vec![HashSet::new(); n]; // technique -> train groups using it
    for (gi, g) in train_g.iter().enumerate() {
        for t in &g.1 {
            users[idx[t.as_str()]].insert(gi);
        }
    }
    let pop: Vec<usize> = users.iter().map(|u| u.len()).collect();
    let mut ties: Vec<(usize, usize, f64)> = Vec::new();
    let tac: Vec<HashSet<&str>> = techs.iter().map(|t| t.tactics.split(',').collect()).collect();
    for i in 0..n {
        for j in i + 1..n {
            let co = users[i].intersection(&users[j]).count() as f64;
            let cos = if co > 0.0 { co / ((pop[i] * pop[j]) as f64).sqrt() } else { 0.0 };
            let same = if tac[i].intersection(&tac[j]).next().is_some() { 0.05 } else { 0.0 };
            if cos + same > 0.0 {
                ties.push((i, j, cos + same));
            }
        }
    }
    // optional sparsification (env TIE_TOPK=k): keep a tie only if it is in the top-k strongest of either endpoint.
    // Needed when the dense tie set is too large for the hypergraph Laplacian (memory grows ~ ties^2).
    if let Some(k) = std::env::var("TIE_TOPK").ok().and_then(|x| x.parse::<usize>().ok()) {
        let mut keep: HashSet<(usize, usize)> = HashSet::new();
        for v in 0..n {
            let mut inc: Vec<&(usize, usize, f64)> = ties.iter().filter(|t| t.0 == v || t.1 == v).collect();
            inc.sort_by(|a, b| b.2.total_cmp(&a.2).then((a.0, a.1).cmp(&(b.0, b.1))));
            for t in inc.into_iter().take(k) {
                keep.insert((t.0, t.1));
            }
        }
        let before = ties.len();
        ties.retain(|t| keep.contains(&(t.0, t.1)));
        println!("sparsified ties: {before} -> {} (top-{k} per entity)", ties.len());
    }
    // connect any leftover components with very weak ties (spectral stages need one component)
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
        let (a, b) = (find(&mut uf, i), find(&mut uf, j));
        uf[a] = b;
    }
    let root0 = find(&mut uf, 0);
    let mut bridged = 0;
    for i in 1..n {
        let r = find(&mut uf, i);
        if r != find(&mut uf, 0) {
            ties.push((0, i, 0.02));
            uf[r] = find(&mut uf, 0);
            bridged += 1;
        }
    }
    let _ = root0;
    let mut b = KbBuilder::new();
    for t in &techs {
        b = b.entity(&t.id, t.tactics.split(',').next().unwrap());
    }
    for &(i, j, w) in &ties {
        b = b.relate(&techs[i].id, &techs[j].id, "co_use_or_tactic", w);
    }
    let kb = b.build()?;
    println!("graph: {} entities, {} ties ({} bridging ties added), connected components {}", n, ties.len(), bridged, kb.components()?);

    let cfg = PipelineConfig { top_k: 10, rerank_pool: 40, prior_weight: 0.0, ..PipelineConfig::default() };
    let t = Instant::now();
    let mut pipe = Pipeline::new(kb, cfg)?;
    println!("pipeline built in {:.1}s (spectral features + connectome RNN, {} units)", t.elapsed().as_secs_f64(), 2 * n);
    let ent: Vec<usize> = techs.iter().map(|t| pipe.kb.index_of(pipe.kb.entity(&t.id).unwrap().id).unwrap()).collect();
    let name_of = |e: usize| -> String { let k = ent.iter().position(|&x| x == e).unwrap(); format!("{} {}", techs[k].id, techs[k].name) };

    // ---- incidents ----
    let incidents = |gs: &[&(String, Vec<String>)]| -> Vec<Incident> {
        let mut out = Vec::new();
        for g in gs.iter().filter(|g| g.1.len() >= 8).take(cap) {
            let mut s: Vec<&String> = g.1.iter().collect();
            s.sort_by_key(|t| (techs[idx[t.as_str()]].order, (*t).clone()));
            for frac in [0.35, 0.6] {
                let k = ((frac * s.len() as f64).round() as usize).clamp(3, 12);
                if s.len() < k + 3 {
                    continue;
                }
                out.push(Incident { group: g.0.clone(), evidence: s[..k].iter().map(|x| x.to_string()).collect(), hidden: s[k..].iter().map(|x| x.to_string()).collect() });
            }
        }
        out
    };
    let mut train_inc = incidents(&train_g);
    let test_inc = incidents(&test_g);
    let mut rng = Lcg(2026);
    for i in (1..train_inc.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        train_inc.swap(i, j);
    }
    println!("incidents: {} train, {} test ({} test groups)", train_inc.len(), test_inc.len(), test_inc.iter().map(|i| i.group.as_str()).collect::<HashSet<_>>().len());
    let query = |inc: &Incident| Query {
        question: format!("Which other techniques is this adversary documented to use? (observed: {})", inc.evidence.len()),
        events: inc.evidence.iter().map(|e| EventSpec { entity: e.clone(), severity: 0.8, note: "observed".into() }).collect(),
    };

    // ---- oracle operator on TRAIN incidents, logged through the real pipeline ----
    let t = Instant::now();
    let (mut approve, mut over) = (0, 0);
    let mut base_train_hits = 0;
    for inc in &train_inc {
        let a = pipe.analyze(&query(inc))?;
        let hidden: HashSet<&str> = inc.hidden.iter().map(|s| s.as_str()).collect();
        let d = match a.brief.candidates.iter().find(|c| hidden.contains(c.entity.as_str())) {
            Some(c) => {
                approve += 1;
                if c.rank == 1 {
                    base_train_hits += 1;
                }
                Decision::Approve { entity: c.entity.clone(), note: "oracle: group documented to use it".into() }
            }
            None => {
                over += 1;
                let best = inc.hidden.iter().max_by_key(|h| (pop[idx[h.as_str()]], std::cmp::Reverse((*h).clone()))).unwrap();
                Decision::Override { entity: best.clone(), note: "oracle: most popular documented technique".into() }
            }
        };
        pipe.record(&a.brief, &d, "oracle")?;
    }
    println!("decisions logged: {} approve, {} override ({} approvals were the reservoir's #1) in {:.0}s", approve, over, base_train_hits, t.elapsed().as_secs_f64());

    // ---- train readouts ----
    let t = Instant::now();
    let set = TrainingSet::from_pipeline(&pipe)?;
    println!("training set: {} examples, skipped {:?} (replayed in {:.0}s)", set.examples.len(), set.skipped, t.elapsed().as_secs_f64());
    let variants = [("all", "all"), ("graph", "graph"), ("reservoir", "reservoir"), ("human", "human"), ("graph+reservoir", "graph,reservoir")];
    let mut models: Vec<(String, Readout)> = Vec::new();
    for (label, g) in variants {
        let rc = ReadoutConfig { groups: Groups::parse(g)?, ..ReadoutConfig::default() };
        let all: Vec<&readout::Example> = set.examples.iter().collect();
        models.push((format!("learned: {label}"), Readout::fit(&all, &rc, set.fingerprint)?));
    }
    // control: same model class trained on shuffled labels
    let mut shuffled = set.clone();
    for ex in shuffled.examples.iter_mut() {
        ex.positive = (rng.next() % ex.rows.len() as u64) as usize;
    }
    let rc = ReadoutConfig::default();
    let all: Vec<&readout::Example> = shuffled.examples.iter().collect();
    models.push(("control: all, shuffled labels".into(), Readout::fit(&all, &rc, set.fingerprint)?));
    let rep = readout::evaluate(&set, &rc)?;
    println!("\nin-sample-protocol evaluation on logged decisions (pool = the {} shown):\n{rep}", pipe.cfg.top_k);
    let (full, _) = (&models[0].1, ());
    println!("group weight (sum |w|): {:?}", full.group_weight().iter().map(|(g, w)| format!("{g:?}={w:.2}")).collect::<Vec<_>>());
    println!("top weights: {:?}", full.weights().iter().take(10).map(|(n, _, w)| format!("{n}={w:+.2}")).collect::<Vec<_>>());
    full.save(format!("{dir}/attack.readout"))?;

    // ---- held-out evaluation ----
    let names: Vec<String> = vec![
        "random (exact expectation)".into(),
        "popularity prior (train groups)".into(),
        "diffusion rule".into(),
        "reservoir rule (pipeline default)".into(),
    ];
    let method_names: Vec<String> = names.iter().cloned().chain(models.iter().map(|m| m.0.clone())).collect();
    let mut per: Vec<Vec<(String, M)>> = vec![Vec::new(); method_names.len()]; // method -> [(group, metrics)]
    let t = Instant::now();
    let mut shown_example: Option<String> = None;
    let mut bucket: Vec<(usize, usize, M, M)> = Vec::new(); // (n_evidence, n_hidden, reservoir, learned)
    for (k, inc) in test_inc.iter().enumerate() {
        let q = query(inc);
        pipe.clear_readout();
        let base = pipe.analyze(&q)?;
        let ev: HashSet<usize> = inc.evidence.iter().map(|e| ent[idx[e.as_str()]]).collect();
        let hidden: HashSet<usize> = inc.hidden.iter().map(|e| ent[idx[e.as_str()]]).collect();
        let cands: Vec<usize> = (0..n).map(|i| pipe.kb.index_of(pipe.kb.entity(&techs[i].id).unwrap().id).unwrap()).filter(|e| !ev.contains(e)).collect();
        let mut by_pop = cands.clone();
        by_pop.sort_by(|&a, &b| pop[ent.iter().position(|&x| x == b).unwrap()].cmp(&pop[ent.iter().position(|&x| x == a).unwrap()]).then(a.cmp(&b)));
        let mut by_diff = cands.clone();
        by_diff.sort_by(|&a, &b| base.diffusion[b].total_cmp(&base.diffusion[a]).then(a.cmp(&b)));
        let res = metrics(&base.baseline_order, &hidden);
        let mut row = vec![random_metrics(cands.len(), hidden.len()), metrics(&by_pop, &hidden), metrics(&by_diff, &hidden), res];
        let mut learned_all = None;
        for (mi, (_, m)) in models.iter().enumerate() {
            pipe.set_readout(m.clone())?;
            let a = pipe.analyze(&q)?;
            let mm = metrics(&a.final_order, &hidden);
            if mi == 0 {
                learned_all = Some(mm);
                if k == 0 || shown_example.is_none() && inc.group == "APT29" {
                    shown_example = Some(format!(
                        "group {} | observed {:?} | documented-but-hidden {} techniques\n{}\nGROUND TRUTH among shown: {:?}",
                        inc.group,
                        inc.evidence.iter().map(|e| name_of(ent[idx[e.as_str()]])).collect::<Vec<_>>(),
                        inc.hidden.len(),
                        human::render(&a.brief, &TemplateExplainer.explain(&a.brief)?),
                        a.brief.candidates.iter().map(|c| (c.rank, c.entity.clone(), hidden.contains(&ent[idx[c.entity.as_str()]]))).collect::<Vec<_>>()
                    ));
                }
            }
            row.push(mm);
        }
        bucket.push((inc.evidence.len(), inc.hidden.len(), res, learned_all.unwrap()));
        for (mi, m) in row.into_iter().enumerate() {
            per[mi].push((inc.group.clone(), m));
        }
        if (k + 1) % 20 == 0 {
            eprintln!("  evaluated {}/{} test incidents ({:.0}s)", k + 1, test_inc.len(), t.elapsed().as_secs_f64());
        }
    }
    println!("held-out evaluation took {:.0}s", t.elapsed().as_secs_f64());

    // macro over groups, bootstrap CI over groups
    let gl: Vec<String> = { let mut v: Vec<String> = test_inc.iter().map(|i| i.group.clone()).collect::<HashSet<_>>().into_iter().collect(); v.sort(); v };
    let gmean = |m: usize| -> Vec<[f64; NM]> {
        gl.iter()
            .map(|g| {
                let v: Vec<&(String, M)> = per[m].iter().filter(|x| &x.0 == g).collect();
                let mut a = [0.0; NM];
                for x in &v {
                    for (k, y) in x.1.arr().iter().enumerate() {
                        a[k] += y / v.len() as f64;
                    }
                }
                a
            })
            .collect()
    };
    let gm: Vec<Vec<[f64; NM]>> = (0..method_names.len()).map(gmean).collect();
    let boot = |f: &dyn Fn(&[usize]) -> f64| -> (f64, f64, f64) {
        let all: Vec<usize> = (0..gl.len()).collect();
        let pt = f(&all);
        let mut r = Lcg(99);
        let mut v: Vec<f64> = (0..2000).map(|_| f(&(0..gl.len()).map(|_| (r.next() % gl.len() as u64) as usize).collect::<Vec<_>>())).collect();
        v.sort_by(|a, b| a.total_cmp(b));
        (pt, v[50], v[1949])
    };
    println!("\n== HELD-OUT RESULTS ({} groups, {} incidents; macro-average over groups, 95% bootstrap CI over groups) ==", gl.len(), test_inc.len());
    for k in 0..NM {
        println!("\n[{}]", MNAMES[k]);
        for (mi, name) in method_names.iter().enumerate() {
            let (p, lo, hi) = boot(&|s: &[usize]| s.iter().map(|&g| gm[mi][g][k]).sum::<f64>() / s.len() as f64);
            println!("  {name:<34} {p:.3}  [{lo:.3}, {hi:.3}]");
        }
    }
    println!("\n== PAIRED DIFFERENCES vs the reservoir rule (learned: all minus reservoir), 95% CI ==");
    for k in 0..NM {
        let (p, lo, hi) = boot(&|s: &[usize]| s.iter().map(|&g| gm[4][g][k] - gm[3][g][k]).sum::<f64>() / s.len() as f64);
        println!("  {:<12} {p:+.3}  [{lo:+.3}, {hi:+.3}]", MNAMES[k]);
    }
    println!("\n== PAIRED DIFFERENCES vs the popularity prior (learned: all minus popularity), 95% CI ==");
    for k in 0..NM {
        let (p, lo, hi) = boot(&|s: &[usize]| s.iter().map(|&g| gm[4][g][k] - gm[1][g][k]).sum::<f64>() / s.len() as f64);
        println!("  {:<12} {p:+.3}  [{lo:+.3}, {hi:+.3}]", MNAMES[k]);
    }
    // by evidence size / hidden size
    println!("\n== by number of observed techniques (precision@5: reservoir -> learned) ==");
    for (lo, hi) in [(3, 5), (6, 8), (9, 12)] {
        let v: Vec<&(usize, usize, M, M)> = bucket.iter().filter(|b| b.0 >= lo && b.0 <= hi).collect();
        if !v.is_empty() {
            let f = |g: &dyn Fn(&M) -> f64, w: usize| v.iter().map(|b| g(if w == 0 { &b.2 } else { &b.3 })).sum::<f64>() / v.len() as f64;
            println!("  evidence {lo}-{hi} ({} incidents): p@5 {:.3} -> {:.3}   hit@5 {:.3} -> {:.3}", v.len(), f(&|m| m.p5, 0), f(&|m| m.p5, 1), f(&|m| m.hit5, 0), f(&|m| m.hit5, 1));
        }
    }
    // per-group wins
    let wins = (0..gl.len()).filter(|&g| gm[4][g][1] > gm[3][g][1] + 1e-12).count();
    let losses = (0..gl.len()).filter(|&g| gm[4][g][1] < gm[3][g][1] - 1e-12).count();
    println!("\nper-group precision@5, learned vs reservoir rule: {wins} better, {losses} worse, {} equal (of {})", gl.len() - wins - losses, gl.len());
    println!("\n== EXAMPLE BRIEF (held-out incident, learned readout) ==\n{}", shown_example.unwrap_or_default());
    println!("\ntotal wall time {:.0}s", t0.elapsed().as_secs_f64());
    if test_inc.is_empty() {
        bail!("no test incidents");
    }
    Ok(())
}
