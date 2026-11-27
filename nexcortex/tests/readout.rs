//! Learned readout: graph features + reservoir trajectory + human decisions.
//!
//! There are no real operator logs here. The "operators" below are deterministic stand-ins with a
//! *known hidden preference*, so a test can tell whether the readout recovers it. Passing these
//! tests shows the machinery works and does not leak labels; it does not show the readout would
//! help real operators.

use nexcortex::human::Decision;
use nexcortex::readout::{evaluate, Groups, ReadoutConfig, TrainingSet};
use nexcortex::*;

/// `prior_weight = 0`: with a learned readout the human history reaches the model through its
/// human features. (With priors on, a 22-entity graph runs out of non-evidence candidates once
/// every entity has been confirmed once.)
fn pipe() -> Pipeline {
    let cfg = PipelineConfig { prior_weight: 0.0, ..PipelineConfig::default() };
    Pipeline::new(demo_builder().build().unwrap(), cfg).unwrap()
}

/// Tiny deterministic generator (no rand dependency in tests).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn random_query(p: &Pipeline, g: &mut Lcg) -> Query {
    let names: Vec<String> = p.kb.entities().iter().map(|e| e.name.clone()).collect();
    let k = 1 + g.below(3);
    let mut chosen: Vec<String> = Vec::new();
    while chosen.len() < k {
        let c = names[g.below(names.len())].clone();
        if !chosen.contains(&c) {
            chosen.push(c);
        }
    }
    Query {
        question: "who else?".into(),
        events: chosen.into_iter().map(|c| EventSpec { entity: c, severity: 0.4 + 0.1 * g.below(6) as f64, note: String::new() }).collect(),
    }
}

/// Run `n` queries through the pipeline; `operator` picks one of the shown candidates (or None = defer).
fn simulate(p: &mut Pipeline, n: usize, seed: u64, operator: &dyn Fn(&Pipeline, &Brief) -> Option<String>) {
    let mut g = Lcg(seed);
    for _ in 0..n {
        let q = random_query(p, &mut g);
        let a = p.analyze(&q).unwrap_or_else(|e| panic!("{e}: confirmed={:?} q={:?}", p.kb.confirmed().unwrap(), q.events));
        let d = match operator(p, &a.brief) {
            Some(entity) => Decision::Approve { entity, note: String::new() },
            None => Decision::Defer { note: String::new() },
        };
        p.record(&a.brief, &d, "sim").unwrap();
    }
}

/// Hidden preference unrelated to the reservoir rule: the operator takes the best-connected asset.
fn likes_hubs(p: &Pipeline, b: &Brief) -> Option<String> {
    b.candidates
        .iter()
        .max_by(|x, y| x.pagerank.total_cmp(&y.pagerank).then(y.rank.cmp(&x.rank)))
        .map(|c| c.entity.clone())
        .filter(|_| !p.kb.is_empty())
}

fn cfg(groups: &str) -> ReadoutConfig {
    ReadoutConfig { groups: Groups::parse(groups).unwrap(), ..ReadoutConfig::default() }
}

#[test]
fn recovers_a_hidden_graph_preference_that_the_reservoir_rule_misses() {
    let mut p = pipe();
    simulate(&mut p, 40, 1, &likes_hubs);
    let set = TrainingSet::from_pipeline(&p).unwrap();
    assert_eq!(set.examples.len(), 40);
    let r = evaluate(&set, &cfg("all")).unwrap();
    println!("{r}");
    assert!(r.learned_loo.top1 > r.reservoir_rule.top1 + 0.3, "{r}");
    let (fwd, _) = r.learned_forward.clone().unwrap();
    assert!(fwd.top1 > r.reservoir_rule.top1 + 0.3 && fwd.top1 > 2.0 * r.chance_top1, "{r}");
}

/// A habit ("vault-01 if shown, else ad-01, else the shortest name") is only visible through the
/// human-decision features: no graph or reservoir feature identifies an entity by name.
#[test]
fn human_features_pick_up_an_operator_habit() {
    let habit = |_: &Pipeline, b: &Brief| -> Option<String> {
        ["vault-01", "ad-01"].iter().find_map(|h| b.candidates.iter().find(|c| c.entity == *h)).or(b.candidates.iter().min_by_key(|c| (c.entity.len(), c.entity.clone()))).map(|c| c.entity.clone())
    };
    let mut p = pipe();
    simulate(&mut p, 50, 2, &habit);
    let set = TrainingSet::from_pipeline(&p).unwrap();
    let human = evaluate(&set, &cfg("human")).unwrap();
    let blind = evaluate(&set, &cfg("graph,reservoir")).unwrap();
    println!("human only:\n{human}\nwithout human:\n{blind}");
    let (hf, _) = human.learned_forward.clone().unwrap();
    assert!(hf.top1 > human.reservoir_rule.top1 && hf.top1 > 2.0 * human.chance_top1, "{human}");
    assert!(human.learned_loo.top1 > blind.learned_loo.top1, "human features should help here\n{human}\n{blind}");
}

/// Example k must be computed from decisions 0..k only: truncating the log after k decisions must
/// leave example k-1 (and every earlier one) bit-for-bit unchanged.
#[test]
fn examples_do_not_leak_later_decisions() {
    let (mut long, mut short) = (pipe(), pipe());
    simulate(&mut long, 14, 3, &likes_hubs);
    simulate(&mut short, 9, 3, &likes_hubs);
    let (a, b) = (TrainingSet::from_pipeline(&long).unwrap(), TrainingSet::from_pipeline(&short).unwrap());
    assert_eq!(b.examples.len(), 9);
    for k in 0..9 {
        assert_eq!(a.examples[k].entities, b.examples[k].entities, "pool {k}");
        assert_eq!(a.examples[k].positive, b.examples[k].positive);
        assert_eq!(a.examples[k].rows, b.examples[k].rows, "features of decision {k} changed when later decisions were added");
    }
}

#[test]
fn training_is_deterministic_and_the_model_round_trips_through_a_file() {
    let mut p = pipe();
    simulate(&mut p, 24, 4, &likes_hubs);
    let (m1, _, _) = readout::train(&p, &cfg("all")).unwrap();
    let (m2, _, _) = readout::train(&p, &cfg("all")).unwrap();
    let path = std::env::temp_dir().join(format!("nexcortex-readout-{}.txt", std::process::id()));
    m1.save(&path).unwrap();
    let loaded = Readout::load(&path).unwrap();
    std::fs::remove_file(&path).ok();
    let set = TrainingSet::from_pipeline(&p).unwrap();
    for ex in &set.examples {
        let rows: Vec<_> = ex.rows.iter().collect();
        assert_eq!(m1.probs(&rows), m2.probs(&rows));
        for (x, y) in m1.probs(&rows).iter().zip(loaded.probs(&rows)) {
            assert!((x - y).abs() < 1e-12);
        }
    }
    assert!(Readout::load("/nonexistent/readout").is_err());
}

#[test]
fn attached_readout_reranks_and_is_refused_on_a_different_reservoir() {
    let mut p = pipe();
    simulate(&mut p, 30, 5, &likes_hubs);
    let (model, _, _) = readout::train(&p, &cfg("all")).unwrap();
    let q = Query { question: "who else?".into(), events: vec![EventSpec { entity: "web-01".into(), severity: 0.9, note: String::new() }, EventSpec { entity: "ws-02".into(), severity: 0.6, note: String::new() }] };
    let before = p.analyze(&q).unwrap().brief;
    assert!(before.candidates.iter().all(|c| c.learned.is_none()));
    p.set_readout(model.clone()).unwrap();
    let after = p.analyze(&q).unwrap().brief;
    assert!(after.candidates.iter().all(|c| c.learned.is_some() && c.rank >= 1));
    let probs: f64 = after.candidates.iter().map(|c| c.learned.unwrap()).sum();
    assert!(probs <= 1.0 + 1e-9, "probabilities over the pool must sum to <= 1, got {probs}");
    assert!(after.candidates.windows(2).all(|w| w[0].learned >= w[1].learned), "candidates must be in learned order");
    assert!(after.flags.iter().any(|f| f.contains("learned readout")));
    assert!(after.facts().contains("learned score"));
    // evidence entities are still never candidates
    assert!(after.candidates.iter().all(|c| c.entity != "web-01" && c.entity != "ws-02"));
    p.clear_readout();
    assert_eq!(p.analyze(&q).unwrap().brief.candidates[0].entity, before.candidates[0].entity);

    // a different knowledge graph (one extra tie) is a different reservoir: refuse
    let other = demo_builder().relate("web-03", "db-02", "extra", 0.9).build().unwrap();
    let mut p2 = Pipeline::new(other, PipelineConfig { prior_weight: 0.0, ..PipelineConfig::default() }).unwrap();
    assert!(p2.set_readout(model).unwrap_err().to_string().contains("fingerprint"));
}

#[test]
fn too_little_or_unlabelled_data_is_refused_not_guessed() {
    let mut p = pipe();
    simulate(&mut p, 5, 6, &likes_hubs);
    let err = readout::train(&p, &cfg("all")).unwrap_err().to_string();
    assert!(err.contains("only 5 labelled decisions"), "{err}");
    // defer / reject are not labels
    let mut p = pipe();
    simulate(&mut p, 6, 7, &|_, _| None);
    let set = TrainingSet::from_pipeline(&p).unwrap();
    assert_eq!(set.examples.len(), 0);
    assert_eq!(set.skipped.no_label, 6);
    // decisions recorded through the legacy path carry no evidence and are skipped, not guessed
    let mut p = pipe();
    p.kb.record_decision("approve", "old", "x", "", 0.5, &["web-01".into()], &["web-01".into()], &[]).unwrap();
    let set = TrainingSet::from_pipeline(&p).unwrap();
    assert_eq!((set.examples.len(), set.skipped.legacy), (0, 1));
    assert!(Groups::parse("graph,bogus").is_err() && Groups::parse("").is_err());
}

#[test]
fn override_pool_excludes_the_shown_candidates_and_logged_decisions_survive_restart() {
    let dir = std::env::temp_dir().join(format!("nexcortex-kb-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let q = Query { question: "who else?".into(), events: vec![EventSpec { entity: "web-01".into(), severity: 0.9, note: String::new() }] };
    {
        let kb = demo_builder().commit_into(nexgraph::Database::open(&dir).unwrap()).unwrap();
        let mut p = Pipeline::new(kb, PipelineConfig { prior_weight: 0.0, ..PipelineConfig::default() }).unwrap();
        let a = p.analyze(&q).unwrap();
        let shown: Vec<_> = a.brief.candidates.iter().map(|c| c.entity.clone()).collect();
        let far = ["vault-01", "db-02", "ws-05"].into_iter().find(|n| !shown.contains(&n.to_string())).unwrap();
        p.record(&a.brief, &Decision::Override { entity: far.into(), note: "n".into() }, "ana").unwrap();
    }
    let kb = KnowledgeBase::open(&dir).unwrap();
    let (log, legacy) = kb.decision_log().unwrap();
    assert_eq!((log.len(), legacy), (1, 0));
    assert_eq!(log[0].evidence, vec![("web-01".to_string(), 0.9)]);
    assert!(log[0].refused.is_some() && log[0].chosen.is_some());
    let p = Pipeline::new(kb, PipelineConfig { prior_weight: 0.0, ..PipelineConfig::default() }).unwrap();
    let set = TrainingSet::from_pipeline(&p).unwrap();
    assert_eq!(set.examples.len(), 1);
    let ex = &set.examples[0];
    let n_cand = p.kb.entities().len() - 1; // everything except the single evidence entity
    assert_eq!(ex.rows.len(), n_cand - p.cfg.top_k, "an override competes against every entity that was NOT shown");
    let shown: Vec<usize> = p.analyze(&q).unwrap().baseline_order.iter().take(p.cfg.top_k).copied().collect();
    assert!(ex.entities.iter().all(|e| !shown.contains(e)), "shown candidates were refused, so they are not alternatives");
    assert!(ex.positive < ex.rows.len());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Recording decisions (including a defer, which is an isolated node) must not change the
/// physics a later query runs on, nor break the spectral stages after a restart.
#[test]
fn recorded_decisions_do_not_perturb_features_or_spectral_stages() {
    let dir = std::env::temp_dir().join(format!("nexcortex-kb2-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let q = Query { question: "who else?".into(), events: vec![EventSpec { entity: "web-01".into(), severity: 0.9, note: String::new() }] };
    let cfg = PipelineConfig { prior_weight: 0.0, ..PipelineConfig::default() };
    let clean = Pipeline::new(demo_builder().build().unwrap(), cfg.clone()).unwrap().analyze(&q).unwrap();
    {
        let kb = demo_builder().commit_into(nexgraph::Database::open(&dir).unwrap()).unwrap();
        let mut p = Pipeline::new(kb, cfg.clone()).unwrap();
        let a = p.analyze(&q).unwrap();
        let top = a.brief.candidates[0].entity.clone();
        p.record(&a.brief, &Decision::Approve { entity: top, note: String::new() }, "a").unwrap();
        p.record(&a.brief, &Decision::Defer { note: String::new() }, "a").unwrap();
        p.record(&a.brief, &Decision::Reject { note: String::new() }, "a").unwrap();
    }
    let p = Pipeline::new(KnowledgeBase::open(&dir).unwrap(), cfg).unwrap();
    let after = p.analyze(&q).unwrap();
    assert_eq!(clean.diffusion, after.diffusion);
    assert_eq!(clean.trace.salience, after.trace.salience);
    assert_eq!(clean.structure.cluster, after.structure.cluster);
    let _ = std::fs::remove_dir_all(&dir);
}
