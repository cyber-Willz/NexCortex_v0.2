use nexcortex::features;
use nexcortex::human::{render, Decision};
use nexcortex::*;
use std::collections::VecDeque;
use std::io::Cursor;

fn pipe() -> Pipeline {
    Pipeline::new(demo_builder().build().unwrap(), PipelineConfig::default()).unwrap()
}
fn q(ev: &[(&str, f64)]) -> Query {
    Query { question: "who else?".into(), events: ev.iter().map(|(n, s)| EventSpec { entity: n.to_string(), severity: *s, note: String::new() }).collect() }
}
const DMZ: [&str; 5] = ["web-01", "web-02", "web-03", "lb-01", "waf-01"];
const DATA: [&str; 5] = ["db-01", "db-02", "db-03", "backup-01", "vault-01"];
const CORP: [&str; 10] = ["ws-01", "ws-02", "ws-03", "ws-04", "ws-05", "ws-06", "alice", "bob", "carol", "ad-01"];

#[test]
fn knowledge_base_is_connected_symmetric_and_validated() {
    let kb = demo_builder().build().unwrap();
    assert_eq!(kb.entities().len(), 22);
    assert_eq!(kb.components().unwrap(), 1);
    let n = kb.entities().len();
    let a = kb.adjacency().unwrap();
    for i in 0..n {
        assert_eq!(a[i * n + i], 0.0);
        for j in 0..n {
            assert!((a[i * n + j] - a[j * n + i]).abs() < 1e-12);
        }
    }
    // bad inputs are refused, not silently dropped
    assert!(KbBuilder::new().entity("a", "x").relate("a", "ghost", "r", 1.0).build().is_err());
    assert!(KbBuilder::new().entity("a", "x").entity("a", "y").build().is_err());
    assert!(KbBuilder::new().entity("a", "x").entity("b", "x").relate("a", "b", "r", 0.0).build().is_err());
}

#[test]
fn disconnected_graph_is_rejected_for_spectral_stages() {
    let kb = KbBuilder::new()
        .entity("a", "x").entity("b", "x").entity("c", "x").entity("d", "x")
        .relate("a", "b", "r", 1.0).relate("c", "d", "r", 1.0)
        .build().unwrap();
    let err = features::structure(&kb, &Default::default()).unwrap_err().to_string();
    assert!(err.contains("2 connected components"), "{err}");
}

#[test]
fn diffusion_decays_with_scale_and_peaks_at_the_source() {
    let p = pipe();
    let i = p.kb.index_of(p.kb.entity("web-01").unwrap().id).unwrap();
    let f = features::event_fields(&p.kb, &p.cfg.features, &[(i, 1.0)]).unwrap().remove(0);
    assert_eq!(f.heat.len(), 3);
    assert!(f.heat.iter().flatten().all(|&x| x >= -1e-12), "heat must be non-negative");
    assert!(f.heat[0][i] > f.heat[1][i] && f.heat[1][i] > f.heat[2][i], "source value must decay as t grows");
    // nodes in the same zone receive more than far-away ones at the short scale
    let near = p.kb.index_of(p.kb.entity("web-02").unwrap().id).unwrap();
    let far = p.kb.index_of(p.kb.entity("db-02").unwrap().id).unwrap();
    assert!(f.heat[0][near] > f.heat[0][far]);
}

#[test]
fn network_respects_dales_law_and_is_stable() {
    let p = pipe();
    let net = p.memory().network();
    net.check_constraints().unwrap();
    assert!(net.spectral_radius() < 1.0, "radius {}", net.spectral_radius());
    let a = p.analyze(&q(&[("web-01", 0.9)])).unwrap();
    assert!(a.trace.activity.iter().all(|x| x.is_finite()));
    assert!(a.trace.activity[a.trace.steps] < a.trace.activity[a.trace.input_end], "activity must fade, not blow up");
}

/// Regression test for the first readout (end-of-run state), which returned the same hub for any evidence.
#[test]
fn ranking_depends_on_the_evidence() {
    let p = pipe();
    let cases: [(&[(&str, f64)], &[&str]); 3] = [
        (&[("web-01", 0.9), ("web-02", 0.5)], &DMZ),
        (&[("db-01", 0.9), ("db-02", 0.6)], &DATA),
        (&[("ws-01", 0.9), ("alice", 0.6)], &CORP),
    ];
    let mut tops = Vec::new();
    for (ev, zone) in cases {
        let a = p.analyze(&q(ev)).unwrap();
        let top3: Vec<&str> = a.brief.candidates.iter().take(3).map(|c| c.entity.as_str()).collect();
        let in_zone = top3.iter().filter(|n| zone.contains(n)).count();
        assert!(in_zone >= 2, "evidence in one zone should surface that zone: {top3:?}");
        tops.push(top3[0].to_string());
    }
    assert!(tops[0] != tops[1] && tops[1] != tops[2] && tops[0] != tops[2], "top candidate must differ across evidence: {tops:?}");
}

#[test]
fn evidence_entities_are_never_candidates_and_inputs_are_validated() {
    let p = pipe();
    let a = p.analyze(&q(&[("web-01", 0.9), ("db-03", 0.7)])).unwrap();
    assert!(a.brief.candidates.iter().all(|c| c.entity != "web-01" && c.entity != "db-03"));
    assert!(a.brief.candidates.windows(2).all(|w| w[0].salience >= w[1].salience));
    assert!((a.brief.candidates[0].salience - 1.0).abs() < 1e-12);
    assert!((0.0..=1.0).contains(&a.brief.confidence));
    assert!(p.analyze(&q(&[("nope", 0.5)])).is_err());
    assert!(p.analyze(&q(&[("web-01", 0.0)])).is_err());
    assert!(p.analyze(&q(&[("web-01", 1.5)])).is_err());
    assert!(p.analyze(&Query { question: "x".into(), events: vec![] }).is_err());
    // path is a chain of existing ties starting at an evidence entity and ending at the candidate
    let c = &a.brief.candidates[0];
    assert!(c.path.len() >= 2 && (c.path[0] == "web-01" || c.path[0] == "db-03") && c.path.last().unwrap() == &c.entity);
}

#[test]
fn analysis_is_deterministic() {
    let (p1, p2) = (pipe(), pipe());
    let ev = [("web-01", 0.9), ("web-02", 0.5), ("db-03", 0.7)];
    let (a, b) = (p1.analyze(&q(&ev)).unwrap(), p2.analyze(&q(&ev)).unwrap());
    assert_eq!(a.brief.facts(), b.brief.facts());
}

#[test]
fn single_evidence_and_order_flags() {
    let p = pipe();
    let one = p.analyze(&q(&[("web-01", 0.9)])).unwrap();
    assert!(one.brief.flags.iter().any(|f| f.contains("only one evidence")));
    // the order check really runs: reversing evidence is a different input to the network
    let a = p.analyze(&q(&[("web-01", 0.9), ("web-02", 0.5), ("db-03", 0.7)])).unwrap();
    let b = p.analyze(&q(&[("db-03", 0.7), ("web-02", 0.5), ("web-01", 0.9)])).unwrap();
    assert!(a.trace.salience != b.trace.salience, "memory should be order sensitive");
}

#[test]
fn human_decision_is_recorded_and_feeds_later_runs() {
    let mut p = pipe();
    let ev = [("web-01", 0.9), ("db-03", 0.7)];
    let top = p.analyze(&q(&ev)).unwrap().brief.candidates[0].entity.clone();
    let mut h = ScriptedDecisionMaker { script: VecDeque::from([Decision::Approve { entity: top.clone(), note: "checked".into() }]) };
    let out = p.run(&q(&ev), &TemplateExplainer, &mut h).unwrap();
    assert_eq!(out.decision.kind(), "approve");
    assert_eq!(p.kb.decisions().unwrap(), vec![("approve".to_string(), "scripted".to_string(), "who else?".to_string(), "checked".to_string())]);
    assert_eq!(p.kb.confirmed().unwrap(), vec![(top.clone(), 1)]);
    // the next analysis carries the confirmation as a prior seed, ahead of the new evidence
    let a = p.analyze(&q(&[("web-02", 0.5)])).unwrap();
    assert!(a.brief.evidence[0].from_prior && a.brief.evidence[0].entity == top);
    assert!(a.brief.candidates.iter().all(|c| c.entity != top), "a prior seed is evidence, not a candidate");
}

#[test]
fn invalid_decisions_are_refused_and_nothing_is_written() {
    let mut p = pipe();
    let a = p.analyze(&q(&[("web-01", 0.9)])).unwrap();
    // approving something that was not offered
    let bad = Decision::Approve { entity: "ws-01".into(), note: String::new() };
    assert!(a.brief.candidates.iter().all(|c| c.entity != "ws-01"));
    assert!(p.record(&a.brief, &bad, "t").is_err());
    assert!(p.record(&a.brief, &Decision::Override { entity: "ghost".into(), note: String::new() }, "t").is_err());
    assert!(p.kb.decisions().unwrap().is_empty());
    // defer writes an audit record but confirms nothing
    p.record(&a.brief, &Decision::Defer { note: "later".into() }, "t").unwrap();
    assert_eq!(p.kb.decisions().unwrap().len(), 1);
    assert!(p.kb.confirmed().unwrap().is_empty());
    // override confirms the named entity and counts a rejection against the machine's top pick
    p.record(&a.brief, &Decision::Override { entity: "ws-01".into(), note: String::new() }, "t").unwrap();
    assert_eq!(p.kb.confirmed().unwrap(), vec![("ws-01".to_string(), 1)]);
    let r = p.kb.db.query("MATCH (n:Entity) WHERE n.rejected > 0 RETURN n.name").unwrap();
    assert_eq!(r.rows[0][0].as_str().unwrap(), a.brief.candidates[0].entity);
}

#[test]
fn decisions_survive_a_restart() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("kb-restart-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    {
        let kb = demo_builder().commit_into(nexgraph::Database::open(&dir).unwrap()).unwrap();
        let mut p = Pipeline::new(kb, PipelineConfig::default()).unwrap();
        let a = p.analyze(&q(&[("web-01", 0.9)])).unwrap();
        let top = a.brief.candidates[0].entity.clone();
        p.record(&a.brief, &Decision::Approve { entity: top, note: "persist me".into() }, "ana").unwrap();
    }
    let kb = KnowledgeBase::open(&dir).unwrap();
    assert_eq!(kb.entities().len(), 22);
    assert_eq!(kb.decisions().unwrap()[0].1, "ana");
    assert_eq!(kb.confirmed().unwrap().len(), 1);
    drop(kb);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stdio_operator_parses_commands_and_never_guesses() {
    let p = pipe();
    let a = p.analyze(&q(&[("web-01", 0.9)])).unwrap();
    let ex = TemplateExplainer.explain(&a.brief).unwrap();
    let run = |input: &str| {
        let mut out = Vec::new();
        let mut h = StdioDecisionMaker { input: Cursor::new(input.to_string()), output: &mut out, name: "op".into() };
        let r = h.decide(&a.brief, &ex);
        (r, String::from_utf8(out).unwrap())
    };
    let (r, shown) = run("banana\n99\n2 -- looks right\n");
    assert_eq!(r.unwrap(), Decision::Approve { entity: a.brief.candidates[1].entity.clone(), note: "looks right".into() });
    assert!(shown.contains("EXPLANATION") && shown.contains("CAVEATS") && shown.contains("not understood"));
    assert_eq!(run("r -- wrong zone\n").0.unwrap(), Decision::Reject { note: "wrong zone".into() });
    assert_eq!(run("d\n").0.unwrap(), Decision::Defer { note: String::new() });
    assert_eq!(run("o ad-01 -- known jump path\n").0.unwrap(), Decision::Override { entity: "ad-01".into(), note: "known jump path".into() });
    // closed input is an error, not an implicit approval
    assert!(run("").0.is_err());
    assert!(run("o\n").0.is_err());
}

#[test]
fn template_explanation_is_grounded_and_grounding_check_catches_strays() {
    let p = pipe();
    let a = p.analyze(&q(&[("web-01", 0.9), ("db-03", 0.7)])).unwrap();
    let ex = TemplateExplainer.explain(&a.brief).unwrap();
    assert!(ex.grounded && ex.source == "template" && ex.text.contains(&a.brief.candidates[0].entity));
    assert!(render(&a.brief, &ex).contains("EXPLANATION [template]"));
    let known: Vec<String> = p.kb.entities().iter().map(|e| e.name.clone()).collect();
    let allowed = explain::brief_entities(&a.brief);
    assert!(explain::ungrounded_mentions(&ex.text, &known, &allowed).is_empty());
    assert_eq!(explain::ungrounded_mentions("blame carol and web-01", &known, &allowed), vec!["carol".to_string()]);
}

#[cfg(feature = "llm")]
mod llm {
    use super::*;
    fn fixture() -> Option<std::path::PathBuf> {
        let d = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../nexgen_llm/nexgen_llm/tests/fixtures/qwen3_yarn_bf16");
        d.join("model.safetensors").exists().then_some(d)
    }

    #[test]
    fn oversized_prompt_falls_back_to_the_template_with_a_warning() {
        let Some(dir) = fixture() else { return };
        let p = pipe();
        let known: Vec<String> = p.kb.entities().iter().map(|e| e.name.clone()).collect();
        let llm = LlmExplainer::load(dir, known, 8).unwrap();
        let a = p.analyze(&q(&[("web-01", 0.9), ("db-03", 0.7)])).unwrap();
        let ex = llm.explain(&a.brief).unwrap(); // 64-token context cannot hold the brief
        assert_eq!(ex.source, "template");
        assert!(ex.grounded && ex.warnings.iter().any(|w| w.contains("context")), "{:?}", ex.warnings);
    }

    #[test]
    fn model_text_that_fails_grounding_is_marked_unverified() {
        let Some(dir) = fixture() else { return };
        let kb = KbBuilder::new().entity("a", "x").entity("b", "x").entity("c", "x").entity("d", "x").entity("e", "x")
            .relate("a", "b", "r", 1.0).relate("b", "c", "r", 1.0).relate("c", "d", "r", 1.0).relate("d", "e", "r", 1.0).relate("e", "a", "r", 1.0)
            .build().unwrap();
        let p = Pipeline::new(kb, PipelineConfig::default()).unwrap();
        let mut a = p.analyze(&q(&[("a", 0.9)])).unwrap();
        // shrink the brief until it fits the fixture's 64-token context
        a.brief.question = "q".into();
        a.brief.candidates.truncate(1);
        a.brief.flags.clear();
        a.brief.candidates[0].path.clear();
        a.brief.memory.half_life = Some(1);
        let mut llm = LlmExplainer::load(dir, vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()], 6).unwrap();
        llm.system = String::new();
        llm.instruction = String::new();
        let ex = llm.explain(&a.brief).unwrap();
        // The prompt fits, so real generation ran. A random-weight toy model cannot produce a
        // grounded explanation; the layer must say so and keep the deterministic text in front.
        assert_eq!(ex.source, "llm", "generation path must run: {:?}", ex.warnings);
        assert!(!ex.grounded);
        assert!(ex.text.starts_with("Given evidence on") && ex.text.contains("[unverified model text]"), "{}", ex.text);
        assert!(ex.warnings.iter().any(|w| w.contains("grounding")));
    }
}

#[test]
fn spectral_clusters_recover_the_three_planted_zones() {
    let p = pipe();
    let s = p.structure();
    let c = |n: &str| s.cluster[p.kb.index_of(p.kb.entity(n).unwrap().id).unwrap()];
    for zone in [&DMZ[..], &DATA[..], &CORP[..]] {
        assert!(zone.iter().all(|n| c(n) == c(zone[0])), "zone split across clusters: {zone:?}");
    }
    assert!(c("web-01") != c("db-01") && c("db-01") != c("ws-01") && c("web-01") != c("ws-01"));
    // three tight zones => three near-zero Laplacian eigenvalues, then a gap
    assert!(s.eigenvalues[2] < 0.1 && s.eigenvalues[3] > 3.0 * s.eigenvalues[2], "{:?}", s.eigenvalues);
}
