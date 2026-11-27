use anyhow::{bail, Context, Result};
use nexcortex::*;
use std::collections::VecDeque;
use std::io::{stdin, stdout, BufReader};

const USAGE: &str = "usage: nexcortex [options]
  --db <dir>             durable knowledge base (seeded with the synthetic demo network if empty)
                         default: in-memory demo network
  --event <name:sev[:note]>   evidence, in the order observed; sev in (0,1]; repeatable
                         default: three demo alerts
  --question <text>      what the operator wants to know
  --llm <model_dir>      explain with a local model (Llama/Mistral/Qwen2/Qwen3 safetensors dir); default: template
  --claude <model>       explain with the Claude API (key from $ANTHROPIC_API_KEY); falls back to the template on any failure
  --max-new <n>          tokens the model may write (default 160)
  --operator <name>      name recorded in the audit trail (default: $USER)
  --script <approve|reject|defer>   NON-INTERACTIVE stand-in for the human (tests / demos); recorded as 'scripted'
  --history              print the decisions stored in the knowledge base and exit
  --seed <n>             working-memory network seed (default 7)
  --prior-weight <x>     severity of the auto-added prior for confirmed entities (default 0.3; use 0 with a learned readout)
  --train-readout <file> fit the learned readout on the decisions logged in --db, print honest held-out numbers, save it, exit
  --readout <file>       rank with a saved learned readout (must match this knowledge base / seed / prior weight)
  --readout-groups <g>   feature sources for --train-readout: all | graph,reservoir,human (default all)
  --l2 <x>               readout L2 strength (default 1.0)";

fn main() -> Result<()> {
    let mut a = std::env::args().skip(1);
    let (mut db, mut llm, mut question, mut script) = (None::<String>, None::<String>, None::<String>, None::<String>);
    let mut claude = None::<String>;
    let mut operator = std::env::var("USER").unwrap_or_else(|_| "operator".into());
    let (mut train_readout, mut readout_file, mut groups, mut l2, mut prior) = (None::<String>, None::<String>, "all".to_string(), 1.0f64, None::<f64>);
    let (mut events, mut max_new, mut history, mut seed) = (Vec::<EventSpec>::new(), 160usize, false, 7u64);
    while let Some(x) = a.next() {
        let mut val = |f: &str| a.next().with_context(|| format!("{f} needs a value\n{USAGE}"));
        match x.as_str() {
            "--db" => db = Some(val("--db")?),
            "--llm" => llm = Some(val("--llm")?),
            "--claude" => claude = Some(val("--claude")?),
            "--question" => question = Some(val("--question")?),
            "--operator" => operator = val("--operator")?,
            "--script" => script = Some(val("--script")?),
            "--max-new" => max_new = val("--max-new")?.parse()?,
            "--seed" => seed = val("--seed")?.parse()?,
            "--history" => history = true,
            "--train-readout" => train_readout = Some(val("--train-readout")?),
            "--readout" => readout_file = Some(val("--readout")?),
            "--readout-groups" => groups = val("--readout-groups")?,
            "--l2" => l2 = val("--l2")?.parse()?,
            "--prior-weight" => prior = Some(val("--prior-weight")?.parse()?),
            "--event" => {
                let v = val("--event")?;
                let mut p = v.splitn(3, ':');
                let (name, sev) = (p.next().unwrap_or(""), p.next().context("--event needs name:severity")?);
                events.push(EventSpec { entity: name.into(), severity: sev.parse().context("severity must be a number")?, note: p.next().unwrap_or("").into() });
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => bail!("unknown option {other}\n{USAGE}"),
        }
    }

    let mut kb = match &db {
        Some(d) => KnowledgeBase::open(d)?,
        None => KnowledgeBase::from_db(nexgraph::Database::in_memory())?,
    };
    if kb.is_empty() {
        kb = demo_builder().commit_into(kb.db)?;
        eprintln!("[seeded the synthetic demo network: {} entities]", kb.entities().len());
    }
    if history {
        for (k, who, q, note) in kb.decisions()? {
            println!("{k:<8} by {who:<10} | {q} | {note}");
        }
        return Ok(());
    }
    if events.is_empty() {
        events = vec![
            EventSpec { entity: "web-01".into(), severity: 0.9, note: "WAF: SQL injection burst".into() },
            EventSpec { entity: "web-02".into(), severity: 0.5, note: "unusual outbound connections".into() },
            EventSpec { entity: "db-03".into(), severity: 0.7, note: "bulk read outside batch window".into() },
        ];
    }
    let q = Query { question: question.unwrap_or_else(|| "Which other entities are most likely involved and should be checked first?".into()), events };

    let mut cfg = PipelineConfig::default();
    cfg.memory.seed = seed;
    if let Some(p) = prior {
        cfg.prior_weight = p;
    }
    let mut pipe = Pipeline::new(kb, cfg)?;

    if let Some(file) = &train_readout {
        let rc = ReadoutConfig { groups: readout::Groups::parse(&groups)?, l2, ..ReadoutConfig::default() };
        let (model, report, set) = readout::train(&pipe, &rc)?;
        println!("{report}");
        println!("skipped: {} without logged evidence, {} reject/defer, {} unknown entities, {} chose evidence", set.skipped.legacy, set.skipped.no_label, set.skipped.unknown_entity, set.skipped.chose_evidence);
        for (g, w) in model.group_weight() {
            println!("  total |weight| on {g:?} features: {w:.2}");
        }
        for (name, g, w) in model.weights().into_iter().take(8) {
            println!("  {name:<26} {g:?}  {w:+.2}");
        }
        let better = report.learned_loo.top1 > report.reservoir_rule.top1;
        println!("{}", if better { "held-out top-1 beats the untrained reservoir rule" } else { "WARNING: held-out top-1 does not beat the untrained reservoir rule; do not use this readout" });
        model.save(file)?;
        println!("saved {file}");
        return Ok(());
    }
    if let Some(file) = &readout_file {
        pipe.set_readout(Readout::load(file)?)?;
        eprintln!("[ranking with learned readout {file}]");
    }

    let known: Vec<String> = pipe.kb.entities().iter().map(|e| e.name.clone()).collect();
    let explainer: Box<dyn Explainer> = if let Some(m) = &claude {
        Box::new(ClaudeExplainer::from_env(m, known.clone())?)
    } else { match &llm {
        #[cfg(feature = "llm")]
        Some(dir) => {
            eprintln!("[loading model from {dir}]");
            Box::new(LlmExplainer::load(dir, known.clone(), max_new)?)
        }
        #[cfg(not(feature = "llm"))]
        Some(_) => bail!("built without the `llm` feature"),
        None => Box::new(TemplateExplainer),
    } };
    let _ = known;

    let outcome = match script.as_deref() {
        Some(kind) => {
            let top = pipe.analyze(&q)?.brief.candidates[0].entity.clone();
            let d = match kind {
                "approve" => Decision::Approve { entity: top, note: "scripted".into() },
                "reject" => Decision::Reject { note: "scripted".into() },
                "defer" => Decision::Defer { note: "scripted".into() },
                o => bail!("--script must be approve, reject or defer, not {o:?}"),
            };
            let mut h = ScriptedDecisionMaker { script: VecDeque::from([d]) };
            let r = pipe.run(&q, explainer.as_ref(), &mut h)?;
            println!("{}", human::render(&r.analysis.brief, &r.explanation));
            r
        }
        None => {
            let mut h = StdioDecisionMaker { input: BufReader::new(stdin()), output: stdout(), name: operator };
            pipe.run(&q, explainer.as_ref(), &mut h)?
        }
    };
    println!("\nRECORDED: {} {}{}", outcome.decision.kind(), outcome.decision.entity().unwrap_or(""), if outcome.decision.note().is_empty() { String::new() } else { format!(" ({})", outcome.decision.note()) });
    println!("(nothing was executed; the decision is stored in the knowledge base as a :Decision node, kb ts {})", pipe.kb.db.last_ts());
    Ok(())
}
