//! Stage 5 — **human decision maker**.
//!
//! The pipeline never acts. It renders a brief, asks a [`DecisionMaker`], and records the answer.
//! [`StdioDecisionMaker`] is the interactive operator; [`ScriptedDecisionMaker`] exists for tests
//! and demos and is labelled as scripted in the audit record.

use crate::brief::Brief;
use crate::explain::Explanation;
use anyhow::{bail, Result};
use std::collections::VecDeque;
use std::io::{BufRead, Write};

#[derive(Clone, Debug, PartialEq)]
pub enum Decision {
    /// Act on candidate `entity` (a name from the brief).
    Approve { entity: String, note: String },
    /// The machine's candidates are not right; nothing is confirmed.
    Reject { note: String },
    /// Not enough to decide; come back later.
    Defer { note: String },
    /// The operator names a different entity (must exist in the knowledge base).
    Override { entity: String, note: String },
}

impl Decision {
    pub fn kind(&self) -> &'static str {
        match self {
            Decision::Approve { .. } => "approve",
            Decision::Reject { .. } => "reject",
            Decision::Defer { .. } => "defer",
            Decision::Override { .. } => "override",
        }
    }
    pub fn note(&self) -> &str {
        match self {
            Decision::Approve { note, .. } | Decision::Reject { note } | Decision::Defer { note } | Decision::Override { note, .. } => note,
        }
    }
    pub fn entity(&self) -> Option<&str> {
        match self {
            Decision::Approve { entity, .. } | Decision::Override { entity, .. } => Some(entity),
            _ => None,
        }
    }
}

pub trait DecisionMaker {
    /// Label stored with the audit record (e.g. the operator's name, or `scripted`).
    fn operator(&self) -> String;
    fn decide(&mut self, brief: &Brief, explanation: &Explanation) -> Result<Decision>;
}

/// What the operator sees.
pub fn render(brief: &Brief, ex: &Explanation) -> String {
    let mut s = brief.facts();
    s.push_str(&format!("\nEXPLANATION [{}{}]:\n{}\n", ex.source, if ex.grounded { "" } else { ", NOT grounded" }, ex.text));
    for w in &ex.warnings {
        s.push_str(&format!("  ! {w}\n"));
    }
    s
}

pub struct StdioDecisionMaker<R: BufRead, W: Write> {
    pub input: R,
    pub output: W,
    pub name: String,
}

impl<R: BufRead, W: Write> DecisionMaker for StdioDecisionMaker<R, W> {
    fn operator(&self) -> String {
        self.name.clone()
    }
    fn decide(&mut self, brief: &Brief, ex: &Explanation) -> Result<Decision> {
        writeln!(self.output, "{}", render(brief, ex))?;
        loop {
            writeln!(
                self.output,
                "Decision: [1-{}] approve that candidate | [r]eject | [d]efer | [o] <entity> override | (add a note after '--')",
                brief.candidates.len().max(1)
            )?;
            write!(self.output, "> ")?;
            self.output.flush()?;
            let mut line = String::new();
            if self.input.read_line(&mut line)? == 0 {
                bail!("input closed before a decision was made (nothing was recorded)");
            }
            let (cmd, note) = match line.split_once("--") {
                Some((a, b)) => (a.trim().to_string(), b.trim().to_string()),
                None => (line.trim().to_string(), String::new()),
            };
            let mut it = cmd.split_whitespace();
            match it.next() {
                Some("r") | Some("reject") => return Ok(Decision::Reject { note }),
                Some("d") | Some("defer") => return Ok(Decision::Defer { note }),
                Some("o") | Some("override") => match it.next() {
                    Some(name) => return Ok(Decision::Override { entity: name.to_string(), note }),
                    None => writeln!(self.output, "override needs an entity name")?,
                },
                Some(k) => match k.parse::<usize>() {
                    Ok(i) if i >= 1 && i <= brief.candidates.len() => {
                        return Ok(Decision::Approve { entity: brief.candidates[i - 1].entity.clone(), note })
                    }
                    _ => writeln!(self.output, "not understood: {k:?}")?,
                },
                None => writeln!(self.output, "empty input")?,
            }
        }
    }
}

pub struct ScriptedDecisionMaker {
    pub script: VecDeque<Decision>,
}

impl DecisionMaker for ScriptedDecisionMaker {
    fn operator(&self) -> String {
        "scripted".into()
    }
    fn decide(&mut self, _b: &Brief, _e: &Explanation) -> Result<Decision> {
        self.script.pop_front().ok_or_else(|| anyhow::anyhow!("scripted decision maker has no more decisions"))
    }
}
