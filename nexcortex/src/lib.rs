//! # nexcortex
//!
//! ```text
//! NexGraph (persistent knowledge)          kb.rs        nexgraph
//!   -> spectral / diffusion features       features.rs  nexgraph physics.* / spectral.* / algo.*
//!   -> connectome RNN (working memory)     memory.rs    human_connectome_rnn (hcrnn)
//!   -> learned readout (optional)          readout.rs   graph + reservoir trajectory + human decisions
//!   -> LLM explanation layer               explain.rs   nexgen-llm (Burn)  [feature `llm`]
//!   -> human decision maker                human.rs     decision recorded back into NexGraph
//! ```
//! [`pipeline::Pipeline`] wires the stages; `src/bin/nexcortex.rs` is the command-line front end.

pub mod brief;
pub mod claude;
pub mod explain;
pub mod features;
pub mod human;
pub mod kb;
pub mod memory;
pub mod pipeline;
pub mod readout;

pub use brief::{Brief, Candidate, Evidence};
pub use claude::ClaudeExplainer;
pub use explain::{Explainer, Explanation, TemplateExplainer};
#[cfg(feature = "llm")]
pub use explain::LlmExplainer;
pub use human::{Decision, DecisionMaker, ScriptedDecisionMaker, StdioDecisionMaker};
pub use kb::{demo_builder, DecisionRecord, KbBuilder, KnowledgeBase};
pub use readout::{HumanHistory, Readout, ReadoutConfig};
pub use pipeline::{EventSpec, Outcome, Pipeline, PipelineConfig, Query};
