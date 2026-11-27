//! Claude (Anthropic Messages API) as the stage-4 explainer.
//!
//! Same contract as the local [`LlmExplainer`](crate::explain): the model sees only
//! [`Brief::facts`], its text is grounding-checked, any failure falls back to the template, and the
//! explanation never changes the ranking. The HTTP layer is a [`Transport`] so tests use a mock
//! and production uses `curl` (the API key is passed on stdin, never on the command line).

use crate::brief::Brief;
use crate::explain::{brief_entities, ungrounded_mentions, Explainer, Explanation, TemplateExplainer};
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Instant;

pub const SYSTEM: &str = "You explain a decision-support result to a human operator. Use ONLY the facts provided. \
Do not invent entities, numbers or causes. In at most four sentences say why the first candidate ranked first, \
what the diffusion-only rank adds, and what the operator should verify.";

/// One HTTPS POST of a JSON body; returns the response body.
pub trait Transport: Send + Sync {
    fn post(&self, url: &str, api_key: &str, body: &str) -> Result<String>;
}

pub struct CurlTransport {
    pub timeout_secs: u32,
}

impl Transport for CurlTransport {
    fn post(&self, url: &str, api_key: &str, body: &str) -> Result<String> {
        if api_key.contains(['"', '\n', '\r', '\\']) {
            bail!("API key contains illegal characters");
        }
        let cfg = format!(
            "header = \"x-api-key: {api_key}\"\nheader = \"anthropic-version: 2023-06-01\"\nheader = \"content-type: application/json\"\n"
        );
        let dir = std::env::temp_dir();
        let path = dir.join(format!("nexcortex_req_{}_{}.json", std::process::id(), rand_tag()));
        std::fs::write(&path, body)?;
        let mut child = Command::new("curl")
            .args(["-sS", "--max-time", &self.timeout_secs.to_string(), "-K", "-", "--data-binary"])
            .arg(format!("@{}", path.display()))
            .args(["-w", "\n%{http_code}"])
            .arg(url)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("spawning curl")?;
        child.stdin.take().unwrap().write_all(cfg.as_bytes())?;
        let out = child.wait_with_output()?;
        let _ = std::fs::remove_file(&path);
        if !out.status.success() {
            bail!("curl failed: {}", String::from_utf8_lossy(&out.stderr).trim());
        }
        let s = String::from_utf8_lossy(&out.stdout).to_string();
        let (resp, code) = s.rsplit_once('\n').ok_or_else(|| anyhow!("malformed curl output"))?;
        if !code.starts_with('2') {
            bail!("HTTP {code}: {}", resp.chars().take(300).collect::<String>());
        }
        Ok(resp.to_string())
    }
}

fn rand_tag() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
}

/// Per-run counters for the evaluation report.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub calls: usize,
    pub api_errors: usize,
    pub grounded: usize,
    pub unverified: usize,
    pub latency_ms: Vec<u128>,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

pub struct ClaudeExplainer {
    transport: Box<dyn Transport>,
    api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: u32,
    pub system: String,
    known: Vec<String>,
    fallback: TemplateExplainer,
    stats: Mutex<Stats>,
}

impl ClaudeExplainer {
    pub fn new(api_key: String, model: &str, known: Vec<String>) -> Self {
        let base_url = std::env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| "https://api.anthropic.com".into());
        Self::with_transport(Box::new(CurlTransport { timeout_secs: 60 }), api_key, base_url, model, known)
    }

    pub fn from_env(model: &str, known: Vec<String>) -> Result<Self> {
        let key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| anyhow!("ANTHROPIC_API_KEY is not set"))?;
        Ok(Self::new(key, model, known))
    }

    pub fn with_transport(t: Box<dyn Transport>, api_key: String, base_url: String, model: &str, known: Vec<String>) -> Self {
        ClaudeExplainer {
            transport: t,
            api_key,
            base_url,
            model: model.to_string(),
            max_tokens: 400,
            system: SYSTEM.to_string(),
            known,
            fallback: TemplateExplainer,
            stats: Mutex::new(Stats::default()),
        }
    }

    pub fn stats(&self) -> Stats {
        self.stats.lock().unwrap().clone()
    }

    fn request(&self, brief: &Brief) -> Result<String> {
        let body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "system": self.system,
            "messages": [{"role": "user", "content": format!("{}\nExplain this result.", brief.facts())}],
        })
        .to_string();
        let t0 = Instant::now();
        let resp = self.transport.post(&format!("{}/v1/messages", self.base_url.trim_end_matches('/')), &self.api_key, &body);
        let ms = t0.elapsed().as_millis();
        let mut st = self.stats.lock().unwrap();
        st.calls += 1;
        st.latency_ms.push(ms);
        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                st.api_errors += 1;
                return Err(e);
            }
        };
        let v: Value = serde_json::from_str(&resp).map_err(|e| {
            st.api_errors += 1;
            anyhow!("bad JSON from API: {e}")
        })?;
        if let Some(err) = v.get("error") {
            st.api_errors += 1;
            bail!("API error: {}", err);
        }
        st.input_tokens += v["usage"]["input_tokens"].as_u64().unwrap_or(0);
        st.output_tokens += v["usage"]["output_tokens"].as_u64().unwrap_or(0);
        let text: String = v["content"]
            .as_array()
            .map(|a| a.iter().filter(|b| b["type"] == "text").filter_map(|b| b["text"].as_str()).collect::<Vec<_>>().join(""))
            .unwrap_or_default();
        Ok(text.trim().to_string())
    }
}

impl Explainer for ClaudeExplainer {
    fn explain(&self, brief: &Brief) -> Result<Explanation> {
        let fallback = |why: String| -> Result<Explanation> {
            let mut e = self.fallback.explain(brief)?;
            e.warnings.push(why);
            Ok(e)
        };
        let text = match self.request(brief) {
            Ok(t) => t,
            Err(e) => return fallback(format!("Claude explanation unavailable ({e}); showing the deterministic template")),
        };
        if text.is_empty() {
            return fallback("Claude returned no text; showing the deterministic template".into());
        }
        let allowed = brief_entities(brief);
        let stray = ungrounded_mentions(&text, &self.known, &allowed);
        let names_top = brief.candidates.first().map_or(true, |c| text.to_lowercase().contains(&c.entity.to_lowercase()));
        let mut warnings = Vec::new();
        if !names_top {
            warnings.push("Claude text does not mention the top candidate".to_string());
        }
        if !stray.is_empty() {
            warnings.push(format!("Claude text mentions entities outside the brief: {}", stray.join(", ")));
        }
        if warnings.is_empty() {
            self.stats.lock().unwrap().grounded += 1;
            return Ok(Explanation { text, source: "claude", grounded: true, warnings });
        }
        self.stats.lock().unwrap().unverified += 1;
        let mut t = self.fallback.explain(brief)?;
        warnings.push("model text failed the grounding check and is shown only as unverified".to_string());
        t.text = format!("{}\n\n[unverified model text] {}", t.text, text);
        t.source = "claude";
        t.grounded = false;
        t.warnings = warnings;
        Ok(t)
    }
}
