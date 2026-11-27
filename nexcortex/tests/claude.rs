use anyhow::{bail, Result};
use nexcortex::claude::Transport;
use nexcortex::*;
use std::sync::{Arc, Mutex};

fn analysis() -> (nexcortex::pipeline::Analysis, Vec<String>) {
    let p = Pipeline::new(demo_builder().build().unwrap(), PipelineConfig::default()).unwrap();
    let q = Query { question: "who else?".into(), events: vec![EventSpec { entity: "web-01".into(), severity: 0.9, note: String::new() }] };
    let known: Vec<String> = p.kb.entities().iter().map(|e| e.name.clone()).collect();
    (p.analyze(&q).unwrap(), known)
}

struct Mock {
    reply: Result<String, String>,
    seen: Arc<Mutex<Vec<(String, String, String)>>>,
}
impl Transport for Mock {
    fn post(&self, url: &str, key: &str, body: &str) -> Result<String> {
        self.seen.lock().unwrap().push((url.into(), key.into(), body.into()));
        match &self.reply {
            Ok(s) => Ok(s.clone()),
            Err(e) => bail!("{e}"),
        }
    }
}
fn api_reply(text: &str) -> String {
    serde_json::json!({"content":[{"type":"text","text":text}],"usage":{"input_tokens":120,"output_tokens":40}}).to_string()
}
fn explainer(reply: Result<String, String>, known: Vec<String>) -> (ClaudeExplainer, Arc<Mutex<Vec<(String, String, String)>>>) {
    let seen = Arc::new(Mutex::new(vec![]));
    (ClaudeExplainer::with_transport(Box::new(Mock { reply, seen: seen.clone() }), "k-test".into(), "http://x/".into(), "claude-sonnet-5-5", known), seen)
}

#[test]
fn grounded_reply_is_accepted_and_request_is_facts_only() {
    let (a, known) = analysis();
    let top = a.brief.candidates[0].entity.clone();
    let (ex, seen) = explainer(Ok(api_reply(&format!("{top} ranked first because of its working-memory activity. Verify it."))), known);
    let e = ex.explain(&a.brief).unwrap();
    assert_eq!(e.source, "claude");
    assert!(e.grounded && e.warnings.is_empty());
    let s = seen.lock().unwrap();
    assert_eq!(s[0].0, "http://x/v1/messages");
    let body: serde_json::Value = serde_json::from_str(&s[0].2).unwrap();
    assert_eq!(body["model"], "claude-sonnet-5-5");
    assert!(body["system"].as_str().unwrap().contains("ONLY the facts"));
    assert_eq!(body["messages"][0]["content"].as_str().unwrap().contains(&a.brief.facts()), true);
    assert!(!s[0].2.contains("k-test"), "key must not appear in the body");
    let st = ex.stats();
    assert_eq!((st.calls, st.grounded, st.unverified, st.api_errors, st.input_tokens, st.output_tokens), (1, 1, 0, 0, 120, 40));
}

#[test]
fn ungrounded_or_topless_reply_is_marked_unverified_and_never_leads() {
    let (a, known) = analysis();
    let stray = known.iter().find(|k| !brief_names(&a).contains(k)).unwrap().clone();
    let (ex, _) = explainer(Ok(api_reply(&format!("{stray} is the culprit."))), known.clone());
    let e = ex.explain(&a.brief).unwrap();
    assert!(!e.grounded && e.text.contains("[unverified model text]") && e.text.starts_with("Given evidence"));
    assert_eq!(ex.stats().unverified, 1);
}
fn brief_names(a: &nexcortex::pipeline::Analysis) -> Vec<String> {
    nexcortex::explain::brief_entities(&a.brief)
}

#[test]
fn api_failures_fall_back_to_the_template() {
    let (a, known) = analysis();
    for reply in [Err("HTTP 401".to_string()), Ok("not json".into()), Ok(r#"{"error":{"type":"overloaded_error"}}"#.into()), Ok(api_reply("  "))] {
        let (ex, _) = explainer(reply, known.clone());
        let e = ex.explain(&a.brief).unwrap();
        assert_eq!(e.source, "template");
        assert!(e.grounded && !e.warnings.is_empty());
    }
}

#[test]
fn curl_transport_talks_http_to_a_local_server_and_hides_the_key_from_argv() {
    use std::io::{Read, Write};
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let (a, known) = analysis();
    let top = a.brief.candidates[0].entity.clone();
    let reply = api_reply(&format!("{top} leads."));
    let h = std::thread::spawn(move || {
        let (mut s, _) = l.accept().unwrap();
        let mut buf = vec![0u8; 65536];
        let mut got = 0;
        loop {
            let n = s.read(&mut buf[got..]).unwrap();
            got += n;
            let t = String::from_utf8_lossy(&buf[..got]).to_string();
            if let Some(i) = t.find("\r\n\r\n") {
                let cl: usize = t.to_lowercase().split("content-length:").nth(1).and_then(|x| x.split("\r\n").next()).map(|x| x.trim().parse().unwrap()).unwrap_or(0);
                if got >= i + 4 + cl { break; }
            }
        }
        let req = String::from_utf8_lossy(&buf[..got]).to_string();
        let resp = format!("HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}", reply.len(), reply);
        s.write_all(resp.as_bytes()).unwrap();
        req
    });
    let ex = ClaudeExplainer::new("sk-fake-123".into(), "claude-sonnet-5-5", known);
    let mut ex = ex;
    ex.base_url = format!("http://127.0.0.1:{port}");
    let e = ex.explain(&a.brief).unwrap();
    let req = h.join().unwrap().to_lowercase();
    assert_eq!(e.source, "claude", "{:?}", e.warnings);
    assert!(req.contains("x-api-key: sk-fake-123") && req.contains("anthropic-version: 2023-06-01") && req.contains("post /v1/messages"));
}
