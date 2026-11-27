//! Stage 1 — **persistent knowledge** (NexGraph).
//!
//! Entities are `:Entity {name, kind}` nodes; relations are undirected 2-member hyperedges
//! labelled `REL` (weight = tie strength); groups are n-member `GROUP` hyperedges. Operator
//! decisions are written back as `:Decision` nodes (see [`KnowledgeBase::record_decision`]),
//! so what a human confirmed survives restarts and feeds later runs.

use anyhow::{anyhow, bail, Context, Result};
use hcrnn::connectome::{Connectome, Region};
use nexgraph::{props, Database, Id, Value};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Entity {
    pub id: Id,
    pub name: String,
    pub kind: String,
}

/// One logged operator decision, replayable as a training example.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionRecord {
    pub id: Id,
    /// `approve` / `reject` / `defer` / `override`.
    pub kind: String,
    pub operator: String,
    pub question: String,
    pub note: String,
    /// Evidence shown to the operator, in observation order (prior seeds excluded).
    pub evidence: Vec<(String, f64)>,
    /// Entity the operator picked (approve / override).
    pub chosen: Option<String>,
    /// Machine candidate the operator turned down (override / reject).
    pub refused: Option<String>,
}

pub struct KnowledgeBase {
    pub db: Database,
    entities: Vec<Entity>,
    by_name: HashMap<String, usize>,
}

/// Collects entities/relations and commits them in a single transaction.
#[derive(Default)]
pub struct KbBuilder {
    entities: Vec<(String, String)>,
    rels: Vec<(String, String, String, f64)>,
    groups: Vec<(String, Vec<String>, f64)>,
}

impl KbBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn entity(mut self, name: &str, kind: &str) -> Self {
        self.entities.push((name.into(), kind.into()));
        self
    }
    pub fn relate(mut self, a: &str, b: &str, rel: &str, weight: f64) -> Self {
        self.rels.push((a.into(), b.into(), rel.into(), weight));
        self
    }
    pub fn group(mut self, name: &str, members: &[&str], weight: f64) -> Self {
        self.groups.push((name.into(), members.iter().map(|s| s.to_string()).collect(), weight));
        self
    }
    /// Write everything into `db` (in-memory or opened from a directory).
    pub fn commit_into(self, db: Database) -> Result<KnowledgeBase> {
        let mut tx = db.begin();
        tx.create_property_index("Entity", "name");
        let mut ids: HashMap<String, Id> = HashMap::new();
        for (name, kind) in &self.entities {
            if ids.contains_key(name) {
                bail!("duplicate entity name {name:?}");
            }
            let id = tx.create_node(
                &["Entity"],
                props([("name", Value::Str(name.clone())), ("kind", Value::Str(kind.clone()))]),
            )?;
            ids.insert(name.clone(), id);
        }
        let get = |n: &str| ids.get(n).copied().ok_or_else(|| anyhow!("relation names unknown entity {n:?}"));
        for (a, b, rel, w) in &self.rels {
            if !(*w > 0.0 && w.is_finite()) {
                bail!("relation {a}-{b} needs a positive finite weight");
            }
            let (ia, ib) = (get(a)?, get(b)?);
            tx.create_hyperedge("REL", &[ia, ib], &[], *w, props([("rel", Value::Str(rel.clone()))]))?;
        }
        for (name, members, w) in &self.groups {
            let m: Vec<Id> = members.iter().map(|x| get(x)).collect::<Result<_>>()?;
            tx.create_hyperedge("GROUP", &m, &[], *w, props([("name", Value::Str(name.clone()))]))?;
        }
        tx.commit()?;
        KnowledgeBase::from_db(db)
    }
    pub fn build(self) -> Result<KnowledgeBase> {
        self.commit_into(Database::in_memory())
    }
}

impl KnowledgeBase {
    /// Open (or create) a durable knowledge base in `dir`. An existing database is loaded as is.
    pub fn open(dir: impl AsRef<Path>) -> Result<KnowledgeBase> {
        let db = Database::open(dir.as_ref()).map_err(|e| anyhow!("open {:?}: {e}", dir.as_ref()))?;
        Self::from_db(db)
    }

    pub fn from_db(db: Database) -> Result<KnowledgeBase> {
        let r = db.query("MATCH (n:Entity) RETURN n, n.name, n.kind")?;
        let mut entities = Vec::new();
        for row in &r.rows {
            let (Value::Node(id), Some(name), Some(kind)) = (&row[0], row[1].as_str(), row[2].as_str()) else {
                bail!("malformed :Entity row {row:?}");
            };
            entities.push(Entity { id: *id, name: name.to_string(), kind: kind.to_string() });
        }
        entities.sort_by(|a, b| a.name.cmp(&b.name));
        let by_name = entities.iter().enumerate().map(|(i, e)| (e.name.clone(), i)).collect();
        Ok(KnowledgeBase { db, entities, by_name })
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }
    pub fn entity(&self, name: &str) -> Option<&Entity> {
        self.by_name.get(name).map(|&i| &self.entities[i])
    }
    pub fn entity_by_id(&self, id: Id) -> Option<&Entity> {
        self.entities.iter().find(|e| e.id == id)
    }
    /// Entity index (position in [`Self::entities`]) of a node id.
    pub fn index_of(&self, id: Id) -> Option<usize> {
        self.entities.iter().position(|e| e.id == id)
    }

    /// Weighted adjacency (sum of `REL` weights between each pair), indexed like [`Self::entities`].
    pub fn adjacency(&self) -> Result<Vec<f64>> {
        let n = self.entities.len();
        let pos: HashMap<Id, usize> = self.entities.iter().enumerate().map(|(i, e)| (e.id, i)).collect();
        let r = self.db.query("MATCH (a:Entity)-[r:REL]-(b:Entity) RETURN a, b, r.weight")?;
        let mut w = vec![0.0; n * n];
        for row in &r.rows {
            if let (Value::Node(a), Value::Node(b), Some(x)) = (&row[0], &row[1], row[2].as_f64()) {
                let (i, j) = (pos[a], pos[b]);
                if i != j {
                    w[i * n + j] += x; // each undirected edge appears once per orientation
                }
            }
        }
        Ok(w)
    }

    /// Number of connected components of the `REL` graph (spectral stages need exactly 1).
    pub fn components(&self) -> Result<usize> {
        let n = self.entities.len();
        let w = self.adjacency()?;
        let mut seen = vec![false; n];
        let mut c = 0;
        for s in 0..n {
            if seen[s] {
                continue;
            }
            c += 1;
            let mut st = vec![s];
            seen[s] = true;
            while let Some(u) = st.pop() {
                for v in 0..n {
                    if w[u * n + v] > 0.0 && !seen[v] {
                        seen[v] = true;
                        st.push(v);
                    }
                }
            }
        }
        Ok(c)
    }

    /// Entities with a positive `confirmed` count, i.e. things an operator has confirmed before.
    pub fn confirmed(&self) -> Result<Vec<(String, i64)>> {
        let r = self.db.query("MATCH (n:Entity) WHERE n.confirmed > 0 RETURN n.name, n.confirmed")?;
        Ok(r.rows.iter().filter_map(|row| Some((row[0].as_str()?.to_string(), row[1].as_i64()?))).collect())
    }

    /// Wiring for the recurrent network: the knowledge graph itself.
    ///
    /// Tie strength -> connection weight. Stronger ties are shorter fibres (25..75 mm, i.e. 1..3
    /// steps of conduction delay at the default 5 mm/ms and dt 5 ms). `positions` (3-D, e.g. a
    /// spectral embedding) and `systems` (e.g. community labels) are metadata for the regions.
    pub fn to_connectome(&self, positions: &[[f64; 3]], systems: &[String]) -> Result<Connectome> {
        let n = self.entities.len();
        if positions.len() != n || systems.len() != n {
            bail!("positions/systems must have one entry per entity");
        }
        let w = self.adjacency()?;
        let wmax = w.iter().cloned().fold(0.0, f64::max);
        if wmax <= 0.0 {
            bail!("the knowledge graph has no REL edges");
        }
        let len: Vec<f64> = w.iter().map(|&x| if x > 0.0 { 25.0 + 50.0 * (1.0 - x / wmax) } else { 0.0 }).collect();
        let regions = self
            .entities
            .iter()
            .enumerate()
            .map(|(i, e)| Region { name: e.name.clone(), hemi: '?', system: systems[i].clone(), pos: positions[i] })
            .collect();
        Connectome::new(regions, w, len, "nexgraph knowledge base").map_err(|e| anyhow!(e))
    }

    /// Persist an operator decision and update confirmation counters. Returns the `:Decision` node id.
    ///
    /// Legacy entry point: stores no evidence, so the decision cannot later be used to train the
    /// learned readout. Prefer [`Self::record_decision_logged`].
    pub fn record_decision(
        &mut self,
        kind: &str,
        question: &str,
        operator: &str,
        note: &str,
        confidence: f64,
        entities: &[String],
        confirm: &[String],
        reject: &[String],
    ) -> Result<Id> {
        self.record_decision_logged(kind, question, operator, note, confidence, entities, confirm, reject, &[])
    }

    /// Like [`Self::record_decision`], and also stores what is needed to replay the decision as a
    /// training example: the evidence the operator was shown (in order, **without** the
    /// auto-added prior seeds), the entity they chose (`confirm[0]`) and the machine pick they
    /// turned down (`reject[0]`).
    pub fn record_decision_logged(
        &mut self,
        kind: &str,
        question: &str,
        operator: &str,
        note: &str,
        confidence: f64,
        entities: &[String],
        confirm: &[String],
        reject: &[String],
        evidence: &[(String, f64)],
    ) -> Result<Id> {
        let wall = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
        let mut tx = self.db.begin();
        let mut p = props([
            ("kind", Value::Str(kind.into())),
            ("question", Value::Str(question.into())),
            ("operator", Value::Str(operator.into())),
            ("note", Value::Str(note.into())),
            ("confidence", Value::Float(confidence)),
            ("unix_time", Value::Int(wall)),
        ]);
        if !evidence.is_empty() {
            p.insert("ev_names".into(), Value::List(evidence.iter().map(|e| Value::Str(e.0.clone())).collect()));
            p.insert("ev_sev".into(), Value::List(evidence.iter().map(|e| Value::Float(e.1)).collect()));
            p.insert("logged".into(), Value::Bool(true));
            p.insert("chosen".into(), Value::Str(confirm.first().cloned().unwrap_or_default()));
            p.insert("refused".into(), Value::Str(reject.first().cloned().unwrap_or_default()));
        }
        let d = tx.create_node(&["Decision"], p)?;
        let ids: Vec<Id> = entities
            .iter()
            .map(|n| self.entity(n).map(|e| e.id).ok_or_else(|| anyhow!("unknown entity {n:?}")))
            .collect::<Result<_>>()?;
        // Logged decisions keep chosen/refused as properties and stay OUT of the graph: NexGraph's
        // diffusion and spectral procedures run over every hyperedge, so a DECIDED_ON edge would
        // let each recorded decision (including its own label) change the features it is later
        // trained on. Legacy (evidence-less) records keep the edge for compatibility.
        if !ids.is_empty() && evidence.is_empty() {
            tx.create_hyperedge("DECIDED_ON", &[d], &ids, 1.0, props([("kind", Value::Str(kind.into()))]))?;
        }
        for (names, key) in [(confirm, "confirmed"), (reject, "rejected")] {
            for n in names {
                let e = self.entity(n).ok_or_else(|| anyhow!("unknown entity {n:?}"))?;
                let cur = self
                    .db
                    .query(&format!("MATCH (n:Entity) WHERE n.name = '{}' RETURN n.{key}", n.replace('\'', "")))?
                    .rows
                    .first()
                    .and_then(|r| r[0].as_i64())
                    .unwrap_or(0);
                tx.set_prop(e.id, key, Value::Int(cur + 1))?;
            }
        }
        tx.commit().context("committing decision")?;
        Ok(d)
    }

    /// Decisions that carry their evidence (see [`Self::record_decision_logged`]), in the order
    /// they were made (node ids grow with every write). Also returns how many older decisions
    /// had no evidence stored and were left out.
    pub fn decision_log(&self) -> Result<(Vec<DecisionRecord>, usize)> {
        let r = self.db.query("MATCH (d:Decision) RETURN d, d.kind, d.operator, d.question, d.note, d.logged, d.ev_names, d.ev_sev, d.chosen, d.refused")?;
        let mut out = Vec::new();
        let mut legacy = 0usize;
        for row in &r.rows {
            let Value::Node(id) = &row[0] else { continue };
            if !matches!(row[5], Value::Bool(true)) {
                legacy += 1;
                continue;
            }
            let (Value::List(names), Value::List(sev)) = (&row[6], &row[7]) else {
                legacy += 1;
                continue;
            };
            let evidence: Vec<(String, f64)> = names.iter().zip(sev).filter_map(|(n, s)| Some((n.as_str()?.to_string(), s.as_f64()?))).collect();
            if evidence.len() != names.len() {
                legacy += 1;
                continue;
            }
            let opt = |v: &Value| v.as_str().filter(|s| !s.is_empty()).map(|s| s.to_string());
            let s = |i: usize| row[i].as_str().unwrap_or("").to_string();
            out.push(DecisionRecord { id: *id, kind: s(1), operator: s(2), question: s(3), note: s(4), evidence, chosen: opt(&row[8]), refused: opt(&row[9]) });
        }
        out.sort_by_key(|d| d.id);
        Ok((out, legacy))
    }

    /// All recorded decisions, oldest first: (kind, operator, question, note).
    pub fn decisions(&self) -> Result<Vec<(String, String, String, String)>> {
        let r = self.db.query("MATCH (d:Decision) RETURN d.kind, d.operator, d.question, d.note, d.unix_time ORDER BY d.unix_time")?;
        Ok(r.rows
            .iter()
            .map(|x| {
                let s = |i: usize| x[i].as_str().unwrap_or("").to_string();
                (s(0), s(1), s(2), s(3))
            })
            .collect())
    }
}

/// A small synthetic enterprise network used by the demo and the tests: three tightly knit zones
/// (DMZ, corporate, data) joined by a jump host and a service account. Not real data.
pub fn demo_builder() -> KbBuilder {
    let mut b = KbBuilder::new();
    let dmz = ["web-01", "web-02", "web-03", "lb-01", "waf-01"];
    let corp = ["ws-01", "ws-02", "ws-03", "ws-04", "ws-05", "ws-06", "alice", "bob", "carol", "ad-01"];
    let data = ["db-01", "db-02", "db-03", "backup-01", "vault-01"];
    for n in dmz {
        b = b.entity(n, if n.starts_with("web") { "server" } else { "network" });
    }
    for n in corp {
        let k = if n.starts_with("ws") { "workstation" } else if n == "ad-01" { "directory" } else { "user" };
        b = b.entity(n, k);
    }
    for n in data {
        b = b.entity(n, if n == "vault-01" { "secrets" } else { "data-store" });
    }
    b = b.entity("jump-01", "bastion").entity("svc-deploy", "service-account");
    let clique = |mut b: KbBuilder, xs: &[&str], w: f64, rel: &str| {
        for i in 0..xs.len() {
            for j in (i + 1)..xs.len() {
                b = b.relate(xs[i], xs[j], rel, w);
            }
        }
        b
    };
    b = clique(b, &dmz, 0.8, "connects_to");
    b = clique(b, &data, 0.8, "replicates_with");
    for (u, ws) in [("alice", ["ws-01", "ws-02"]), ("bob", ["ws-03", "ws-04"]), ("carol", ["ws-05", "ws-06"])] {
        for w in ws {
            b = b.relate(u, w, "logs_in_to", 0.9);
        }
        b = b.relate(u, "ad-01", "authenticates_with", 0.7);
    }
    b = clique(b, &["ws-01", "ws-02", "ws-03", "ws-04", "ws-05", "ws-06"], 0.4, "same_vlan");
    for w in ["ws-01", "ws-02", "ws-03", "ws-04", "ws-05", "ws-06"] {
        b = b.relate(w, "ad-01", "domain_joined", 0.6);
    }
    // cross-zone ties: weaker, few, and concentrated on two bridges
    b = b.relate("web-01", "jump-01", "admin_access", 0.5).relate("web-02", "jump-01", "admin_access", 0.4);
    b = b.relate("jump-01", "ad-01", "authenticates_with", 0.6).relate("jump-01", "ws-03", "ssh", 0.3);
    b = b.relate("svc-deploy", "ad-01", "authenticates_with", 0.5).relate("svc-deploy", "db-01", "deploys_to", 0.5);
    b = b.relate("svc-deploy", "web-03", "deploys_to", 0.4).relate("db-03", "ws-06", "reporting_query", 0.2);
    b = b.relate("vault-01", "svc-deploy", "issues_secrets_to", 0.4);
    b.group("pci-scope", &["web-01", "db-01", "db-02", "vault-01"], 0.05)
}
