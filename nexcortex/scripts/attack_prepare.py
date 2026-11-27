#!/usr/bin/env python3
"""Download MITRE ATT&CK Enterprise (STIX 2.1, live from GitHub) and write two TSV files.

techniques.tsv : id <TAB> name <TAB> tactic order index (first tactic in matrix order) <TAB> comma list of tactic shortnames
groups.tsv     : group name <TAB> comma list of parent technique ids it is documented to use

Only non-revoked, non-deprecated objects. Sub-techniques are rolled up to their parent technique.
Usage: attack_prepare.py OUT_DIR
"""
import collections, hashlib, json, os, sys, urllib.request

URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
raw = urllib.request.urlopen(URL, timeout=300).read()
open(os.path.join(out, "enterprise-attack.json"), "wb").write(raw)
d = json.loads(raw)
objs = [o for o in d["objects"] if not o.get("revoked") and not o.get("x_mitre_deprecated")]
version = [o.get("x_mitre_version") for o in d["objects"] if o["type"] == "x-mitre-collection"]
matrix = next(o for o in objs if o["type"] == "x-mitre-matrix")
by_id = {o["id"]: o for o in objs}
order = [by_id[r]["x_mitre_shortname"] for r in matrix["tactic_refs"] if r in by_id]
tidx = {t: i for i, t in enumerate(order)}
tid, tech = {}, {}
for o in objs:
    if o["type"] != "attack-pattern":
        continue
    ext = next(r["external_id"] for r in o["external_references"] if r.get("source_name") == "mitre-attack")
    tid[o["id"]] = ext.split(".")[0]
    if not o.get("x_mitre_is_subtechnique"):
        tac = sorted({p["phase_name"] for p in o.get("kill_chain_phases", []) if p["phase_name"] in tidx}, key=tidx.get)
        if tac:
            tech[ext] = (o["name"], tidx[tac[0]], tac)
uses = collections.defaultdict(set)
for o in objs:
    if o["type"] == "relationship" and o["relationship_type"] == "uses" and o["source_ref"] in by_id and by_id[o["source_ref"]]["type"] == "intrusion-set" and o["target_ref"] in tid:
        if tid[o["target_ref"]] in tech:
            uses[by_id[o["source_ref"]]["name"]].add(tid[o["target_ref"]])
with open(os.path.join(out, "techniques.tsv"), "w") as f:
    for k in sorted(tech):
        n, i, tac = tech[k]
        f.write(f"{k}\t{n}\t{i}\t{','.join(tac)}\n")
with open(os.path.join(out, "groups.tsv"), "w") as f:
    for g in sorted(uses):
        f.write(f"{g}\t{','.join(sorted(uses[g]))}\n")
print(f"ATT&CK Enterprise v{version[0] if version else '?'}  sha256 {hashlib.sha256(raw).hexdigest()}")
print(f"{len(tech)} techniques, {len(uses)} groups with documented technique use, {len(order)} tactics")
