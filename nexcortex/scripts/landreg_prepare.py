#!/usr/bin/env python3
"""Belize land-registry dataset -> attack_live format (techniques.tsv / groups.tsv).
entity = block cluster (10 consecutive blocks within a registration section); group = a proprietor,
its set = every cluster the party has ever held land in. Usage: landreg_prepare.py <registry_dir> <out_dir> [all|company]"""
import sys, pandas as pd
src, out = sys.argv[1], sys.argv[2]
kind = sys.argv[3] if len(sys.argv) > 3 else "all"
P = pd.read_csv(f"{src}/data/parcels.csv", usecols=["parcel_id", "registration_section", "block_no"])
P["cell"] = P.registration_section + "-C" + (P.block_no // 10).astype(str).str.zfill(3)
pr = pd.read_csv(f"{src}/data/proprietorship.csv", usecols=["parcel_id", "party_id"])
pt = pd.read_csv(f"{src}/data/parties.csv", usecols=["party_id", "party_type"])
m = pr.merge(P[["parcel_id", "cell"]], on="parcel_id").merge(pt, on="party_id")
cells = P.drop_duplicates("cell").sort_values("cell")[["cell", "registration_section", "block_no"]].reset_index(drop=True)
cells["name"] = cells.registration_section + " blocks " + (cells.block_no // 10 * 10).astype(str) + "+"
with open(f"{out}/techniques.tsv", "w") as f:
    for i, r in cells.iterrows():
        f.write(f"{r.cell}\t{r['name']}\t{i}\t{r.registration_section}\n")
n = 0
with open(f"{out}/groups.tsv", "w") as f:
    for (pid, ty), g in m.groupby(["party_id", "party_type"]):
        if kind == "company" and ty not in ("COMPANY", "GOVERNMENT", "COMMUNITY"): continue
        s = sorted(set(g.cell))
        if len(s) >= 5:
            f.write(f"{ty}-{pid}\t{','.join(s)}\n"); n += 1
print(len(cells), "entities", n, "groups")
