"""Out-of-roster reject post-pass on a generic-format jersey state.

Estimates a per-team number set from identities with strong vote evidence
(nv >= --high accepted reads), pooled per game with a side flip between halves
(scope game) or per half (scope half), then nulls the jersey of any player
identity whose committed number is outside the set. Vote counts come from a
GSR_DUMP_MEMBERS dump of the BASE assembler run (record-aligned to the
composed tree), so the pass needs no re-assembly.

    python roster_reject.py --dump-dir /mnt/d/jersey-lab/chain_compose/dump_k1jr \
        --in-tag tun_k1jr_sportsl80 --out-tag tun_k1jr_sportsl80_rej30 \
        --jersey-tag jnrstar2_u02_c08 --high 30 --scope game
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

PREDS = Path("/home/dxdxxd/projects/soccer-app/gsr/out/preds/SoccerNetGS-valid")
CACHE = Path("/home/dxdxxd/projects/soccer-app/gsr/out/valid")
GT = Path("/mnt/d/datasets/soccernet2025/valid")
OCR_CONF_TH = 0.5


def team_key(scope: str, game: str, half: str, side: str):
    if scope == "half":
        return (game, half, side)
    flip = {"left": "right", "right": "left"}
    return (game, side if half == "1" else flip.get(side, side))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--base-tag", required=True, help="tag emitted by the dump run (record-aligned)")
    ap.add_argument("--in-tag", required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--jersey-tag", required=True)
    ap.add_argument("--high", type=int, default=30)
    ap.add_argument("--scope", choices=("game", "half"), default="game")
    args = ap.parse_args()

    seqs = sorted(p.stem for p in (PREDS / args.in_tag / "data").glob("*.json"))
    docs, nv_of, nb_of, num_of, side_of, role_of, grp = {}, {}, {}, {}, {}, {}, {}
    for s in seqs:
        rows = pickle.load(open(args.dump_dir / f"{s}.pkl", "rb"))
        doc = json.load(open(PREDS / args.in_tag / "data" / f"{s}.json"))
        docs[s] = doc
        preds = doc["predictions"]
        # merge-gap interpolations are appended after the len(records) aligned rows
        assert len(rows) <= len(preds), f"{s}: {len(rows)} dump rows vs {len(preds)} predictions"
        base_doc = json.load(open(PREDS / args.base_tag / "data" / f"{s}.json"))
        base_preds = base_doc["predictions"]
        assert all(a["image_id"] == b["image_id"]
                   for a, b in zip(base_preds[: len(rows)], preds[: len(rows)])), \
            f"{s}: record order differs between {args.base_tag} and {args.in_tag}"
        preds = preds[: len(rows)]
        jer = pickle.load(open(CACHE / s / f"jersey_{args.jersey_tag}.pkl", "rb"))["frames"]
        info = json.load(open(GT / s / "Labels-GameState.json"))["info"]
        grp[s] = (info["game_id"], info["game_time_start"].split(" - ")[0])
        base2comp = {}
        nv, nb = Counter(), Counter()
        for (fi, j, base_id, _, _, _, _), p in zip(rows, preds):
            cid = p["track_id"]
            if base2comp.setdefault(base_id, cid) != cid:
                raise SystemExit(f"{s}: base id {base_id} maps to two composed ids")
            nb[cid] += 1
            nums, confs = jer[fi]
            if j < len(nums) and int(nums[j]) >= 0 and float(confs[j]) >= OCR_CONF_TH:
                nv[cid] += 1
            at = p["attributes"]
            num_of.setdefault(s, {})[cid] = at.get("jersey")
            side_of.setdefault(s, {})[cid] = at.get("team")
            role_of.setdefault(s, {})[cid] = at.get("role")
        nv_of[s], nb_of[s] = nv, nb

    roster = defaultdict(set)
    for s in seqs:
        for cid, n in num_of[s].items():
            if n is not None and role_of[s][cid] == "player" and nv_of[s][cid] >= args.high:
                roster[team_key(args.scope, *grp[s], side_of[s][cid])].add(int(n))
    print(f"scope={args.scope} high={args.high} rosters:")
    for k in sorted(roster, key=str):
        print("  ", k, sorted(roster[k]))

    ch = Counter()
    nulled = []
    for s in seqs:
        kill = set()
        for cid, n in num_of[s].items():
            if n is None or role_of[s][cid] != "player":
                continue
            r = roster.get(team_key(args.scope, *grp[s], side_of[s][cid]))
            if r and int(n) not in r:
                kill.add(cid)
                nulled.append((s, cid, n, nv_of[s][cid], nb_of[s][cid]))
        dst = PREDS / args.out_tag / "data"
        dst.mkdir(parents=True, exist_ok=True)
        for p in docs[s]["predictions"]:
            if p["track_id"] in kill and p["attributes"].get("role") == "player":
                if p["attributes"]["jersey"] is not None:
                    ch["rows_nulled"] += 1
                p["attributes"]["jersey"] = None
        json.dump(docs[s], open(dst / f"{s}.json", "w"))
    print(f"identities nulled {len(nulled)}, rows nulled {ch['rows_nulled']}")
    for s, cid, n, nv, nb in nulled:
        print(f"  {s} id {cid} number {n} nv {nv} nb {nb}")
    print(f"wrote {PREDS / args.out_tag / 'data'}")


if __name__ == "__main__":
    main()
