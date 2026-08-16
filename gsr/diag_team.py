"""Diagnose team assignment vs GT (soccer_eda env).

For each seq: reproduce the s3b team assignment (KMeans(2) on OSNet mean
embeddings, sides by mean pitch-x), map each tracker track to its GT track/team,
and report per-seq accuracy plus whether a seq's errors are a global left<->right
swap (side-labelling bug) or scattered (clustering bug).

    python gsr/diag_team.py <split> <seqs|all>
"""
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")
sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")
import numpy as np
from sklearn.cluster import KMeans

import common as C
from s3b_assemble import fill_homographies, project, iou_to, gt_by_frame, _left_cluster, SIDE


def run(split, seqs):
    gtot = cor = 0
    for seq in seqs:
        base = C.OUT_ROOT / split / seq
        emb = C.load(base / "emb.pkl")["frames"]
        calib = C.load(base / "calib.pkl")["frames"]
        tk = C.load(base / "track_base.pkl")["records"]
        labels = C.load_labels(split, seq)
        image_ids = C.frame_index(labels)[0]
        H = fill_homographies(calib)
        emb_lookup = {(fi, j): emb[fi][j] for fi in range(len(emb)) for j in range(len(emb[fi]))}
        by = defaultdict(list)
        for r in tk:
            by[r[2]].append(r)
        role_of, emb_of, px, boxx = {}, {}, {}, {}
        for tid, rs in by.items():
            role_of[tid] = C.CLS_TO_ROLE[Counter(r[3] for r in rs).most_common(1)[0][0]]
            v = [emb_lookup[(fi, j)] for fi, j, *_ in rs if (fi, j) in emb_lookup]
            emb_of[tid] = np.mean(v, 0) if v else np.zeros(512, np.float32)
            bx = [(fi, project(H[fi], (b[0] + b[2]) / 2, b[3])[0]) for fi, j, _, _, b, _ in rs]
            boxx[tid] = bx
            px[tid] = float(np.median([x for _, x in bx])) if bx else 0.0
        field = [t for t in by if role_of[t] in ("player", "goalkeeper")]
        team_of = {}
        if len(field) >= 2:
            X = np.stack([emb_of[t] / max(np.linalg.norm(emb_of[t]), 1e-9) for t in field])
            lab = KMeans(2, n_init=10, random_state=0).fit_predict(X)
            cl = {c: [t for t, l in zip(field, lab) if l == c] for c in (0, 1)}
            lc = _left_cluster(cl, px, role_of, boxx)
            for t, l in zip(field, lab):
                team_of[t] = "left" if l == lc else "right"
        if SIDE != "gk":
            for t in field:
                if role_of[t] == "goalkeeper":
                    team_of[t] = "right" if px[t] > 0 else "left"

        # GT team per tracker track via IoU-majority
        gtf = gt_by_frame(labels, image_ids)
        gt_team_track = {}
        for tid in field:
            votes = Counter()
            for fi, j, _, _, b, _ in by[tid]:
                if fi in gtf and len(gtf[fi][0]):
                    gb = gtf[fi][0]; k = int(iou_to(gb, b).argmax())
                    if iou_to(gb, b)[k] >= 0.5 and gtf[fi][2][k] in ("left", "right"):
                        votes[gtf[fi][2][k]] += 1
            if votes:
                gt_team_track[tid] = votes.most_common(1)[0][0]
        n = sum(1 for t in gt_team_track if team_of.get(t) == gt_team_track[t])
        m = len(gt_team_track)
        # swap test: accuracy if we flip our labels
        nsw = sum(1 for t in gt_team_track if ({"left": "right", "right": "left"}[team_of[t]]) == gt_team_track[t])
        tag = " SWAP-better" if nsw > n else ""
        gtot += m; cor += max(n, 0)
        print(f"  {seq}: team acc {n}/{m}={n/max(1,m):.2f}  (if swapped {nsw}/{m}={nsw/max(1,m):.2f}){tag}")
    print(f"TOTAL team acc {cor}/{gtot} = {cor/max(1,gtot):.3f}")


if __name__ == "__main__":
    split = sys.argv[1]
    seqs = C.list_seqs(split) if sys.argv[2] == "all" else sys.argv[2].split(",")
    run(split, seqs)
