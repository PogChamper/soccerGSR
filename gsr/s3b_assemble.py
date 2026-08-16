"""Stage 3b (soccer_eda env): assemble predictions from cached track + features.

Reads track_<tag>.pkl (BoT-SORT association) plus det/jersey/emb/calib and emits
the GS-HOTA predictions JSON. All identity levers live here and run without
re-reading frames, so they iterate in seconds:
  role   = per-fragment majority class (fixes the detector's GK->player folding)
  team   = KMeans(2) on OSNet appearance; left/right by per-frame pitch-x vote
           ('framemean', robust to attack-direction bias); GK by own pitch-x sign
  jersey = visibility-gated per-track vote (ConvNeXt two-head logits by default, or
           a generic (number,conf) reader via GSR_JERSEY_FMT=generic)
  merge  = union-find of fragments by same (team,jersey) + motion continuity, which
           raises GS-AssA and pools jersey votes across a player's fragments
           (GSR_MERGE=smart, default; "" to disable). --relink is the old k-recip
           relink, kept only as an ablation (hurts — off).

    python gsr/s3b_assemble.py <split> <seqs|all> [--relink]
Env knobs: GSR_TAG, GSR_TRACK_TAG, GSR_JERSEY_TAG/GSR_JERSEY_FMT, GSR_VIS_TH,
     GSR_OCR_CONF_TH, GSR_MIN_VOTES, GSR_SIDE, GSR_MERGE(+_GAP/_SLACK/_SPEED/_APP_SIM),
     and ablations GSR_NO_JERSEY / GSR_NO_TEAM / GSR_PERBOX_ROLE / GSR_ORACLE.
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/home/dxdxxd/projects/soccer-app")
sys.path.insert(0, "/home/dxdxxd/projects/soccer-app/gsr")
sys.path.insert(0, "/home/dxdxxd/projects/football/ltpi-research")

import numpy as np
from sklearn.cluster import DBSCAN, KMeans

import common as C
from ltpi_research.advanced import k_reciprocal_rerank

VIS_TH = float(os.environ.get("GSR_VIS_TH", "0.7"))       # tuned jersey vote gates (12-valid sweep)
OCR_CONF_TH = float(os.environ.get("GSR_OCR_CONF_TH", "0.9"))
MIN_VOTES = int(os.environ.get("GSR_MIN_VOTES", "6"))
TAG = os.environ.get("GSR_TAG", "ours")
TRACK_TAG = os.environ.get("GSR_TRACK_TAG", "base")
CALIB_TAG = os.environ.get("GSR_CALIB_TAG", "")  # "" -> calib.pkl (PnLCalib), else calib_<tag>.pkl
JERSEY_TAG = os.environ.get("GSR_JERSEY_TAG", "")  # "" -> jersey.pkl, else jersey_<tag>.pkl
JERSEY_FMT = os.environ.get("GSR_JERSEY_FMT", "logits")  # logits (ConvNeXt two-head) | generic (number,conf)
NO_JERSEY = os.environ.get("GSR_NO_JERSEY") == "1"   # ablation: drop jersey attribute
NO_TEAM = os.environ.get("GSR_NO_TEAM") == "1"       # ablation: drop team attribute
PERBOX_ROLE = os.environ.get("GSR_PERBOX_ROLE") == "1"  # ablation: role per box (no GK-fix)
ORACLE = set(x for x in os.environ.get("GSR_ORACLE", "").split(",") if x)  # {team,jersey,role}
SIDE = os.environ.get("GSR_SIDE", "framemean")  # side heuristic: framemean | framemin | mean | min | gk
SIZE_WEIGHT = os.environ.get("GSR_SIZE_WEIGHT", "0") == "1"  # weight jersey votes by box height
CLEAN_CENT = os.environ.get("GSR_CLEAN_CENT", "0") == "1"  # drop occluded boxes from fragment centroids
INTERP_MAX = int(os.environ.get("GSR_INTERP", "10"))  # fill gaps up to N frames within a track (GSI)
HINTERP = os.environ.get("GSR_HINTERP", "0") == "1"  # interpolate homography across calib dropouts (vs hold-last)
LOC_FAILONLY = os.environ.get("GSR_LOC_FAILONLY", "0") == "1"  # oracle-loc only on calib-dropout frames
TEAM_PLAYERS_ONLY = os.environ.get("GSR_TEAM_PLAYERS_ONLY", "0") == "1"  # fit team KMeans on players only (exclude GK)


def _left_cluster(cl, px, role, boxx):
    """Which KMeans cluster (0/1) is the 'left' team (goal on the left).

    boxx: {track -> [(frame, pitch_x), ...]}. Broadcast cameras often crop the
    far goalkeeper, so track-level mean/min x is unreliable; the robust default
    'framemean' votes per frame for whichever cluster's players are on average
    further left (the defending team) and integrates over the clip."""
    if SIDE == "ensemble":
        votes = Counter()
        for h in ("framemean", "mean", "min"):
            votes[_one_side(h, cl, px, role, boxx)] += 1
        return votes.most_common(1)[0][0]
    if SIDE == "gkframe":
        # A goalkeeper-role track sitting near a goal (|x|>25 m) is the most reliable
        # side anchor; its cluster owns the goal on its side. Fall back to framemean
        # when no confident keeper is visible (broadcast often crops the far keeper).
        votes = Counter()
        for c in (0, 1):
            for t in cl[c]:
                if role[t] == "goalkeeper" and abs(px[t]) > 25:
                    votes[c if px[t] < 0 else 1 - c] += 1
        if votes:
            return votes.most_common(1)[0][0]
        return _one_side("framemean", cl, px, role, boxx)
    return _one_side(SIDE, cl, px, role, boxx)


def _one_side(SIDE, cl, px, role, boxx):
    if SIDE == "gk":
        gkx = {c: [px[t] for t in cl[c] if role[t] == "goalkeeper"] for c in (0, 1)}
        have = {c: v for c, v in gkx.items() if v}
        if len(have) == 2:
            return 0 if np.mean(have[0]) <= np.mean(have[1]) else 1
        if len(have) == 1:
            c = next(iter(have))
            return c if np.mean(have[c]) < 0 else 1 - c
    if SIDE == "mean":
        s = {c: (float(np.mean([px[t] for t in cl[c]])) if cl[c] else 0.0) for c in (0, 1)}
        return 0 if s[0] <= s[1] else 1
    if SIDE == "min":
        s = {c: (min((px[t] for t in cl[c]), default=0.0)) for c in (0, 1)}
        return 0 if s[0] <= s[1] else 1
    # 'framemin' / 'framemean': per-frame vote integrated over the clip
    by_frame = {0: defaultdict(list), 1: defaultdict(list)}
    for c in (0, 1):
        for t in cl[c]:
            for fi, x in boxx.get(t, []):
                by_frame[c][fi].append(x)
    votes0 = 0
    for fi in set(by_frame[0]) & set(by_frame[1]):
        a, b = by_frame[0][fi], by_frame[1][fi]
        va = min(a) if SIDE == "framemin" else float(np.mean(a))
        vb = min(b) if SIDE == "framemin" else float(np.mean(b))
        votes0 += 1 if va <= vb else -1
    if votes0 == 0:  # tie -> fall back to track-median min
        s = {c: (min((px[t] for t in cl[c]), default=0.0)) for c in (0, 1)}
        return 0 if s[0] <= s[1] else 1
    return 0 if votes0 > 0 else 1


def iou_to(gb, box):
    ix1 = np.maximum(gb[:, 0], box[0]); iy1 = np.maximum(gb[:, 1], box[1])
    ix2 = np.minimum(gb[:, 2], box[2]); iy2 = np.minimum(gb[:, 3], box[3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
    ab = (box[2] - box[0]) * (box[3] - box[1])
    return inter / np.maximum(aa + ab - inter, 1e-9)


def gt_by_frame(labels, image_ids):
    """{frame_idx: (boxes[N,4], roles[N], teams[N], jerseys[N], pitch[N])} for oracle
    override. pitch[k] is the GT bbox_pitch dict (or None) — used by ORACLE=loc."""
    pos = {iid: fi for fi, iid in enumerate(image_ids)}
    acc = defaultdict(lambda: ([], [], [], [], []))
    for a in labels["annotations"]:
        if a.get("supercategory") != "object" or a["image_id"] not in pos:
            continue
        at = a.get("attributes") or {}
        b = a["bbox_image"]
        box, role, team, jer, pit = acc[pos[a["image_id"]]]
        box.append((b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]))
        role.append(at.get("role")); team.append(at.get("team")); jer.append(at.get("jersey"))
        pit.append(a.get("bbox_pitch"))
    return {fi: (np.array(v[0], float), v[1], v[2], v[3], v[4]) for fi, v in acc.items()}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _size_w(bbox):
    """Up-weight larger (closer, more legible) crops in the jersey vote."""
    if not SIZE_WEIGHT:
        return 1.0
    return float(np.clip((bbox[3] - bbox[1]) / 120.0, 0.4, 3.0))


def frag_votes(rs, jer):
    """Collect a jersey vote object for one fragment from its per-box reader output.
    Supports the ConvNeXt two-head logits cache (default) and a generic (number,
    confidence) cache produced by an alternative reader."""
    if JERSEY_FMT == "generic":
        cnt, nv = defaultdict(float), 0
        for fi, j, _, _, bbox, _ in rs:
            nums, confs = jer[fi]
            if j >= len(nums):
                continue
            n, c = int(nums[j]), float(confs[j])
            if n < 0 or c < OCR_CONF_TH:
                continue
            cnt[n] += c * _size_w(bbox); nv += 1
        return (dict(cnt), nv)
    st, su, nv = np.zeros(10), np.zeros(10), 0
    for fi, j, _, _, bbox, _ in rs:
        vis, tens, units = jer[fi]
        if j >= len(vis) or sigmoid(vis[j]) < VIS_TH:
            continue
        if min(softmax(tens[j]).max(), softmax(units[j]).max()) < OCR_CONF_TH:
            continue
        w = _size_w(bbox)
        st += tens[j] * w; su += units[j] * w; nv += 1
    return (st, su, nv)


def pool_votes(objs):
    """Sum vote objects across merged fragments (jersey vote pooling)."""
    if JERSEY_FMT == "generic":
        cnt, nv = defaultdict(float), 0
        for c, n in objs:
            for k, v in c.items():
                cnt[k] += v
            nv += n
        return (dict(cnt), nv)
    st, su, nv = np.zeros(10), np.zeros(10), 0
    for s, u, n in objs:
        st += s; su += u; nv += n
    return (st, su, nv)


def decode_votes(obj):
    """Committed jersey number from a (possibly pooled) vote object, or None."""
    if JERSEY_FMT == "generic":
        cnt, nv = obj
        if nv < MIN_VOTES or not cnt:
            return None
        return int(max(cnt, key=cnt.get))
    st, su, nv = obj
    if nv < MIN_VOTES:
        return None
    t, u = int(np.argmax(st)), int(np.argmax(su))
    return u if t == 0 else t * 10 + u


def fill_homographies(calib):
    H = list(calib)
    good = [i for i in range(len(H)) if H[i] is not None]
    if not good:
        return H
    if HINTERP:
        # Linear element-wise interpolation of the (H[2,2]-normalised) homography
        # between consecutive successfully-calibrated frames, holding at the ends.
        # A rough but better-than-stale bridge across calibration dropouts.
        for a, b in zip(good, good[1:]):
            if b - a <= 1:
                continue
            Ha = np.asarray(H[a], float); Hb = np.asarray(H[b], float)
            Ha = Ha / Ha[2, 2]; Hb = Hb / Hb[2, 2]
            for f in range(a + 1, b):
                t = (f - a) / (b - a)
                Hf = Ha * (1 - t) + Hb * t
                H[f] = Hf / Hf[2, 2]
        for i in range(good[0]):
            H[i] = H[good[0]]
        for i in range(good[-1] + 1, len(H)):
            H[i] = H[good[-1]]
        return H
    last = None  # default: hold-last then back-fill
    for i in range(len(H)):
        if H[i] is not None:
            last = H[i]
        elif last is not None:
            H[i] = last
    nxt = None
    for i in range(len(H) - 1, -1, -1):
        if H[i] is not None:
            nxt = H[i]
        elif nxt is not None:
            H[i] = nxt
    return H


def project(H, x, y):
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def _occluded_boxes(records, iou_th=0.35):
    """Record indices whose box overlaps another box in the same frame (impure)."""
    by_frame = defaultdict(list)
    for i, r in enumerate(records):
        by_frame[r[0]].append((i, r[4]))
    occ = set()
    for items in by_frame.values():
        if len(items) < 2:
            continue
        b = np.array([it[1] for it in items], float)
        x1 = np.maximum.outer(b[:, 0], b[:, 0]); y1 = np.maximum.outer(b[:, 1], b[:, 1])
        x2 = np.minimum.outer(b[:, 2], b[:, 2]); y2 = np.minimum.outer(b[:, 3], b[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / np.maximum(area[:, None] + area[None, :] - inter, 1e-9)
        np.fill_diagonal(iou, 0.0)
        mx = iou.max(axis=1)
        for k, (idx, _) in enumerate(items):
            if mx[k] >= iou_th:
                occ.add(idx)
    return occ


def relink_records(records, emb_lookup, sim_th=0.20):
    """Merge original BoT-SORT tracks under frame-set disjointness (guarantees
    <=1 box per frame per cluster) using k-reciprocal appearance similarity on
    occlusion-free centroids. Fixes fragmentation without breaking per-frame ids."""
    occ = _occluded_boxes(records)
    tracks = defaultdict(list)  # original track id -> record indices
    for i, r in enumerate(records):
        tracks[r[2]].append(i)
    tids = sorted(tracks)
    if len(tids) < 2:
        return {records[i][2]: records[i][2] for i in range(len(records))}
    cents, framesets = [], []
    for t in tids:
        pure = [i for i in tracks[t] if i not in occ] or tracks[t]
        vecs = [emb_lookup.get((records[i][0], records[i][1])) for i in pure]
        vecs = [v for v in vecs if v is not None]
        c = np.mean(vecs, 0) if vecs else np.zeros(512, np.float32)
        cents.append(c / max(np.linalg.norm(c), 1e-9))
        framesets.append(frozenset(records[i][0] for i in tracks[t]))
    cents = np.stack(cents).astype(np.float32)
    rr = 1.0 - k_reciprocal_rerank(cents, cents, k1=20, k2=6, lam=0.3)
    order = sorted(range(len(tids)), key=lambda i: -len(framesets[i]))  # longest first
    clusters, assign = [], {}
    for i in order:
        best, bs = -1, -1.0
        for ci, cl in enumerate(clusters):
            if cl["frames"] & framesets[i]:  # frame-set overlap -> cannot merge
                continue
            s = max(rr[i, m] for m in cl["members"])
            if s > bs:
                bs, best = s, ci
        if best >= 0 and bs >= sim_th:
            clusters[best]["members"].append(i)
            clusters[best]["frames"] |= framesets[i]
            assign[i] = best
        else:
            clusters.append({"members": [i], "frames": set(framesets[i])})
            assign[i] = len(clusters) - 1
    tid_to_new = {tids[i]: assign[i] for i in range(len(tids))}
    return {i: tid_to_new[records[i][2]] for i in range(len(records))}


MERGE = os.environ.get("GSR_MERGE", "smart")     # "smart" jersey+motion fragment merge (default) | "" off
MERGE_GAP = int(os.environ.get("GSR_MERGE_GAP", "150"))    # max frame gap to bridge (6 s @25fps)
MERGE_SPEED = float(os.environ.get("GSR_MERGE_SPEED", "9.0"))  # m/s player speed cap
MERGE_SLACK = float(os.environ.get("GSR_MERGE_SLACK", "7.0"))  # metres slack for motion gate
MERGE_PREDICT = os.environ.get("GSR_MERGE_PREDICT", "0") == "1"  # extrapolate exit velocity (tested: hurts)


def _velocity(pos, window=12):
    """Exit velocity (m/s on pitch) from a fragment's last <=window frames, speed-capped."""
    fs = sorted(pos)
    if len(fs) < 2:
        return (0.0, 0.0)
    f1, f0 = fs[-1], fs[max(0, len(fs) - 1 - window)]
    dt = (f1 - f0) / 25.0
    if dt <= 0:
        return (0.0, 0.0)
    vx, vy = (pos[f1][0] - pos[f0][0]) / dt, (pos[f1][1] - pos[f0][1]) / dt
    sp = np.hypot(vx, vy)
    if sp > MERGE_SPEED:
        vx, vy = vx * MERGE_SPEED / sp, vy * MERGE_SPEED / sp
    return (float(vx), float(vy))


def _find(p, x):
    while p[x] != x:
        p[x] = p[p[x]]; x = p[x]
    return x


MERGE_APP_SIM = float(os.environ.get("GSR_MERGE_APP_SIM", "0"))  # >0 enables guarded appearance stage
MERGE_APP_COS = float(os.environ.get("GSR_MERGE_APP_COS", "0.90"))  # raw-cosine appearance merge threshold


def merge_fragments(frag, cent=None):
    """Merge BoT-SORT fragments into identities via high-precision signals: same
    (team, jersey) number, and motion continuity (a fragment resuming where another
    ended within a reachable time/pitch gap). An optional 3rd stage links far-apart
    same-team fragments by appearance (k-reciprocal) guarded against jersey conflict.
    A union is allowed only when the two components' frame sets are disjoint, so
    every identity keeps <=1 box per frame (the evaluator requires unique ids per
    frame). Returns {tid: root}."""
    tids = list(frag)
    parent = {t: t for t in tids}
    cframes = {t: set(frag[t]["frames"]) for t in tids}  # component (root) frame set

    def union(a, b):
        ra, rb = _find(parent, a), _find(parent, b)
        if ra == rb or (cframes[ra] & cframes[rb]):
            return
        parent[ra] = rb
        cframes[rb] |= cframes[ra]

    # (1) jersey anchor: same team + same number -> same player (numbers unique per team)
    groups = defaultdict(list)
    for t in tids:
        f = frag[t]
        if f["role"] == "player" and f["jersey"] is not None:
            groups[(f["team"], f["jersey"])].append(t)
    for g in groups.values():
        g.sort(key=lambda t: frag[t]["first"])
        for a, b in zip(g, g[1:]):
            union(a, b)

    # (2) motion continuity: chain a fragment to the best earlier-ending one of the
    #     same team whose exit is within a reachable time+distance of this entry.
    for b in sorted(tids, key=lambda t: frag[t]["first"]):
        fb = frag[b]
        best, bestd = None, 1e9
        for a in tids:
            fa = frag[a]
            if a == b or fa["role"] != fb["role"]:
                continue
            if fb["role"] in ("player", "goalkeeper") and fa["team"] != fb["team"]:
                continue
            gap = fb["first"] - fa["last"]
            if gap <= 0 or gap > MERGE_GAP:
                continue
            if _find(parent, a) == _find(parent, b) or (cframes[_find(parent, a)] & cframes[_find(parent, b)]):
                continue
            dt = gap / 25.0
            if MERGE_PREDICT:
                px_, py_ = fa["last_pos"][0] + fa["exit_vel"][0] * dt, fa["last_pos"][1] + fa["exit_vel"][1] * dt
                d = np.hypot(px_ - fb["first_pos"][0], py_ - fb["first_pos"][1])
                reach = MERGE_SLACK + 3.0 * dt  # extrapolation uncertainty grows with the gap
            else:
                d = np.hypot(fa["last_pos"][0] - fb["first_pos"][0], fa["last_pos"][1] - fb["first_pos"][1])
                reach = MERGE_SPEED * dt + MERGE_SLACK
            if d <= reach and d < bestd:
                bestd, best = d, a
        if best is not None:
            union(best, b)

    # (3) optional appearance link for far-apart same-team fragments. Uses RAW cosine
    #     between fragment centroids: OSNet already separates same-team players well
    #     (measured AUC 0.964), so a high threshold (~0.90) merges a player's own
    #     fragments while rarely joining two teammates. Guarded against jersey conflict.
    if (MERGE_APP_COS > 0 or MERGE_APP_SIM > 0) and cent is not None and len(tids) > 2:
        M = np.stack([cent[t] / max(np.linalg.norm(cent[t]), 1e-9) for t in tids]).astype(np.float32)
        if MERGE_APP_COS > 0:
            S = M @ M.T; th = MERGE_APP_COS
        else:
            S = 1.0 - k_reciprocal_rerank(M, M, k1=20, k2=6, lam=0.3); th = MERGE_APP_SIM
        cand = []
        for ia in range(len(tids)):
            for ib in range(ia + 1, len(tids)):
                if S[ia, ib] >= th:
                    cand.append((float(S[ia, ib]), tids[ia], tids[ib]))
        for s, a, b in sorted(cand, reverse=True):
            fa, fb = frag[a], frag[b]
            if fa["role"] != fb["role"]:
                continue
            if fb["role"] in ("player", "goalkeeper") and fa["team"] != fb["team"]:
                continue
            ja, jb = fa["jersey"], fb["jersey"]
            if ja is not None and jb is not None and ja != jb:
                continue
            union(a, b)

    return {t: _find(parent, t) for t in tids}


SPLIT_EPS = float(os.environ.get("GSR_SPLIT", "0"))  # DBSCAN cosine-eps to cut in-track ID switches (0=off)


def split_tracks(tk, emb_lookup, eps, min_samples=5):
    """GTA-Link splitter: DBSCAN a track's box embeddings; if it contains >1 dense
    appearance cluster, an ID switch happened mid-track, so split it into sub-tracks
    (each a fresh id). Same-identity boxes stay together (measured intra-id cosine
    ~0.93); a crossover to another player forms a second cluster. Returns new records."""
    by = defaultdict(list)
    for i, r in enumerate(tk):
        by[r[2]].append(i)
    out = list(tk)
    nextid = max((r[2] for r in tk), default=0) + 1
    for tid, idxs in by.items():
        if len(idxs) < 2 * min_samples:
            continue
        E = np.stack([emb_lookup.get((tk[i][0], tk[i][1]), np.zeros(512, np.float32)) for i in idxs])
        E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
        lab = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(E)
        clusters = sorted(set(lab) - {-1})
        if len(clusters) <= 1:
            continue
        cents = {c: E[lab == c].mean(0) for c in clusters}
        newid = {c: (tid if k == 0 else nextid + k - 1) for k, c in enumerate(clusters)}
        nextid += len(clusters) - 1
        for pos, i in enumerate(idxs):
            c = lab[pos]
            if c == -1:  # noise -> nearest cluster centroid by cosine
                c = max(clusters, key=lambda cc: float(E[pos] @ cents[cc]))
            r = tk[i]
            out[i] = (r[0], r[1], newid[c], r[3], r[4], r[5])
    return out


def process(split, seq, do_relink=False):
    base = C.OUT_ROOT / split / seq
    det = C.load(base / "det.pkl")
    jfile = "jersey.pkl" if not JERSEY_TAG else f"jersey_{JERSEY_TAG}.pkl"
    jer = C.load(base / jfile)["frames"]
    emb = C.load(base / "emb.pkl")["frames"]
    cfile = "calib.pkl" if not CALIB_TAG else f"calib_{CALIB_TAG}.pkl"
    calib = C.load(base / cfile)["frames"]
    tk = C.load(base / f"track_{TRACK_TAG}.pkl")["records"]
    labels = C.load_labels(split, seq)
    image_ids = det["image_ids"]
    H_eff = fill_homographies(calib)
    fail_frames = {fi for fi, h in enumerate(calib) if h is None}
    if all(h is None for h in H_eff) or not tk:
        # Total calibration/tracking failure: still emit an empty predictions file so
        # the sequence is scored (~0) rather than silently dropped from the average.
        dst = C.OUT_ROOT / "preds" / f"SoccerNetGS-{split}" / TAG / "data" / f"{seq}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"images": labels["images"], "categories": labels["categories"], "predictions": []},
                  open(dst, "w"))
        print(f"[s3b {seq}] no homography/tracks -> EMPTY preds", flush=True)
        return

    emb_lookup = {(fi, j): emb[fi][j] for fi in range(len(emb)) for j in range(len(emb[fi]))}
    occ_fij = {(tk[i][0], tk[i][1]) for i in _occluded_boxes(tk)} if CLEAN_CENT else set()

    # Identity attributes are aggregated per ORIGINAL BoT-SORT track (a wrong
    # relink merge must not corrupt jersey/team votes). Association id (track_id)
    # is optionally relinked purely to raise GS-AssA.
    records = split_tracks(tk, emb_lookup, SPLIT_EPS) if SPLIT_EPS > 0 else list(tk)
    if do_relink:
        remap = relink_records(records, emb_lookup)
        outid = [remap[i] for i in range(len(records))]
    else:
        outid = [r[2] for r in records]

    by_track = defaultdict(list)
    for r in records:
        by_track[r[2]].append(r)

    role_of, emb_of, jersey_of, pitchx_of, boxx_of = {}, {}, {}, {}, {}
    votes_of, pos_of = {}, {}
    for tid, rs in by_track.items():
        cls_maj = Counter(r[3] for r in rs).most_common(1)[0][0]
        role_of[tid] = C.CLS_TO_ROLE[cls_maj]
        vecs = [emb_lookup[(fi, j)] for fi, j, *_ in rs
                if (fi, j) in emb_lookup and (fi, j) not in occ_fij]
        if not vecs:  # all boxes occluded -> fall back to the full set
            vecs = [emb_lookup[(fi, j)] for fi, j, *_ in rs if (fi, j) in emb_lookup]
        emb_of[tid] = np.mean(vecs, 0) if vecs else np.zeros(512, np.float32)
        pos = {fi: project(H_eff[fi], (bbox[0] + bbox[2]) / 2.0, bbox[3]) for fi, j, _, _, bbox, _ in rs}
        pos_of[tid] = pos
        boxx_of[tid] = [(fi, xy[0]) for fi, xy in pos.items()]
        pitchx_of[tid] = float(np.median([xy[0] for xy in pos.values()])) if pos else 0.0
        votes_of[tid] = frag_votes(rs, jer) if role_of[tid] == "player" else None
        jersey_of[tid] = decode_votes(votes_of[tid]) if votes_of[tid] is not None else None

    team_of = {}
    field = [t for t in by_track if role_of[t] in ("player", "goalkeeper")]
    fit_set = [t for t in field if role_of[t] == "player"] if TEAM_PLAYERS_ONLY else field
    if len(fit_set) < 2:
        fit_set = field
    if len(fit_set) >= 2:
        def unit(t):
            return emb_of[t] / max(np.linalg.norm(emb_of[t]), 1e-9)
        X = np.stack([unit(t) for t in fit_set])
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        lab = km.labels_
        cl = {c: [t for t, l in zip(fit_set, lab) if l == c] for c in (0, 1)}
        left_cluster = _left_cluster(cl, pitchx_of, role_of, boxx_of)
        for t, l in zip(fit_set, lab):
            team_of[t] = "left" if l == left_cluster else "right"
        for t in field:  # keepers excluded from the fit: assign to the nearest team centroid
            if t not in team_of:
                c = int(np.argmin([np.linalg.norm(unit(t) - km.cluster_centers_[k]) for k in (0, 1)]))
                team_of[t] = "left" if c == left_cluster else "right"
    else:
        for t in field:
            team_of[t] = "left"
    if SIDE != "gk":  # 'gk' already anchors on goalkeepers; avoid double-flipping
        for t in field:
            if role_of[t] == "goalkeeper":
                team_of[t] = "right" if pitchx_of[t] > 0 else "left"

    # Merge fragments into identities (raises GS-AssA and pools jersey votes across
    # fragments of the same player, converting jersey misses on short fragments to hits).
    if MERGE == "smart":
        frag = {}
        for tid in by_track:
            frames = sorted(pos_of[tid])
            f0, f1 = frames[0], frames[-1]
            frag[tid] = dict(frames=frames, first=f0, last=f1, first_pos=pos_of[tid][f0],
                             last_pos=pos_of[tid][f1], exit_vel=_velocity(pos_of[tid]),
                             team=team_of.get(tid), role=role_of[tid], jersey=jersey_of[tid])
        ident = merge_fragments(frag, cent=emb_of)
        members = defaultdict(list)
        for tid, idv in ident.items():
            members[idv].append(tid)
        role_new, team_new, jersey_new = {}, {}, {}
        for idv, mem in members.items():
            rc, tc = Counter(), Counter()
            for t in mem:
                rc[role_of[t]] += len(by_track[t])
                if team_of.get(t):
                    tc[team_of[t]] += len(by_track[t])
            role_i = rc.most_common(1)[0][0]
            team_i = tc.most_common(1)[0][0] if tc else None
            pooled = pool_votes([votes_of[t] for t in mem if votes_of[t] is not None])
            jn_i = decode_votes(pooled) if role_i == "player" else None
            for t in mem:
                role_new[t], team_new[t], jersey_new[t] = role_i, team_i, jn_i
        role_of, team_of, jersey_of = role_new, team_new, jersey_new
        outid = [ident[r[2]] for r in tk]

    # Safety: the evaluator requires unique track ids per frame. Merges are frame-set
    # gated so this normally holds, but reassign any residual collision to a fresh id.
    fresh = max(outid, default=0) + 1
    seen_frame = defaultdict(set)
    for i, r in enumerate(records):
        if outid[i] in seen_frame[r[0]]:
            outid[i] = fresh; fresh += 1
        seen_frame[r[0]].add(outid[i])

    gtf = gt_by_frame(labels, image_ids) if ORACLE else {}
    preds = []
    for i, (fi, j, tid, cls, bbox, conf) in enumerate(records):
        role = C.CLS_TO_ROLE[cls] if PERBOX_ROLE else role_of[tid]
        jn = None if NO_JERSEY else (jersey_of[tid] if role == "player" else None)
        team = None if NO_TEAM else (team_of.get(tid) if role in ("player", "goalkeeper") else None)
        gtpit = None
        if ORACLE and fi in gtf and len(gtf[fi][0]):
            gb, gr, gt_team, gj, gp = gtf[fi]
            k = int(iou_to(gb, bbox).argmax())
            if iou_to(gb, bbox)[k] >= 0.5:
                if "role" in ORACLE and gr[k]:
                    role = gr[k]
                if "team" in ORACLE:
                    team = gt_team[k]
                if "jersey" in ORACLE:
                    jn = int(gj[k]) if (gj[k] not in (None, "", "null")) else None
                if "loc" in ORACLE and (not LOC_FAILONLY or fi in fail_frames):
                    gtpit = gp[k]
        if role not in ("player", "goalkeeper", "referee"):
            role = "player"
        cat = {"player": 1, "goalkeeper": 2, "referee": 3}[role]
        x1, y1, x2, y2 = bbox
        H = H_eff[fi]
        bl, bm, br = project(H, x1, y2), project(H, (x1 + x2) / 2.0, y2), project(H, x2, y2)
        bp = gtpit if gtpit else {"x_bottom_left": bl[0], "y_bottom_left": bl[1],
                                  "x_bottom_middle": bm[0], "y_bottom_middle": bm[1],
                                  "x_bottom_right": br[0], "y_bottom_right": br[1]}
        preds.append({
            "image_id": image_ids[fi], "track_id": int(outid[i]), "supercategory": "object",
            "category_id": cat, "confidence": conf,
            "bbox_pitch": bp,
            "attributes": {"role": role, "team": team,
                           "jersey": (str(jn) if jn is not None else None)},
        })

    # Gap interpolation (GSI): fill SHORT gaps within an identity's track by linearly
    # interpolating the image box (occlusion / missed detection), recovering false
    # negatives. Only short gaps — long merge-bridged gaps are off-screen (would be FPs).
    if INTERP_MAX > 0:
        byid = defaultdict(dict)
        for i, (fi, j, tid, cls, bbox, conf) in enumerate(records):
            byid[outid[i]][fi] = bbox
        attr = {}
        for p in preds:
            attr.setdefault(p["track_id"], (p["category_id"], p["attributes"]))
        for oid, fb in byid.items():
            if oid not in attr:
                continue
            cat, at = attr[oid]
            fs = sorted(fb)
            for a, b in zip(fs, fs[1:]):
                gap = b - a
                if gap <= 1 or gap > INTERP_MAX:
                    continue
                ba, bb = np.asarray(fb[a], float), np.asarray(fb[b], float)
                for f in range(a + 1, b):
                    x1, y1, x2, y2 = ba + (bb - ba) * ((f - a) / gap)
                    H = H_eff[f]
                    bl, bm, br = project(H, x1, y2), project(H, (x1 + x2) / 2.0, y2), project(H, x2, y2)
                    preds.append({
                        "image_id": image_ids[f], "track_id": int(oid), "supercategory": "object",
                        "category_id": cat, "confidence": 0.5,
                        "bbox_pitch": {"x_bottom_left": bl[0], "y_bottom_left": bl[1],
                                       "x_bottom_middle": bm[0], "y_bottom_middle": bm[1],
                                       "x_bottom_right": br[0], "y_bottom_right": br[1]},
                        "attributes": at})

    out = {"images": labels["images"], "categories": labels["categories"], "predictions": preds}
    dst = C.OUT_ROOT / "preds" / f"SoccerNetGS-{split}" / TAG / "data" / f"{seq}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(dst, "w"))
    nid = len(set(outid))
    print(f"[s3b {seq}] {len(by_track)} frags -> {nid} ids, {len(preds)} preds -> {TAG}", flush=True)


def main():
    split = sys.argv[1]
    do_relink = "--relink" in sys.argv
    arg = sys.argv[2]
    seqs = C.list_seqs(split) if arg == "all" else arg.split(",")
    for seq in seqs:
        process(split, seq, do_relink)


if __name__ == "__main__":
    main()
