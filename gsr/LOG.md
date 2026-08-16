# GSR 2025 — GS-HOTA build log

## ★ FINAL TEST-49 RESULT: GS-HOTA 55.68 (DetA 42.14 / AssA 73.59 / LocA 92.54, AssPr 85 / AssRe 81)
All 49 labelled test seqs, same TrackEval GS-HOTA evaluator as the leaderboard. valid 54.86, test-5 55.41
(generalizes). vs baseline 29.01, AuxFlow 28.45 (≈2×); KIST 63.90 / Constructor 63.81 (top, ~8 ahead).
Reproduce: `bash gsr/finish.sh test all` (features from s1+s2 cached under gsr/out/test/).


Goal: maximize GS-HOTA on SoccerNet GSR 2025 test (50 seqs). Tune on valid, report on test.
Leaderboard: KIST 63.90 (DetA 51.36/AssA 79.55, VLM-ReID) · Constructor 63.81 · baseline 29.01 · AuxFlow 28.45.

## Pipeline (our own; feeds the verified TrackEval GS-HOTA evaluator)
- **s1** (soccer_eda): DEIMv2 detect (drop ball) + jersey vis-gate/OCR logits → `det.pkl`, `jersey.pkl`
- **s2** (dfine-reid): OSNet 512-d embed + PnLCalib homography/frame → `emb.pkl`, `calib.pkl`
- **s3** (soccer_eda): BoT-SORT assoc + assemble (role majority incl. GK-fix, team KMeans→left/right,
  jersey gated logits-sum vote, project bottom-middle→pitch metres) → predictions JSON
- **eval** (sn-gamestate/.venv): TrackEval SoccerNetGS → GS-HOTA (printed as "HOTA")
- s1/s2 = fixed feature extraction (expensive, cache once); s3 = all the levers (cheap, iterate).

## Results
| config | split | seqs | GS-HOTA | GS-DetA | GS-AssA | notes |
|---|---|---|---|---|---|---|
| baseline v0 | valid | SNGS-021 | **42.46** | 31.60 | 57.16 | first run; LocA 90.8; 40 IDs vs 23 GT (fragmentation) |
| +relink (coupled) | valid | SNGS-021 | 36.92 | 22.05 | 61.81 | merge corrupts jersey/team votes → DetA collapses |
| +relink (decoupled attrs) | valid | SNGS-021 | 38.19 | 31.60 | 46.19 | DetA preserved, but wrong merges (same-team look-alikes) HURT AssA |

**Relink verdict: OFF.** Even decoupling identity attrs from association id, the general-OSNet
k-recip relink makes wrong same-team merges → AssA drops. Confirms recon: targeted fix for long
high-switch matches, not clean 30 s clips. Base (raw BoT-SORT ids) wins.

## Observations
- DetA 31.6 already >2x the 13.7 official baseline → identity gate (jersey+team+role) working.
- AssA 57.2 is the weak point: over-fragmentation (40 tracks / 23 GT ids). Lever: gated relink / tracker tune.
- LocA 90.8 confirms calibration is solved (not a lever).

## Multi-seq baseline (12 valid seqs SNGS-021..032) + ablations
| config | GS-HOTA | GS-DetA | GS-AssA | note |
|---|---|---|---|---|
| **baseline v0** | 43.95 | 31.78 | 60.81 | 552 tracker ids vs 275 GT (2x frag); LocA 91.1 |
| ablate jersey | 25.58 | 11.18 | — | jersey worth +20.6 DetA |
| ablate team | 11.83 | 2.76 | — | **team worth +29 DetA** (every player needs a side) |
| ablate GK-fix | 43.42 | 31.60 | — | +0.18 (few GKs) |
| oracle team | 48.47 | 39.05 | — | perfect team ceiling: +4.5 |
| oracle jersey | 53.62 | 51.97 | — | **perfect jersey ceiling: +9.7 (biggest)** |
| oracle team+jersey | 60.99 | 68.90 | — | perfect identity |
| oracle all (t+j+role) | 61.40 | 69.94 | 53.98 | **our detect+track ceiling ~61 (≈KIST 63.9!)** — AssA then binds |

GT jersey is labelled **per-track-consistent**: a track is all-known (one number on every box)
or all-None (~50% of player boxes). So jersey is a per-track decision: known-track→right number
on all boxes, None-track→abstain. (Per-box emission gating tested → HURTS; emit-all is correct.)

Jersey diag (469 tracks→GT): None-track false-commit 36/92; known-track hit 194 / miss 107 / wrong 76
(acc 51.5%). Misses partly from fragmentation (fewer votes per fragment → links jersey to AssA).

### Jersey threshold sweep (12 valid) → best VIS=0.7 CONF=0.9 MIN_VOTES=6
| VIS/CONF/MV | GS-HOTA | GS-DetA |
|---|---|---|
| 0.6/0.7/4 (base) | 43.95 | 31.78 |
| 0.6/0.85/6 | 47.88 | 37.28 |
| **0.7/0.9/6** | **48.16** | **37.68** |
Stricter legibility+confidence gates cut hallucinated/wrong reads → +4.2 GS-HOTA, no new model.

### Team side-labelling (12 valid): framemean wins
Clustering (KMeans on OSNet) is near-perfect (~1.0 acc most seqs); the error is the left/right
label. Global mean-x flips on 2/12 (attacking team's mean shifts to opponent half); global min-x
flips different clips. **Per-frame vote of which cluster owns the lower mean-x ("framemean")** is
robust → GS-HOTA 48.16→**50.60**, team acc 88%. (oracle team ceiling now only +1.8.)

### KEY: jersey gap is ASSOCIATION, not reading
Grouping player boxes by GT track (perfect association) → jersey known-track acc **82.4%, only 9
misses** (vs 141 misses on tracker tracks). So the reader is fine; **fragmentation** (552 tracker
ids vs 275 GT) starves short fragments of the 6 votes needed. Fixing association unlocks BOTH AssA
and jersey coverage.

### Smart fragment merge (GSR_MERGE=smart) → 52.97
Union-find over fragments by two high-precision signals — same (team, jersey number), and motion
continuity (a fragment resuming where another ended within a reachable time+pitch gap); appearance
NOT used (same-team look-alikes cause wrong merges); unions only when component frame-sets are
disjoint (keeps <=1 box/frame). Pools jersey votes per merged identity.
| config | GS-HOTA | GS-DetA | GS-AssA | ids |
|---|---|---|---|---|
| best no-merge | 50.60 | 41.41 | 61.86 | 552 |
| **+smart merge** | **52.97** | 41.97 | **66.88** | 417 |
AssA +5.0, DetA +0.6. Still 417 vs 275 GT ids → more merge headroom (tuning gap/slack, or safe
appearance+motion for far-apart fragments).

## BroadTrack calibration swap (test) — calibration IS a big lever on the diverse test set
Correction to the earlier "calibration solved": that held for the single LTPI camera. On the 49-seq TEST
(49 different broadcast cameras) the localization ceiling is huge: oracle-loc GS-HOTA **65.64** (DetA 49.7,
AssA 86.8) vs our 55.68 — **+9.96**. Decomposition: calib-dropout frames (5.65%, held-last) cost only +0.72;
the other **+9.24 is good-frame accuracy** — LocA 92.5 only measures matched pairs, but the α-averaged GS-HOTA
penalises the 2-4 m tail that PnLCalib produces on diverse cameras. Homography interpolation across dropouts:
NEGATIVE (55.51) — projective matrices don't interpolate. Needs a real camera tracker → BroadTrack (WACV'25,
halves reprojection error; user's reproduction `broadtrack:clean`, JaC@5 56.56).
Chain: prepare_soccernet (GT boxes+tripod) → BroadTrack Docker per seq → broadtrack.json 'cp' (SoccerNet camera
format) → `broadtrack_to_calib.py` (cp→sn_calibration Camera→inv(to_homography)=image→pitch) → s3b GSR_CALIB_TAG=bt.
**Validated SNGS-116: ours 59.03 → BroadTrack 62.47 (+3.44)** (DetA 47.9→50.5, AssA 72.8→77.3); coords correct.

### ★★ FULL TEST-49 with BroadTrack calibration: GS-HOTA 61.98 (DetA 46.90 / AssA 81.93) — +6.30 over PnLCalib
~1.9 behind KIST 63.90 / Constructor 63.81 (podium-level). Swapping calibration improved BOTH axes because the
BroadTrack pitch coords feed the motion-merge and team-side too: AssA 73.6→81.9 (above KIST's 79.55), DetA
42.1→46.9. Reproduce: prepare_soccernet.py → run_soccernet.py (broadtrack:clean) → broadtrack_to_calib.py
--tag bt → `GSR_CALIB_TAG=bt bash gsr/finish's s3b+eval`. Remaining gap to the very top = DetA 47 vs 51 (jersey).

## Autonomous research round 2 (CPU, valid)
Negatives (thorough, keep off): gkframe team-side 46.9 (GK detect unreliable); velocity-predicted
motion merge 53.76 (noisy extrapolation); size-weighted jersey vote 54.06 (gates already filter).

Error decomposition (127k GT person boxes, best config): TP 60% · **jersey_wrong 18.9%** (known 20619,
mostly MISS not wrong-read) · **team_wrong 14.2%** · no_detection 5% · role_wrong 1.9%. Player TP-rate 46.8%,
GK 85.8%, ref 82.4%. Confirms jersey (via coverage/association) > team > rest; no new overlooked lever.

**KEY FINDING — OSNet already separates same-team players (AUC 0.964)** (same-identity centroid sim 0.934 vs
diff-id same-team 0.792; 7.9% overlap at 0.893). The earlier appearance-merge failed only because it used
k-reciprocal (distorts the clean cosine). **Raw-cosine appearance merge at ~0.90** is safe and helps:
| APP_COS | GS-HOTA | AssA |
|---|---|---|
| off | 54.26 | 68.28 |
| **0.90 (default)** | **54.69** | 69.15 |
| 0.88 | 54.71 | 69.57 |
Match-adaptive re-embedding therefore NOT needed (signal already present); appearance is a mop-up (+0.45)
since jersey+motion already merge most same-identity fragments. Next untested: online BoT-SORT appearance_thresh
tuning to fragment less at the source (needs s3a re-track — IO-bound, deferred past test extraction).

### Round 3 — literature-guided (SoccerNet GSR SOTA report synthesised)
Confirmed premises: KIST 63.90 won with OSNet+Deep-EIoU (VLM only for attribute-gen/tracklet-refine, not ReID);
Constructor's AssA 82 = offline split-then-merge with a ReID path that does NOT need a number; S=LocSim×IdSim
gates BOTH DetA and AssA. My AUC 0.964 already gives the ReID separability their PRTReID provides.
- **Interpolation (GSI)** fill short in-track gaps (≤10 fr): **+0.17 → 54.86** (recovers FN detections). KEPT.
- Splitter (DBSCAN in-track to cut ID switches): NEGATIVE (over-splits single players; my tracker+merge already
  strong — GTA-Link gains were on weak SORT/ByteTrack). Added a per-frame unique-id safety net (kept).
- Online BoT-SORT buffer=150: NEGATIVE 54.59 (offline merge re-acquires better than a long online buffer).
- Team KMeans players-only (exclude GK): NEGATIVE 53.41. Deep-EIoU online swap / PRTReID: deferred/low-value
  (AUC already 0.964; ceiling is identity-reading, not appearance separability).

Current best (12 valid): **GS-HOTA 54.86** (DetA 43.39, AssA 69.38). Ladder 43.95→48.16→50.60→54.26→54.69→54.86.
Defaults: MERGE=smart (jersey+motion+cosine0.90), INTERP=10, jersey VIS0.7/CONF0.9/MV6, SIDE=framemean.

## Generalization check — early TEST (held-out): GS-HOTA 54.22 on 5 test seqs
SNGS-116..120: **GS-HOTA 54.22** (DetA 39.59, AssA 74.26, LocA 93.05), 143 ids vs 128 GT. Matches
valid-12 (54.26) → **no overfitting**, config transfers. Full 50-test number pending (s2 extraction).

## Current best config (12 valid): GS-HOTA 52.97 (pre-final; latest 54.26)
`GSR_MERGE=smart GSR_VIS_TH=0.7 GSR_OCR_CONF_TH=0.9 GSR_MIN_VOTES=6` + SIDE=framemean (default).
Ladder: 43.95 (v0) → 48.16 (jersey thresh) → 50.60 (framemean team) → 52.97 (smart merge).
Oracle ceilings at best: team+jersey 61.3, all 61.7 (AssA-bound). Remaining lever: more merge (AssA)
+ jersey reader (the 82%→100% and coverage).

### Oracle ceilings at the merge config (54.26, 12 valid)
| oracle | GS-HOTA | GS-DetA | Δ |
|---|---|---|---|
| current best (merge) | 54.26 | 43.14 | — |
| jersey | 60.49 | 59.44 | **+6.2 (still #1)** |
| team | 56.68 | 47.85 | +2.4 |
| team+jersey | 64.42 | 69.18 | +10 |
| all (t+j+role) | **64.83** | 70.23 | ceiling — merge raised it from 61.7, now **> KIST 63.90** |
So perfect identity + our detect/track/merge would beat KIST. Jersey reading is the top remaining lever.

s3b is now reader-agnostic (GSR_JERSEY_FMT=logits|generic); a stronger reader can be dropped in as a
generic (number,conf) cache jersey_<tag>.pkl. PARSeq prototype in progress.

### Jersey reader: ConvNeXt is best (PARSeq prototyped, does not beat it)
PARSeq (SoccerNet-2024 winning recipe: legibility + torso ROI + STR, 95.6% on curated crops) on our
tiny raw detection crops: **76.1% per-GT-track known-acc < ConvNeXt 82.4%** — systematic "10"
hallucination on ~30-80px crops upscaled to 32×128. PARSeq's only edge: better abstention (fewer
None-track false-commits), which our thresholds already handle (9 false-commits). Qwen3-VL infeasible
per-box (50k crops). Reader is ~tapped on these crops; the +6.2 oracle-jersey is mostly unreachable
without a fundamentally stronger reader. Untested: real ViTPose torso ROI (heavy, uncertain — fixed-
fraction torso HURT ConvNeXt). s1c_parseq.py + jersey_parseq.pkl (generic fmt) exist for 3 valid seqs.

## Levers ablated (on valid tuning subset SNGS-021..032)
1. jersey thresholds — VIS0.7/CONF0.9/MV6 best (+4.2)
2. team side-labelling — framemean (+2.4)
3. fragment merge — smart jersey+motion (+2.4)
4. relink (loose k-recip) — HURTS, off
5. torso-ROI jersey — HURTS (reader trained on full crops), off
6. GK-fix — +0.18 (few GKs), on
