# Jersey reader track journal

Charter: GSR-START-HERE/reports/2026-08-17-charter-jersey-reader.md. Data and results under
/mnt/d/jersey-lab/. One entry per step: command, wall time, numbers, verdict. Newest at the bottom.

## 2026-08-17 D1: pool sizing on cached GT train crops

Command: `MODEL_AUTO_DOWNLOAD=false $EDA gsr/jersey_track/pool_gt_crops.py --split train --out /mnt/d/jersey-lab/pool`
(env soccer_eda, ORT 1.20.1 CUDA; deployed gate + OCR, cv2 stretch preprocessing as in s1).
Input: ltpi-research/cache/soccernet_crops, 54,822 crops from 40 of 57 train clips (GT boxes,
players only, frame stride 8). Output: /mnt/d/jersey-lab/pool/{gt_crops_scores_train.csv,pool_train.md}.
Result (MEASURED, ~7 min wall, GPU light): 54,822 crops / 808 tracks; 46,928 on known-jersey tracks.
Reader on GT train crops: ungated 30.6%; VIS 0.7 -> 11,688 pass (24.9%) at 78.2%, 2,544 wrong on 39
numbers; VIS 0.7 + CONF 0.9 -> 9,380 (20.0%) at 91.2%; VIS 0.46 -> 13,631 at 73.9%, 3,554 wrong.
By height at 0.7/0.9: 81.3% (<80 px) / 87.8 / 90.5 / 92.1 / 95.1 / 98.9 (>=200). None-track crops
passing VIS 0.7: 122 of 7,894 (1.5%). Top confusions: 11->17 53, 17->10 50, 50->10 46, 14->10 43,
11->10 42, 44->14 35, 24->10 35. Hard pool per number: #24 234, #36 200, #14 168, #8 167, #4 132.
Verdict: D1 gate passes by projection (2,544 hard at stride 8 on 40 clips; all frames on 57 clips is
~11x more, correlated). Note the reader is better on GT train crops (91.2% gated) than on valid
detector crops (84.15%, analysis/jersey-2026-08-17); D3 measures the det-vs-GT box effect on the
same clips.

## 2026-08-17 D2: s1 (DEIMv2 + gate + OCR) over GSR-train

Smoke: `bash gsr/jersey_track/run_s1_train.sh SNGS-060` -> 750 frames, 12,831 person-boxes, 93 s wall
(incl. model load), det.pkl 353 KB + jersey.pkl 1.15 MB (log /mnt/d/jersey-lab/logs/s1_smoke.log).
Full run launched 22:14:50, detached, resumable (skip-if-exists per clip): remaining 56 clips, expected
~80 min. Log /mnt/d/jersey-lab/logs/s1_train_run.log; outputs gsr/out/train/SNGS-*/{det,jersey}.pkl.
Relaunch line: `cd /home/dxdxxd/projects/soccer-app && bash gsr/jersey_track/run_s1_train.sh`.
Mining smoke (tables only) on valid SNGS-021: 10,728 dets, 97.1% matched to GT at IoU 0.5; det vs GT
box IoU median 0.856, h_det/h_gt median 1.013 (p10 0.975, p90 1.052) - detector geometry is close to GT.

## 2026-08-17 tooling (while D2 runs)

- vLLM env for the VLM annotator: `uv venv --python 3.12 /home/dxdxxd/.venvs/vllm-qwen3vl` +
  `uv pip install vllm pandas pillow` (cache /mnt/d/jersey-lab/uv-cache) -> vllm 0.27.1, EXIT 0
  (log /mnt/d/jersey-lab/logs/vllm_env.log). Not yet smoke-tested on GPU (GPU busy with s1).
- Scripts in gsr/jersey_track/: pool_gt_crops.py (generalised: --ocr/--gate/--seqs dev/--tag, saves raw
  logits, per-track vote at MV 1/2/4 on stride-8 GT crops), mine_train_crops.py (D3), make_sheets.py
  (D4 contact sheets), vlm_label_crops.py (D4 VLM annotator), jnr_valid12.py + jnr_valid12_eval.py
  (D5), build_dev_val.py (D6 val set from valid games 3+5, gate-filtered GT crops), chain_after_s1.sh
  (starts D3 mining and D5 JNR extraction when s1 finishes; armed 22:21).
- Footgun: manifest_full.csv of the ltpi crop cache has ABSOLUTE paths for train rows and paths
  RELATIVE to ltpi-research for valid rows; pool_gt_crops.py resolves both.
- Footgun: run_s1_train.sh pipes through `grep -v Warning | tee`, so the log lags by a block; count
  finished clips with `ls gsr/out/train | wc -l`.
- Dev-split baseline (deployed reader on GT crops of valid games 3+5, 55,970 crops / 40 clips)
  running: /mnt/d/jersey-lab/pool/pool_valid_dev_deployed.md.
Dev-split baseline DONE (MEASURED): 55,970 GT crops / 792 player tracks (691 known, 101 None) on
valid games 3+5. Deployed reader: VIS 0.7 -> 9,016 pass (18.3%) at 69.8%; VIS 0.7 + CONF 0.9 -> 6,467
(13.1%) at 86.4%; by height 67.5% (<80) / 84.2 / 87.5 / 87.9 / 92.3 / 88.5. Per-track vote on stride-8
GT crops (VIS 0.7 / CONF 0.95): MV1 516 hit / 60 wrong / 115 miss of 691 (74.7%), 12 false commits of
101 None tracks; MV2 466/40/185, 2 FC. Top confusions kit-specific: 19->10 68, 11->10 61, 36->16 60,
15->10 51, 43->3 45, 13->18 46. This is the D6 selection benchmark; test_refined is not.
D2 DONE 2026-08-18 00:05: 57/57 clips, 6,610 s for the 56-clip run (~118 s/clip incl. contention with
the dev scoring), outputs gsr/out/train/SNGS-*/{det,jersey}.pkl (~85 MB). Chain fired 00:05:26: D3
mining (mine_train_crops.py, all 57 clips, hard 40 / easy 15 / abstain 16 per track, VIS 0.7, IoU 0.5) and
D5 JNR extraction on valid-12 started in parallel.

## 2026-08-18 D3: mined dataset v1 (DONE 00:25, ~20 min CPU/IO)

`mine_train_crops.py --split train --seqs all --out /mnt/d/jersey-lab/gsr_train_crops_v1` (VIS 0.7, IoU 0.5,
hard 40 / easy 15 / abstain 16 per track). MEASURED: 631,548 player/gk detections, 96.3% matched to GT;
496,314 on known-jersey tracks; gate-pass 122,419 (24.7%) at 80.1% -> 24,322 hard candidates. Selected
33,329 crops (187 MB): hard 17,577 (41 numbers, 878 tracks; height deciles 28/80/94/111/130/151/297 px),
easy 12,616, abstain 3,136 (828 gate-pass + 2,308 gate-fail on GT-None tracks). Detector vs GT box:
IoU median 0.841 (p10 0.682), h_det/h_gt median 1.001 (p10 0.958, p90 1.042), dy_top/dy_bot ~0 px:
no framing shift between GT-crop training and detector-crop deployment. Top hard confusions: 8->10 258,
14->10 248, 14->4 243, 24->14 230, 8->0 227, 9->8 211, 50->10 208. Files: images/, labels/ (json with
jersey_number, readability medium|low, stream, vis_p, pred, conf, h), manifest.csv, all_matches.csv, stats.md.

## 2026-08-18 D5: uncertainty-JNR vs ConvNeXt on valid-12 detector crops, perfect association (DONE 00:38)

`jnr_valid12.py` (12 clips, 113,667 matched player/gk crops, ViT-B/8 fp16, ~60 crops/s -> 30 min) +
`jnr_valid12_eval.py`; artifacts /mnt/d/jersey-lab/jnr_valid12/{SNGS-*.csv,SNGS-*.npz,summary.md}.
MEASURED, 187 known / 50 None player GT tracks:
- ConvNeXt shipped (VIS 0.7 / CONF 0.95 / MV 6): 156 hit / 19 wrong / 12 miss, FC 8.
- JNR alone (Dirichlet u <= 0.05, MV 6): 154 / 3 / 30, FC 2; (u <= 0.05, MV 2): 159 / 6 / 22, FC 7.
  JNR trades wrong reads for abstentions; paired vs shipped at MV 6: fixes 8, breaks 10, net -2.
- Hybrids: JNR first (u <= 0.05, MV 6) + ConvNeXt fills None: 164 / 12 / 11, FC 8 (+8 hits, in budget).
  Per-box mix as the s3b generic vote would do it (JNR read where u <= 0.05 with weight 2, else the
  shipped ConvNeXt read, MV 6): 164 / 14 / 9, FC 9; u <= 0.02: 162 / 15 / 10, FC 8.
- Per crop on known-track crops passing VIS 0.7 (n = 26,961): ConvNeXt 64.0%, JNR 71.5% (+7.5 pp,
  monotone across height: 18.5/45.8/55.9/66.4/70.0/76.7 vs 27.2/54.5/63.5/73.9/78.2/80.4);
  JNR at u <= 0.05 (n = 16,650): 96.7%.
Verdict vs the D5 gate (paired net >= +10, FC <= 8): NOT met by JNR alone (net -2); the hybrid reaches
+8 in budget. JNR is the stronger per-crop reader with a usable abstention; GS-HOTA leg follows
(generic caches jersey_jnrmix_*.pkl through the 67.26 chain).
D5 GS-HOTA leg (DONE 00:55, CPU): generic caches `gsr/out/valid/SNGS-*/jersey_<tag>.pkl` from
`build_jnr_cache.py` (JNR read where Dirichlet u <= U with weight W, else the shipped ConvNeXt read
where VIS >= 0.7 and CONF >= 0.95; boxes without a JNR read fall back to ConvNeXt), run through the
k1 chain via `clipcalib/tools/tune_gsr.py --s3b tools/s3b_splitfix.py ... GSR_JERSEY_FMT=generic
GSR_JERSEY_TAG=<tag> GSR_OCR_CONF_TH=0.5` (journal rows jt_*; per-seq jsons artifacts/tuning/perseq/tun_jt_*).
Baseline tun_k1_on 67.26 (DetA 57.85 / AssA 78.22). MEASURED:
| tag | U | W | GS-HOTA | DetA | clips up/down | worst clip |
| jnrmix_u05w2 | 0.05 | 2 | 68.37 (+1.11) | 59.65 | 8/3 | SNGS-032 -3.68 |
| jnrmix_u05w15 | 0.05 | 1.5 | 68.37 | 59.65 | 8/3 | -3.68 |
| jnrmix_u05w1 | 0.05 | 1 | 67.91 | 58.80 | 8/3 | -3.68 |
| jnrmix_u03w2 | 0.03 | 2 | 68.06 | 59.10 | 7/3 | -3.68 |
| jnrmix_u02w2 | 0.02 | 2 | **68.18 (+0.92)** | 59.33 | 6/1 | SNGS-028 -0.53 |
| jnrmix_u02w1 | 0.02 | 1 | 68.04 | 59.13 | 6/1 | -0.53 |
| jnr_u05 (JNR only) | 0.05 | - | 66.40 (-0.86) | 56.47 | | |
Pick: jnrmix_u02w2 (+0.92, one clip -0.53); the U 0.03-0.05 points buy +0.2 more at the price of a
-3.7 clip (SNGS-032) and fail do-no-harm. Caveat: valid-12 is one match; a second signal (dev split
per-track) is wanted before any test exposure. Test untouched.

## 2026-08-18 D4: base audit of mined v1 (in progress) and D6 control run

D4a JNR agreement (`/mnt/d/jersey-lab/audit_v1/jnr_agreement.md`, jnr_dir.py 33,329 crops in 511 s = 65/s):
easy stream 97.0% JNR == label (98.9% at u <= 0.05, 88.6% kept); hard stream 54.9% overall, at u <= 0.05
36.2% kept with 87.5% agreement -> JNR-confirmed hard subset 5,563 crops / 36 numbers / 648 tracks;
abstain_pass (None tracks, gate pass) JNR commits at u <= 0.05 on 11.5%.
D4b visual audit (Opus vision agents on magnified original crops, protocol A/B/C/D, verdicts in
audit_v1/verdicts/): raw hard sample sheets 0-3 (120 cells) A 4 / B 18 / C 83 / D 15 -> A+B 18%, D 12.5%;
sheets 6-9 (120) similar (mostly C, D from two-player boxes); JNR-confirmed hard sample (150):
A 24 / B 43 / C 68 / D 15 -> A+B 45%, D 10%. My own read of the same crops is looser on B vs C (about
half of the C cells are partially legible to me) but confirms the D calls: e.g. SNGS-071 track labelled 11
shows "17" on two crops, SNGS-100 labelled 62 shows "12" - and JNR "confirms" the wrong label there, so
JNR is not an independent annotator on GSR-train (likely overlap of SoccerNet-Jersey training games with
GSR games; memorised kit/number priors). Charter gate (A+B >= 90%, D <= 2%) FAILS for both raw and
JNR-confirmed hard. Next: VLM (Qwen3-VL-8B-FP8) as the second, independent reader on the 448 audited
crops; training set v1 = easy + hard confirmed by JNR AND VLM (D-free by two independent readers), gate
re-checked on the audited sample. Auditor 2 (sheets 3-6) still running.
D6 control run A (`train_run.sh ctrl_old_devval gsr_devval_old`, old 4,741-image mix, 23-40-49 recipe,
selection on the dev val set): best dev-val number accuracy 72.5% (early stop epoch 56, 25 min),
`/mnt/d/jersey-lab/runs/ctrl_old_devval/checkpoints/best_model.pth`. Deployed weights on the same 9,016
dev crops: 69.8% -> the recipe alone is +2.7 pp in domain (ESTIMATE of the gap, same crops, argmax).
vLLM on WSL2: needs VLLM_WSL2_ENABLE_PIN_MEMORY=1 (V2 model runner requires UVA/pinned memory).
D4 VLM leg (DONE 01:38): Qwen3-VL-8B-Instruct-FP8 via vLLM 0.27.1 (`vlm_label_crops.py`, min side 336,
the corpus prompt), 448 audited crops in ~2.5 min at batch 8 (~3 img/s), 5,563 JNR-confirmed hard crops
in ~10 min at batch 64 (~9 img/s). Against the audit classes: raw hard - VLM == label on 7% (A 69%, B 10%,
C 4%, D 0%); JNR-confirmed hard - VLM == label on 16% (A 58%, B 7%, C 7%, D 13%). VLM == label keeps
0 of 36 D in the raw sample and 2 of 15 in the JNR-confirmed sample; kept crops are 62-71% A+B.
Training set v1f (`filter_labels.py --require-both`, labels_f1): easy 12,616 + hard 1,064 (JNR u <= 0.05
AND VLM == label; 35 numbers) = 13,680 in-domain crops; total train 18,421 with the old mix.
D6 run B launched 01:47: `train_run.sh blend_v1f gsr_blend_v1f` (same recipe, dev-val selection),
/mnt/d/jersey-lab/runs/blend_v1f, log /mnt/d/jersey-lab/logs/train_blend_v1f.log, ETA ~2 h.

## 2026-08-18 D6 results (eval_reader.sh: dev per-crop, valid-12 per-track, valid-12 GS-HOTA with CONF sweep)

Recipe note: the 23-40-49 recipe (CE + label smoothing 0.1) caps the max softmax near 0.93, so the shipped
CONF 0.95 gate rejects every crop of a retrained reader; CONF is a model-specific knob and is swept on
valid-12 (0.95 / 0.9 / 0.85 / 0.8). Dev per-crop at matched coverage: control conf >= 0.85 keeps 72.2% at
88.3% vs deployed conf >= 0.9 keeps 71.7% at 86.4% (+1.9 pp).
Run A control (old 4,741 mix, dev-val selection, best 72.5% dev-val): valid-12 perfect association at
CONF 0.9: 159 hit / 17 wrong / 11 miss, FC 7 (deployed 156/19/12, FC 8); GS-HOTA CONF 0.9 -> 67.33
(+0.06), 4 up / 3 down, worst SNGS-027 -5.40; CONF 0.85 -> 66.77; 0.8 -> 65.73. Verdict: recipe alone is
a wash on GS-HOTA (journal rows jt_ctrl_old_devval_c*).
Run B blend v1f (old + 12,616 easy + 1,064 hard confirmed by JNR AND VLM = 18,421 train, early stop at
epoch 29, best epoch 14): best dev-val 74.5% (+2.0 over control, +4.7 over deployed on the same crops).
Eval running.
Run B blend v1f eval (DONE 10:25; the first eval attempt died at 03:05 on a $CC variable clash with the
conda activate hook - fixed, GS-HOTA leg rerun): dev per-crop argmax 74.5% (best of the three) but the
vote-level metrics are WORSE everywhere. Dev per-track (691 known / 101 None tracks, GT crops stride 8,
VIS 0.7, MV 1) at matched CONF: deployed 0.9 -> 518 hit / 68 wrong / 105 miss / FC 14; control 0.9 ->
514 / 41 / 136 / 9; blend 0.9 -> 465 / 57 / 169 / 15, blend 0.85 -> 489 / 79 / 123 / 20. Valid-12 perfect
association: blend CONF 0.9 -> 133 / 28 / 26, FC 11; 0.85 -> 146 / 32 / 9, FC 16 (deployed 156 / 19 / 12,
FC 8; control 159 / 17 / 11, FC 7). GS-HOTA valid-12 through the k1 chain: blend CONF 0.9 60.00 (-7.27),
0.85 61.46 (-5.80), 0.8 60.59, all clips down (worst -21.3). Journal rows jt_blend_v1f_c*.
Verdict: NEGATIVE. The in-domain blend (12,616 easy + 1,064 confirmed hard from 3 train matches = 73% of
the train set) makes the reader more wrong and more false-committing on other matches; per-crop argmax
accuracy on gate-pass crops is not a proxy for the vote. Combined with the audit, mining v1 as built is
closed. What survives the night: the JNR hybrid (+0.92 GS-HOTA valid, generic cache jnrmix_u02w2) and the
measured facts about the gate and the labels.

## 2026-08-18 morning probes for the orientation idea (CPU, cached data)

- VLM position vs audit class (raw hard 300): A+B cells are 95% "back"; C cells 37% front/profile, 63% back
  but illegible (occlusion, blur, contrast). JNR-confirmed sample: C cells only 12% front/profile.
- JNR Dirichlet uncertainty as the visibility gate for the ConvNeXt vote (no ShuffleNet, CONF 0.95):
  u <= 0.05 / 0.1 / 0.2 -> 65.38 / 65.50 / 65.54 GS-HOTA (-1.7..-1.9 vs 67.26), 3 clips up (021, 023, 031)
  and 6-7 down (worst -8.9). Journal rows jt_cnxjgate_u*. JNR uncertainty is a good gate for JNR's own read,
  not for ConvNeXt's.

## 2026-08-18 VLM track-level second opinion on valid-12 (zero-shot Qwen3-VL-8B-FP8) - NEGATIVE

`s3b_dump.py` (splitfix copy + GSR_DUMP_MEMBERS, predictions bit-identical to tun_k1_on) ->
`vlm_track_select.py` (top-8 crops per predicted player identity by gate prob x height, spread in time:
298 identities, 2,209 crops) -> `vlm_label_crops.py` (~5 min) -> `vlm_track_apply.py` (rules on the k1
predictions, official evaluator). Crop level: the VLM gives a number on 71.7% of crops; on committed tracks it
agrees with the committed number on 66.4% of crops. GS-HOTA (baseline 67.26): fill m=4/5/6 -> 66.22 / 66.49 /
66.69 (8/4/2 fills, all harmful); override m=5/6 -> 66.96 (3-4 changes, harmful), m=8 -> no change; abstain
m=4 -> 66.48; combos worse. Files /mnt/d/jersey-lab/vlm_track/{manifest.csv,vlm_reads.csv,changes_*.json},
prediction tags gsr/out/preds/SoccerNetGS-valid/vlm_*, per-seq jsons in clipcalib perseq/.
Verdict: a zero-shot VLM is a weaker reader than the shipped vote and cannot judge it; the KIST-style
fine-tuned VLM remains the untested route. Note to self: killed the mis-launched vLLM job with a substring
pkill - a hub-rule violation (kill by PID); no collateral, do not repeat.

## 2026-08-18 afternoon: owner's visual check on soc/supertestv2 and the gate-preprocessing control

`viz_jersey_pipe.py` over /home/dxdxxd/projects/soc/supertestv2/images (40 frames, 646 player/gk boxes,
median height 75 px): the deployed gate (128x128 stretch) passes 96 boxes at 0.7; the same gate under its
calibration transform (Resize 144 + CenterCrop 128) passes 158; the owner sees clearly legible numbers
labelled "no" (e.g. esptur07: red 10 v0.19|0.89 OCR 10 c1.00; 22 v0.09|0.93 c1.00; 15 v0.60|0.95 c1.00).
Among gate-rejected boxes 130 (24%) carry an OCR read with c >= 0.95, 47 of them "10" (prior collapse).
Control in the pipeline (`regate_cc.py`: valid-12 vis logits recomputed under the calibration transform, OCR
logits unchanged, jersey_gatecc.pkl; pass@0.7 roughly doubles per clip): GS-HOTA VIS 0.7 -> 62.81 (-4.45),
0.85 -> 63.83, 0.9 -> 63.93, 0.95 -> 64.80 (-2.47); SNGS-021 -14..-18 in every setting. Journal rows
jt_gatecc_v*. Verdict: the owner's observation is right (the deployed gate rejects legible numbers), but
loosening it is negative because the extra admitted crops carry confident-wrong OCR reads that the CONF gate
cannot separate; the strict mis-preprocessed gate is a crutch for a reader whose confidence is not
trustworthy. The gate and the reader can only be fixed together (reader with real abstention).
Three-reader per-box consensus (>= 2 of {JNR u <= U, deployed ConvNeXt at VIS 0.7 / CONF 0.95, retrained
control at VIS 0.7 / CONF 0.9} agree), valid-12 through the k1 chain: U 0.05 -> 67.02 (-0.25), U 0.2 ->
66.94 (-0.33), 2 clips up / 5 down (SNGS-027 -3.9..-4.7). Journal rows jt_cons3_u*. Negative: majority
voting of the existing readers does not beat the JNR-first hybrid (68.18).

## 2026-08-18 evening: human legibility labels (CVAT task 69) and the star-head reader v1

Owner labelled 500 crops (400 valid-12 detections stratified by gate prob + 100 supertestv2; tag jersey with
visible yes/partial/no and number, masks like *3): yes 109 / partial 18 / no 373. Analysis
/mnt/d/jersey-lab/cvat_legibility_v1/human_vs_pipeline.md. MEASURED:
- deployed gate (stretch) at 0.7: precision 0.63, recall 0.66 (yes+partial), 43 legible crops rejected (their gate
  probs 0.07-0.6; ConvNeXt conf >= 0.95 on 60% of them); calibration transform at 0.7: recall 0.84, precision 0.46.
- ConvNeXt on legible: argmax 0.81 (yes only 0.87), conf >= 0.95: 0.98 (n 86 of 127); on illegible: conf >= 0.95
  on 9% (35 of 373), 4 of them gate-pass. JNR on legible (valid-12 rows): argmax 0.76, u <= 0.05: 0.89 (55% kept).
- GT track number disagrees with the human read on 17% of legible valid-12 crops (17 of 100; e.g. SNGS-032/t16
  GT 42 read 12 twice, SNGS-024/t19 GT 28 read 23 twice, SNGS-021/t4 GT 28 read 18/25).
- legibility by height: 16% (<80 px) .. 40% (150-200 px).
Star head v1 (owner's idea: drop the gate, let the heads predict "*"): 11 classes per head, train = old 4,741 +
7,894 GT-None-track crops of GSR-train as (*,*), val = dev val + 3,000 None-track dev crops; recipe 23-40-49
(CE + smoothing), best combined val 78.6% (early stop 30, 25 min). `star_eval.py` (star_eval.md):
- 500 human crops: abstains on 83% of 'no' (reads a number on 17%, none at conf >= 0.95); on legible abstains 5%,
  correct 81% of all legible (85% among reads) - recall of legible 95% vs 66% for the gate, crop precision lower.
- valid-12 GS-HOTA (generic caches, k1 chain): no gate CONF 0.5 -> 67.26 (0.00, 5 up / 6 down), CONF 0.8 -> 67.57
  (+0.30, 7 up / 2 down, SNGS-027 -7.2), CONF 0.9 -> 66.35; with the deployed gate AND-ed: 66.91 / 67.24 / 66.07.
  Journal rows jt_star_v1_*. Verdict: a v1 abstaining reader without any gate matches the tuned gate+reader system;
  its confidence is flat (label smoothing) so the CONF gate cannot sharpen it. Next: same data with FocalLoss
  (sharp confidence, the deployed recipe) - star_v1_focal running.
Star v1 focal (same data, FocalLoss gamma 2 = the deployed recipe): 500 human crops - abstains on 87% of 'no',
legible correct 77% (98% at conf >= 0.95, n 56 = 44% coverage); GS-HOTA no gate: CONF 0.5 -> 66.81 (-0.46),
0.9 -> 66.81, 0.95 -> 65.66, 0.99 -> 60.86; with gate: 67.27 / 66.22 / 65.06 / 60.18. Sharp confidence does not
help the vote here (journal rows jt_star_v1_focal_*).
JNR-first hybrid with the star v1 reader (no gate) as the fill instead of ConvNeXt: u <= 0.02 + star conf >= 0.5 ->
67.47 (+0.20); u <= 0.02 + star conf >= 0.8 -> 67.97 (+0.70, 7 up / 2 down, worst -3.84); u <= 0.05 -> 67.19.
Below the ConvNeXt-based hybrid (68.18). Rows jt_jnrstar_*.

## 2026-08-18 night: zero-shot VLM bake-off on the 500 human-labelled crops (D4b)

Same prompt for all (the corpus prompt), min side 336, temperature 0.1; Qwen3.5 with enable_thinking=False;
Gemma 4 needs the -it checkpoints (base ones have no chat template) and transformers 5.12 (5.15 breaks vLLM
0.27.1 on per-layer head_dim); Gemma 4 fp8-on-the-fly is broken in vLLM (issues 39049/39407/48651-type dispatch),
E4B INT8-AWQ fails on the humming JIT (NVRTC), E4B AWQ-4bit runs. Table (illegible n 373 / legible n 127):
| reader | number on illegible | correct of all legible | correct among reads | 'yes' correct |
| Qwen3-VL-8B-Instruct-FP8 | 0.17 | 0.63 | 0.75 (n 106) | 0.70 |
| Qwen3.5-4B | 0.18 | 0.69 | 0.74 (n 119) | 0.72 |
| Qwen3.5-9B (QuantTrio AWQ) | 0.23 | 0.71 | 0.76 (n 119) | 0.75 |
| Gemma-4-E2B-it | 0.05 | 0.64 | 0.82 (n 99) | 0.68 |
| Gemma-4-E4B-it (AWQ-4bit) | 0.07 | 0.64 | 0.79 (n 102) | 0.68 |
| ConvNeXt deployed (argmax) | always reads | 0.81 | 0.81 | 0.87 |
| JNR (argmax) | always reads | 0.76 | 0.76 | 0.78 |
| star_v1 (11-class, no gate) | 0.17 | 0.81 | 0.85 (n 121) | 0.83 |
Verdict: no zero-shot VLM beats the specialised readers on legible crops (best 71% vs 81%); Gemma E2B abstains
best (5%) but reads 64%. Files: cvat_legibility_v1/vlm_reads_*.csv, models under /mnt/d/jersey-lab/models.

## 2026-08-19 night: FIFA/WorldPose data - mining, CVAT tasks, orientation check

- Mining (`mine_broadcast.py`, 1 fps, DEIMv2 + gate + ConvNeXt + star_v1; JNR via jnr_dir.py): 109 clips
  (20 challenge + 89 WorldPose compressed) -> 61,066 crops, median height 89 px, /mnt/d/jersey-lab/mine_fifa.
  Gate pass 18.9%, ConvNeXt vote (gate & conf >= 0.95) 11.9%, star_v1 reads a number on 40% (top "10" 4,776,
  and 1,083 reads of "0" = both heads 0, junk).
- CVAT (`select_for_cvat.py`, 20 per clip: 40% reader disagreement / 40% confident / 20% random, x3 upscale,
  prefill visible from star, number from star or ConvNeXt): 2,180 crops -> tasks **70-74** (fifa_legibility_v1_01..05,
  500/500/500/500/180), bundles /mnt/d/jersey-lab/cvat_fifa_v1/task_XX/{images,manifest.csv,annotations.xml}.
- Orientation from the challenge BODY_25 skeletons (`fifa_orient.py`, 13,431 mined crops of the 20 clips matched
  to reference boxes at IoU >= 0.5; front = person's right shoulder on the image left, |dx| >= 0.15 of box width):
  | orientation | crops | gate pass | ConvNeXt vote | star_v1 reads | JNR u <= 0.05 |
  | back | 5,008 | 0.38 | 0.26 | 0.79 | 0.39 |
  | side | 2,999 | 0.05 | 0.01 | 0.13 | 0.00 |
  | front | 5,424 | 0.02 | 0.00 | 0.17 | 0.00 |
  The deployed gate and JNR are orientation-disciplined; star_v1 commits a number on 17% of front-facing players
  (its abstain training set - GSR None tracks - had few front views). Front crops (dx > 0.25: see
  gsr_abstain/fifa_front_labels) become free abstain examples for star v2. Caveat: some WC kits carry a small
  chest number, so "front = no number" is approximate at 90 px.
- Detection external check (`fifa_detect.py` all frames + `fifa_eval_det.py`, /mnt/d/fifa/eval/det_eval.md):
  488,234 reference person boxes over 30,873 frames vs 513,228 DEIMv2 person detections. **Recall@0.5 0.980**
  (per clip 0.957-0.995), recall@0.7 0.839, median IoU of matches 0.79; 93.3% of our detections match a
  reference box (bench/staff/duplicates are the rest, ~1.2 extra per frame). By height: < 40 px 0.48,
  40-60 px 0.87, >= 60 px 0.96-0.99. Reference boxes are the challenge's own (SAM-3D-body pipeline), so
  IoU@0.7 also measures box-convention differences.
- Star v2 (= v1 data with the None-track abstain capped at 4,000 + 4,199 FIFA front-facing crops as abstain;
  best combined val 79.0%): 500 human crops - abstains on 90% of 'no' (v1 83%), legible correct 80% (v1 81%),
  abstains 6% on legible; FIFA orientation - reads a number on 0% of front (v1 17%), 2% of side (v1 13%),
  57% of back (v1 79%). valid-12 GS-HOTA no gate: CONF 0.5 -> 67.66 (+0.40), CONF 0.8 -> 67.73 (+0.46), 6 up /
  5 down, worst SNGS-024 -3.3. JNR-first (u <= 0.02, w2) + star v2 fill: conf 0.8 -> **68.40 (+1.13)** 7 up / 4
  down, worst -3.30 (SNGS-024); conf 0.5 -> **68.68 (+1.41)** 7/4, worst -4.92 (SNGS-024). Above the
  JNR+ConvNeXt hybrid (68.18) on aggregate, worse on the worst clip (-0.53 there). Rows jt_star_v2_*, jt_jnrstar2_*.
- Tracking external check, raw BoT-SORT (`fifa_track_chain.sh`: s1 2 h, s2 2.5 h, s3a 45 min on the SoccerNet
  layout /mnt/d/fifa/sn_layout; `fifa_eval_track.py --tracker botsort_base`, TrackEval MotChallenge2DBox, all persons,
  IoU 0.5, DO_PREPROC off): **COMBINED HOTA 61.19 (DetA 71.37 / AssA 52.56), IDF1 72.53, MOTA 92.09, 613 IDSW;
  1,219 our ids vs 446 reference ids** (2.7x fragmentation, the SoccerNet v0 pattern). Per clip HOTA 52.7
  (ENG_FRA_224512, 4,275 frames) .. 72.3 (CRO_MOR_182145). /mnt/d/fifa/eval/track_eval_botsort_base.md.
  Caveats: tracker frame_rate 25 vs 30/50 fps clips; reference ids are the challenge's own tracked subjects.

## 2026-08-19 SNGS-024 under the star variants: one false commit on a GT-None front-facing player

`track_diff.py` / members dumps (`s3b_dump.py`, /mnt/d/jersey-lab/star024/): between tun_k1_on and
tun_jt_jnrstar2_u02_c08 the boxes of SNGS-024 change jersey in three identities only, association in one:
- id 38, 311 boxes, GT identity 15 (jersey None, faces the camera most of the clip, red chest sponsor patch):
  star v2 reads "10" on 6 boxes (conf ~0.89), MIN_VOTES 6 is met -> committed "10" (baseline None). This is the
  whole -3.30: 311 of 8,010 boxes turn from a correct None into a wrong number.
- id 35 (GT 11): 22 -> 14, both wrong; id 43 (GT None): 23 -> 3, both wrong; the 92-box fragment of a "23"
  player (crops read 23 clearly, GT places it under identity 19 = 28 within 2 m) is now its own identity instead
  of being merged into id 43 - a correct split.
`vote_density.py` over all valid-12 identities under jnrstar2_u02_c08 (conf 0.8): committed right tracks have
reads on 7-25 % of their boxes (10/25/50 % quantiles 0.072 / 0.13 / 0.246, votes 26 / 52 / 106); the 12 false
commits sit at 0.014 / 0.019 / 0.067 (votes 6 / 7 / 8), the 23 wrong at 0.021 / 0.041 / 0.09. A floor of
2 % reads-per-box would abstain 7 committed identities: 4 false commits (SNGS-021 id 30, 030 id 49, 026 id 33,
024 id 38; 1,789 boxes) + 3 wrong, 0 right; 3 % takes 3 right ones. Implemented as GSR_MIN_VOTE_FRAC (default
0, bit-identical) in clipcalib/tools/s3b_splitfix.py and the s3b_dump copy (the vote object now carries the box
count). Chain /mnt/d/jersey-lab/vote_frac_chain.sh: verify run, jnrstar2 c08 f0.02 / f0.03, jnrmix f0.02, k1 f0.02.
Results (valid-12, k1 chain; journal rows jt_vf_verify, jt_*_f02/f03):
| config | GS-HOTA | up / down | worst |
|---|---|---|---|
| verify run, floor 0 (must equal k1_on) | 67.26 | 0 / 0 | 0.00 |
| JNR-first + star v2 (c0.8), floor 0 | 68.40 | 7 / 4 | -3.30 (024) |
| **JNR-first + star v2 (c0.8), floor 0.02** | **69.06 (+1.80)** | 7 / 2 | -1.90 (027), -0.8 (022) |
| JNR-first + star v2 (c0.8), floor 0.03 | 68.48 | 7 / 3 | -4.71 (030: right tracks lost) |
| JNR + ConvNeXt hybrid (jnrmix_u02w2), floor 0.02 | 68.37 | 5 / 2 | -3.61 (027) |
| ConvNeXt + gate (k1 chain), floor 0.02 | 66.98 (-0.28) | 2 / 2 | -4.70 |
The floor helps only the readers that vote on many boxes (JNR / star: right tracks read on 7-25 % of their
boxes); the gated ConvNeXt path votes sparsely and loses right tracks. 021 +4.4, 024 and 030 back to 0.
Remaining downs under f02 (`members_diff.py`, /mnt/d/jersey-lab/star024/diff_027_022_f02.md): 027 is a
merge/split rearrangement of three identities driven by the changed votes (a right "19" identity loses its
number, a "23" identity gains one, net -1.9); 022 is a "15" identity read as "5" (417 boxes) plus a merge.
Selection caveat: u 0.02 / w2 / star conf 0.8 / floor 0.02 are all chosen on valid-12 = one match.
Sensitivity (same chain, rows jt_*_f015/f025, jt_jnrstar2_u02_c05_f02, jt_star_v2_nogate_c0.8_f02):
floor 0.015 -> 68.87 (024 -3.3 stays: 6/311 = 0.019), 0.02 -> 69.06, 0.025 -> 68.79 (029 -1.7, a right identity
lost), 0.03 -> 68.48. Star conf 0.5 + floor 0.02 -> 68.81 (024 -4.9 stays: more low-conf reads pass). Star v2
alone (no JNR, no gate) + floor 0.02 -> 68.39 (+1.12, worst -1.90). The plateau 0.015-0.025 is 68.8-69.1; the
exact optimum is a step on single identities of one match.

## 2026-08-19 FIFA tracking, assembler rows

`s3b_dump.py fifa all` (GSR_DATA_ROOT=/mnt/d/fifa/sn_layout, GSR_MERGE_PREDICT=1 GSR_SPLIT=0.22 GSR_OCR_CONF_TH=0.95,
GSR_UNDISTORT_FOOT=0 because the s2 PnLCalib calib carries no k1) -> members_pnl -> `fifa_eval_track.py --tracker
splitfix_pnl`: **HOTA 68.11 (DetA 71.42 / AssA 65.03), IDF1 86.30, MOTA 92.11, 511 IDSW, 774 ids for 446** vs raw
BoT-SORT 61.19 / 52.56 / 72.53 / 613 / 1,219: the merge stage adds +6.9 HOTA and +13.8 IDF1 out of domain, with
PnLCalib pitch coordinates. BroadTrack-camera rows (btf1 / btf2, with k1) follow from fifa_bt_chain.sh.
/mnt/d/fifa/eval/track_eval_splitfix_pnl.md.
BroadTrack rows (fifa_bt_chain.sh: pass 1 07:26-09:05, self-tripod, pass 2 09:05-10:56, broadtrack_to_calib
--tag btf1/btf2, s3b_dump with GSR_UNDISTORT_FOOT=1): splitfix + btf1 **68.40 / 65.59 / 86.64 / 500 IDSW / 770 ids**,
+ btf2 68.18 / 65.17 / 86.26 / 504 / 769. Camera source is a +0.3 / -0.2 effect (per clip -1.0 .. +1.4); the
assembler itself is +7.2 HOTA over raw BoT-SORT (18/20 clips up, median +4.7). /mnt/d/fifa/eval/track_eval_*.md.

## 2026-08-27 chain-compose: roster-abstain port NEGATIVE; label-free JNR path validated

- roster_reject.py (new): out-of-roster reject as a post-pass over a GSR_DUMP_MEMBERS
  dump, generic-format vote counts, scope=game (halves joined by side flip), high=30 -
  the pre-registered winner from pipeline-lens 2026-08-17. On the composed 69.31 state
  (tun_k1jr_sportsl80): 69.19 (-0.12), worst SNGS-021 -2.56. Nulled 8 identities:
  2 GT-None false commits pay +2.15 (SNGS-024/028), but SNGS-021 id 11 committed 3 with
  GT 3 (299/362 boxes) and a wrong team side, and the side-keyed roster nulls it.
  Verdict: the vote-density floor already took the old rule's payoff (GT-None commits);
  the residual is dominated by team-side errors. CLOSED NEGATIVE on this state; no
  knob variants swept (n=1 fishing). Dump: /mnt/d/jersey-lab/chain_compose/dump_k1jr
  (s3b_dump fork, emitted tree bit-identical to tun_k1jr_g0).
- jnr_all.py (new): label-free single-pass JNR extraction over ALL player/gk detections
  (no labels file opened; frame names from frame_idx). Validated on SNGS-021/029 vs the
  matched+unmatched union of jnr_valid12: key coverage EXACT, max |dprob| 2e-3, max
  |dunc| 2e-3 (fp16 storage + batch composition). This is the extraction that may run
  on test-49 (a logged touch, owner decision); jnr_valid12.py must not.
- build_jnr_cache.py: seq list now derived from the npz dir (was hardcoded 21..32).
- Chain composition results recorded in clipcalib journal 2026-08-27 and
  ltpi-research/results/bt3_valid_chain/README.md: 69.06 + SportsL links = 69.31,
  4 up / 0 down, first conjunction-significant candidate on valid-12.
