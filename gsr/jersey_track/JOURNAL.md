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

## 2026-08-27 tracklet-reader P0 rails: generic per-track harness, exact 69.31 profile

- `analysis/jersey-2026-08-17/jersey-error-decomp/oracle_assoc.py` replays both jersey cache
  formats. The branch is picked from the arity of a frame entry, the way the assembler picks
  GSR_JERSEY_FMT: 3-tuple (vis, tens, units) = ConvNeXt logits, 2-tuple (numbers, confs) =
  generic reader. The generic branch is the assembler's own rule (s3b_splitfix.py frag_votes /
  decode_votes): accept a crop when number >= 0 and conf >= CONF_TH, vote weight = conf,
  commit = argmax of the weighted counts when nv >= MIN_VOTES. No GSR_MIN_VOTE_FRAC floor -
  the harness scores GT tracks, not assembled identities. VIS_TH is inert in that branch.
  Valid-12, perfect association (best GT IoU >= 0.5), 187 known-jersey GT player tracks:
  | reader | hit | wrong | miss | false commit (of 50 GT-None tracks) |
  |---|---|---|---|---|
  | jersey.pkl logits, VIS 0.7 / CONF 0.95 / MV 6 | 156 | 19 | 12 | 8 |
  | jnrstar2_u02_c08 generic, CONF 0.5 / MV 6 | 165 | 16 | 6 | 9 |
  The logits row is bit-identical to the pre-change run (the recorded 156/19/12, FC 8), so the
  branch is additive. The 69.06 reader is +9 hits / -3 wrong / -6 miss for +1 false commit, and
  all 6 of its misses are "conf killed it" - that format has no visibility gate to blame.
- `vote_profile.py` (new, fork of vote_density.py; the original is untouched). Two fixes:
  identities whose boxes match no GT box at IoU >= 0.5 are no longer scored as false commits on
  a GT-None identity (unmatched_commit vs false_commit, and the same split on the abstain side),
  and the vote-density floor is `--floor` instead of a hardcoded ladder. `--compose-tag` pools
  the dump identities through a later tree (dump row i is composed prediction i for i < len(rows))
  and takes the committed jersey from there, so one pass profiles the pre-link and the composed
  state. Run: `--dump-dir /mnt/d/jersey-lab/chain_compose/dump_k1jr --cache jnrstar2_u02_c08
  --conf 0.5 --floor 0.02 --compose-tag tun_k1jr_sportsl80`.
- Exact error profile of the 69.31 state (identities / boxes; player and goalkeeper roles,
  114,882 dump rows of 123,885). base = the dump = tun_k1jr_g0 = 69.06; composed = the same
  identities pooled through tun_k1jr_sportsl80 = 69.31:
  | class | base ids | base boxes | composed ids | composed boxes | box share |
  |---|---|---|---|---|---|
  | right | 162 | 78,144 | 162 | 78,376 | 68.22 % |
  | wrong | 20 | 7,457 | 20 | 7,457 | 6.49 % |
  | false_commit (GT jersey None) | 8 | 2,388 | 8 | 2,388 | 2.08 % |
  | unmatched_commit (no GT match) | 0 | 0 | 0 | 0 | 0 % |
  | abstain_known | 48 | 7,201 | 46 | 6,969 | 6.07 % |
  | abstain_none | 70 | 19,669 | 70 | 19,669 | 17.12 % |
  | abstain_unmatched | 2 | 23 | 2 | 23 | 0.02 % |
  | total | 310 | 114,882 | 308 | 114,882 | |
  /mnt/d/jersey-lab/tracklets_v1/profile_6931.csv, per-identity detail in
  profile_6931_identities.csv. The 996 interpolated rows of the composed tree (0.8 % of 124,881)
  are outside the dump alignment and not counted.
  Readings: (1) the split was worth making but binds on nothing here - all 8 false commits are
  real GT-None identities, so the old vote_density number was right by luck; (2) SportsL buys
  exactly two abstain_known fragments, absorbed into an already-right identity - SNGS-026 base
  id 14, 198 boxes, GT 14, and SNGS-031 base id 71, 34 boxes, GT 11 - 232 boxes in all, and it
  touches no wrong and no false commit; (3) all 15 goalkeeper identities are abstain_none, and
  the remaining 55 abstain_none are player identities matched to GT-None tracks; (4) at floor
  0.02 no committed identity sits below the floor, which re-confirms the dump was built with
  GSR_MIN_VOTE_FRAC=0.02 (floor 0.05 would take 18 identities: 14 right, 4 wrong).
  9 of the 20 wrong are tens-digit errors with the units right (42->12 x2, 23->3 x2, 14->4 x2,
  15->5, 28->18, 33->13); the largest single wrong identity is SNGS-023 id 52, 657 boxes, read 14
  for GT 11.
- `clipcalib/tools/tune_gsr.py` LEVERS: added GSR_MIN_VOTE_FRAC and GSR_JERSEY_FMT. Journal rows
  were dropping both, which is why the k1jr_g0 config cell needed a prose note to say the state
  is generic-format at floor 0.02.

## 2026-08-27 tracklet-reader P1: tracklet dataset v1

One script, three subcommands, all CPU and skip-if-exists per clip:
`gsr/jersey_track/build_tracklets.py {logits,crops,manifest}` (base conda python, numpy/cv2/pandas).
Detector-to-GT matching is `mine_train_crops.match_seq(split, seq, 0.5)` unchanged.
Output root /mnt/d/jersey-lab/tracklets_v1, 147 MB in 21,024 files. Wall time 3.3 min total.

- `logits --split train` (41 s, 57 npz, 60.1 MB) -> logits_train/<seq>.npz. Per clip: the matched
  group `m_*` (frame, det, track, role, vis logit, tens[10], units[10], xyxy, det conf, IoU,
  scored flag) sorted by (track, frame) so every GT track is a contiguous slice; the unmatched
  group `u_*` with the same per-detection fields (the abstain-side distribution at inference);
  the track index `t_*` (track, role, label with -1 for GT-None, GT box count, slice start/end)
  covering every player/gk GT track including those with zero matched detections; plus
  `image_ids` for frame_idx -> image_id.
  631,548 player/gk detections, 608,214 matched / 23,334 unmatched, 1,224 GT tracks.
- `logits --split valid --seqs valid12` (8 s, 12 npz, 11.4 MB) -> logits_valid12/.
  119,360 detections, 113,667 matched / 5,693 unmatched, 251 tracks, 187 known player tracks -
  the same 187 the P0 per-track harness scores.
- `crops --seqs tail17` (2 min 24 s, 20,936 jpg quality 95, 75.1 MB) -> crops_tail17/<seq>_<track>/
  <seq>_<image_id>.jpg plus index.csv (seq, track_id, image_id, path, gt_jersey, role, team, w, h)
  and _index/<seq>.csv as the resume marker. GT player boxes (category 1) of SNGS-154..170 on
  frames 1, 9, 17 ... 745 - the same 94 frames per clip the ltpi cache uses. 335 player tracks,
  269 with a number and 66 GT-None; 14 boxes skipped as under 4x8 px; crop height median 101 px
  against 100 px in the pool, so the two sources are the same sampling.
- `manifest` (0.6 s) -> manifest_tracks.csv, 1,224 rows (seq, gt_track, label, role, n_boxes_gt,
  n_dets_matched, n_crops_stride8, crop_source).

Totals over the 57 train clips (MEASURED, manifest_tracks.csv):

| class | tracks | GT boxes | matched dets | stride-8 crops |
|---|---|---|---|---|
| player, known number | 933 | 512,009 | 496,314 | 64,122 |
| player, GT-None | 212 | 92,962 | 88,891 | 11,636 |
| goalkeeper | 79 | 24,107 | 23,009 | 0 |

933 known and 212 None hit the recon targets exactly, and 512,009 GT boxes / 496,314 supervised
detections reproduce the charter's two independent counts. Crop sources: 865 tracks from the pool
(40 clips), 359 from crops_tail17 (17 clips). 81 tracks carry no stride-8 crop: the 79 goalkeepers
(neither source cuts them) and two 4- and 5-box GT-None player tracks (SNGS-110 track 16,
SNGS-112 track 26) that never land on a sampled frame. Detector recall against GT is 96.9 % of
boxes on known player tracks; the shortest known player track has 7 crops, the median 73.

Verdict: P1 (a), (b), (d) done. (c), the dev-split logits for the 40 valid clips, waits on the
running s1 pass over SNGS-039..059 / 078..096.

## 2026-08-27 tracklet-reader P2: feature-level v0 CLOSED NEGATIVE (the control did its job)

- Rails: dev-40 full-rate logits built after the s1 pass (40 clips, 464,290 dets,
  691 known / 101 None player tracks, tracklets_v1/logits_dev40). Incumbent on this
  rail (unweighted s3b vote VIS0.7/CONF0.95/MV6): 521 hit / 44 wrong / 126 miss,
  FC 15, net 462. Ungated poolings are far negative (-210..-326), as recorded.
- v0a (transformer d96 x2 over per-det ConvNeXt softmax + geometry, fragment-window
  augmentation, tau swept on dev): best dev net 397 (tau 0.7: 495/96/100 FC 19).
- v0b (+ the gate decision and its track rate as input features, tau grid x8):
  best dev net 407; wrongs drop 96 -> 66 but hits stay ~470.
- VERDICT: G-dev (net >= 477 at FC <= 15) failed by 55-65 net on both variants.
  Mechanism: the gated vote's precision comes from hard evidence selection; a
  learned reweighting of the SAME per-crop posteriors converts misses into wrongs
  because on hard tracks the posteriors are systematically wrong (88.5 percent of
  accepted crops wrong on wrong tracks, recorded 2026-08-17). No feature-level
  model can change what the evidence says. 933 known tracks is also thin for a
  100-class readout. Feature-level arm CLOSED; per charter the pixel arm proceeds -
  its premise (sub-threshold pixel glimpses integrated across crops) is exactly
  what v0 cannot see.
- Runs: tracklets_v1/runs/v0{a,b}_s42 (best.pt, last.pt); code tracklet_v0.py
  (train/eval/baselines; the baselines subcommand is the standing incumbent rail).

## 2026-08-27 tracklet-reader P3: pixel arm PASSES G-dev as a vote-first hybrid

Iteration ladder on the dev split (691 known / 101 None player tracks; incumbent
unweighted vote 521/44/126 FC 15, net 462; gate net >= 477 at FC <= 15):
- pix_a (ConvNeXt-T + cross-crop transformer + tied-digit head, track-level CE only):
  net -110. Train 870/933 vs dev 230/691 = kit memorisation of the 3 train matches.
- pix_b (+ per-crop aux loss on trunk tokens, + external kit-diverse mix
  jersey-2023/soccer_crops ~8k crops, + trunk LLRD x0.1, stronger colour augs):
  net 243 (415/167 FC 5). Memorisation broken, precision still short.
- pix_c (aux loss only on vis_p >= 0.7 crops - stop teaching hallucination on backs;
  eval selection top-K=16 by vis_p, train pick half top-vis half random):
  net 384 (ep 34: 524/120/47 FC 20). HITS NOW EXCEED THE VOTE (524-531 vs 521);
  the whole残 gap is commit precision.
- Decision-rule layer (no training): abstain if absent > tau OR digit conf < c;
  conf separates (right median 0.885 vs wrong 0.517). Grid on dev: net 418
  (472/47/172 FC 7) at tau 0.10 / conf 0.55.
- HYBRID (the recorded winning shape, vote-first + fill): vote where it commits,
  else pix_c at (tau 0.10, conf 0.50): net 504 (566/53/72 FC 9). Pix-override
  variant (model overrides the vote at conf >= 0.87): net 510 (569/50/72 FC 9).
  G-dev PASSED. All thresholds selected on dev; valid-12 untouched.
Artifacts: tracklets_v1/runs/pix_{a,b,c}_s42/ (best.pt, dev_dump.npz for c);
code tracklet_pix.py. Next: P4 - fragment-level inference on valid-12, priority-merge
generic cache (jnrstar2 reads + model fills), chain vs 69.31; 3 seeds.

## 2026-08-27 tracklet-reader P4: chain integration NEGATIVE with a full mechanism; track closed

Three integration forms of the G-dev-passing hybrid (thresholds frozen on dev), all
measured against the 69.31 composed state (tun_k1jr_sportsl80):
- merge-cache with overrides (pix_infer.py, jersey_pixmix1): 68.06 (+SportsL). 71
  overrides at conf >= 0.87 mostly agree with GT, but the killers repeat the recorded
  per-crop confusions ON FRAGMENTS at high confidence: 19->10 three times (the "10"
  training-prior collapse), 14->4 (tens drop), None->1.
- fill-only cache (jersey_pixmix2): 68.60. 21 fills; SNGS-021 -4.23 via the
  ASSOCIATION AMPLIFIER: a wrong or false fragment commit merges by (team, jersey)
  into a real player identity and poisons hundreds of boxes; the dev rail cannot see
  this by construction.
- post-pass fill on the emitted tree (no merge exposure): 69.02 (-0.29). 17 identities
  filled; 16 are metric-zero (most filled GT-None identities are already METRIC-DEAD -
  team or role mismatched, so jersey is irrelevant - and the alive known-but-abstained
  identities the model commits on are tiny fragments); the sign is set by ONE alive
  GT-None identity (SNGS-032 id 57, 491 boxes, fill "8", -3.67).
- G-track rail (oracle_assoc generic, detector crops): pixmix2 163/18/6 FC 10 vs
  incumbent 165/16/6 FC 9 - gate (>= 170, FC <= 9) not met.
- Box-weighted re-selection of the fill point on dev says "fill more" (GT-None box
  mass on dev is small) - the dev split, being GT tracks, structurally lacks the two
  populations that decide the chain outcome: DBSCAN fragments and ghost identities.
VERDICT: the pixel tracklet reader is a real track-level advance (G-dev net 510 vs
462, hits 569 vs 521, FC 9 vs 15) but its chain integration on this state is negative;
converting track-level reading gains into GS-HOTA needs fragment-level selection data
that neither valid-12 (selection budget spent) nor dev-as-GT-tracks can provide. The
missing piece is the dev-40 fragment build (s2+s3a over valid games 3+5) - an
owner-gated protocol decision. Track closed at the charter's fallback: mechanisms
measured and written. Runs: tracklets_v1/runs/pix_{a,b,c}_s42; caches jersey_pixmix{1,2};
decisions tracklets_v1/decisions/; code tracklet_pix.py, pix_infer.py.

## 2026-08-28 tracklet-reader P4b: fragment rail built end to end; safe-null verdict on 69.31

The owner pushed past the "needs dev-40" stop, so the missing piece was built:
- s2 embeddings (emb-only variant s2_embed_only.py, ~2.5 min/clip) + s3a BoT-SORT over
  the 40 dev clips -> real DBSCAN fragments for selection (2,192 fragments: 1,534 known
  / 606 alive GT-None / 48 ghosts; frag_rail.py, dump tracklets_v1/frag_rail_dev40*.csv).
- Fill objective selected ON REAL FRAGMENTS (gain = box mass of right fills on
  vote-abstained known fragments, loss x2 = mass of commits on alive GT-None; wrongs on
  known free per the evaluator): pix_c point (tau 0.20, conf 0.80) = 38 right / 10
  wrong / 6 none (4,612 vs 327 boxes); pix_d (logit-adjust 1.0, retrained) at its point
  (0.55, 0.40) = 45 / 27 / 12 (8,451 vs 687). pix_d dev net 329 (adjustment flattens
  the confidence scale, does not improve right-wrong separation).
- Frozen transfer to valid-12 (post-pass fill on tun_k1jr_sportsl80): pix_c point fills
  4 identities, pix_d point fills 8 - COMBINED exactly 69.30751 in BOTH cases, zero
  clips changed. The rail fixes SAFETY completely (previous forms lost 0.29-1.25); the
  GAIN on this state is empty: every confident fill lands on a metric-dead or tiny
  identity, because the JNR+star+floor ensemble already harvested the abstain margin.
FINAL TRACK VERDICT: the tracklet reader reads tracks better than the vote (G-dev +48
net) but that advantage is redundant on the 69.31 state's abstain margin, and the
remaining prize (wrong-commit direction, +3.80 family) needs override-grade
right-vs-wrong separation on fragments that neither pix_c nor pix_d has. The reusable
deliverables stand: the fragment rail (the honest selection loop any future track-level
reader needs), dev-40 emb/track caches, the safe fill machinery. Next real move if the
line reopens: train ON fragments (data now exists) for override-grade precision.

## 2026-08-28 (night) SoccerFactory jersey stream: pre-registered audit run, gate met via star confirmation

- 300-crop visual audit (4 Opus vision graders, every crop graded twice, disagreement 21%),
  sample: parts 1-2, 160 clips, numbered boxes at authors' legibility >= 0.5, median h 133 px.
  RAW verdicts: A 171 / B 63 / C 62 / D 4 -> A+B 78% (gate >= 90% FAILED), D 1.3% (gate <= 2%
  passed). The failure is illegibility, not wrong labels; legibility thresholding cannot fix it
  (A+B 86% even at leg >= 0.9; curve in the workflow record wf_651b6c5f-099).
- CONFIRMATION FILTER (the recorded 45.2->76.9 route, star v2 as the confirmer): keep if
  leg >= 0.8 AND star v2 reads the SAME number at conf >= 0.5. On the audited crops: kept
  198/241 (yield 82%), A+B 91.4%, D 0.0% - GATE MET for the filtered stream.
- Night launch: sf_mine.py (5 CPU shards, leg 0.8, cap 40/clip, <= 1 crop of a number per
  second) over the 2,000 part-1/2 clips -> sf_confirm.py (star ONNX, CPU). Output
  /mnt/d/jersey-lab/sf_mine_v1/{crops,manifests,confirmed}. Purpose: kit-diversity single-crop
  stream (~500 games) for star_v3 and pix_e retrains - match diversity is the one axis with a
  recorded large win (81.96 -> 94.16) and it just moved pix_a -> pix_b by +353 dev net.
- Parallel GPU night chain: s2_calib_only (PnLCalib cameras, dev-40) -> jnr_all (dev-40) -
  the legs for an apply-only transfer check of the 67.26 / 69.06 / 69.31 configs on 40 fresh
  clips of 2 games before any test exposure. Camera source moves the assembler by +-0.3 only
  (FIFA measurement), adequate for an identity-knob check.

## 2026-08-28 (night) dev-40 transfer check: the jersey candidate's gain TRANSFERS (n=3 games now)

Apply-only replays on the 40 never-tuned valid clips (games 3+5), all knobs frozen from
the valid-12 selections; PnLCalib cameras (calib.pkl), no k1 - deltas are the answer,
absolutes are not comparable to the bt5k chain. Journal rows dev_base / dev_hy / dev_jr;
perseq tun_dev_*.json.
- base (OCR 0.95): 52.24. JNR+ConvNeXt hybrid: 54.47. jersey 69.06-config
  (jnrstar2 + floor 0.02): 56.03.
- Paired per-clip stats vs base: hybrid +2.14, CI [+1.17, +3.16], game3 +3.58 / game5
  +0.55, 23 up / 3 down, worst SNGS-095 -3.40. Jersey config +3.77, CI [+2.29, +5.36],
  game3 +6.19 / game5 +1.10, 28 up / 7 down, worst SNGS-095 -4.68.
- VERDICT: the jersey candidate's gain is not a one-match artifact - CI excludes zero on
  two unseen games, positive in both, ordering (jersey > hybrid > base) preserved. This
  was the missing evidence for the pending apply-only test exposure of the 69.31 chain.
  Caveats: different camera rail (deltas only), real between-game heterogeneity (game5
  ~3x weaker), 7 of 40 clips negative with worst -4.68 - the test-side do-no-harm
  expectation should be set accordingly.
- Legs built this night and now standing assets: calib.pkl (PnLCalib) 40/40,
  /mnt/d/jersey-lab/jnr_dev40 (label-free JNR, 40/40), star_v2_dev caches 40/40,
  jersey_{jnrmix_u02w2,jnrstar2_u02_c08}.pkl on all 40 dev clips.

## 2026-08-28 (night) star_v3 (SoccerFactory diversity): crop-level gain, chain-level null - closed

- Trained with the confirmed SF stream (36,980 crops, class cap 1500) added to the star v2
  recipe (gsr_star_v3.yaml, model.num_classes=11; one false start without that override).
  Dev-val (games 3+5 crops) number_acc 78.7 vs v2's 77.1 (+1.6 pp on the same selection split).
- Chain, frozen knobs: valid-12 69.31-form with jnrstar3 = 69.36 (+0.05, noise); dev-40
  jersey config 55.67 vs 56.03 (-0.36 ON THE SAME GAMES whose crops got more accurate).
  The per-crop-vs-vote trap reproduced again (blend v1f pattern, milder). VERDICT: SF kit
  diversity does not convert through the current vote layer at frozen thresholds; the SF
  stream stands as a clean asset (audit-gated) for future readers; per-knob re-sweeps for
  v3 not attempted (would be dev-selection work for a sub-noise prize).
- Assets: /mnt/d/jersey-lab/runs/star_v3 (ckpt + reader.onnx), jersey_star_v3* and
  jersey_jnrstar3_u02_c08.pkl caches on valid-12 and dev-40.

## 2026-08-28 (night, addendum) star_v3 re-selected on the dev-40 chain rail: the splits disagree

Caught my own protocol miss: the star_v3 null verdict above used INHERITED v2 knobs
("never inherit a confidence threshold across models"). Re-selected on the dev-40 chain
rail (a legitimate selection split; 8 configs, journal rows dsw_*): best is conf 0.5 /
floor 0.02 at 56.34 - ABOVE the v2 config's 56.03 there. Applied apply-only to valid-12
(69.31 form): 67.70, well below v2's 69.31. So dev-40 prefers v3, valid-12 prefers v2:
the reader sub-choice is game-dependent at a scale larger than the v2-v3 gap, and with
n=3 games it is UNRESOLVABLE. What stays resolved on both rails: the config family
(JNR-first + star fill + floor >> base, and > hybrid). Standing verdict: v2 remains the
incumbent by valid-12 precedent; v3 files as equivalent-within-game-noise with a clean
data asset behind it. Any future reader adjudication needs more games (the dev-40 rail
now makes that routine).
