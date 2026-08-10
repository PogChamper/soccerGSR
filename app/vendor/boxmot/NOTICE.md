# Vendored BoxMOT (BoT-SORT only)

This directory contains a redistribution of selected files from the
[BoxMOT project](https://github.com/mikel-brostrom/boxmot)
by Mikel Broström, licensed under the **GNU Affero General Public License
v3.0 (AGPL-3.0)**. The full license text is in `LICENSE-AGPL`.

## What is included

Only the minimal set of files required to run the BoT-SORT tracker on
pre-computed detections + (optionally) external appearance embeddings:

```
motion/
  kalman_filters/{base,xywh}.py
  cmc/ecc.py + __init__.py
trackers/
  basetracker.py
  botsort/{basetrack,botsort,botsort_track,botsort_utils}.py
utils/
  matching.py
  ops.py            (xyxy2xywh, xywh2xyxy)
  __init__.py       (standard-library logger only)
```

The original `boxmot.reid.*`, `boxmot.engine.*`, all other trackers and
`boxmot.utils.{checks,misc,torch_utils}` are **not** vendored.

## Modifications

Local changes to the vendored code:

1. **All absolute imports rewritten** from `from boxmot.X import Y` to
   `from app.vendor.boxmot.X import Y`.
2. **`trackers/botsort/botsort.py`**: removed `import torch`, the
   `torch.device` type hint and the dependency on
   `boxmot.reid.core.auto_backend.ReidAutoBackend` - `with_reid=True` is
   supported only when the caller passes pre-computed embeddings via
   `update(..., embs=...)`.
3. **`utils/ops.py`** rewritten as a numpy-only subset (`xyxy2xywh`,
   `xywh2xyxy`).
4. **`trackers/__init__.py`** and **`trackers/botsort/__init__.py`** emptied;
   `BotSort` is re-exported from the package root only.
5. **Camera-motion compensation** trimmed to the ECC implementation used by
   the service, and the single-implementation `BaseCMC` base class folded
   into `ECC`.
6. **Tracker visualization** and its rendering-only history/`max_obs` state
   removed; rendering is implemented by the service.
7. **`utils/__init__.py`** uses a standard-library logger and does not modify
   process-wide logging configuration.
8. **Per-class tracking** removed from the vendored core. The service isolates
   humans and the ball with independent tracker instances.
9. **Oriented-bounding-box support removed.** Detections are accepted only as
   `(x1, y1, x2, y2, conf, cls)`.
10. **The pluggable association layer `utils/iou.py` removed.** Matching is
    IoU-only; the one surviving kernel lives in `utils/matching.py`.
11. **Unused Kalman surface removed**: the stateful matrix filter API, the
    gating distances and the `chi2inv95` table, and the single-track
    `predict` step (BoT-SORT only uses the vectorized `multi_predict`).
12. **Unused tracker knobs removed**: `det_thresh`, `max_age`, `min_hits`,
    `iou_threshold`, `asso_func`, `reid_weights`, `device`, `half`.

## Why vendor instead of `pip install boxmot`

We needed a torch-free runtime for tracking (the rest of the inference
stack runs through `onnxruntime-gpu`). The motion + matching core of
BoT-SORT is pure numpy/scipy/cv2; only the bundled ReID backends bring
PyTorch. The service supplies OSNet embeddings through the tracker API, so
removing those bundled backends eliminates the ~2 GB of
`torch + nvidia-cuda-*` wheels from this code path.

## License obligations

This codebase as a whole, when distributed alongside the vendored
BoxMOT files, must comply with AGPL-3.0. If you operate a network
service based on this software, AGPL section 13 requires that you offer your
users the corresponding source.
