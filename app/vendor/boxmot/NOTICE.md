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
  cmc/{base_cmc,ecc,sof,orb}.py + __init__.py
trackers/
  basetracker.py
  detection_layout.py
  botsort/{basetrack,botsort,botsort_track,botsort_utils}.py
utils/
  iou.py
  matching.py
  visualization.py
  ops.py            (subset: xyxy2xywh, xywh2xyxy only)
  __init__.py       (trimmed: logger setup only)
```

The original `boxmot.reid.*`, `boxmot.engine.*`, all other trackers and
`boxmot.utils.{checks,misc,ops,torch_utils}` are **not** vendored.

## Modifications

Local changes to the vendored code:

1. **All absolute imports rewritten** from `from boxmot.X import Y` to
   `from app.vendor.boxmot.X import Y` (including the lazy CMC registry
   strings in `motion/cmc/__init__.py`).
2. **`trackers/botsort/botsort.py`**:
   - removed `import torch` and the `torch.device` type hint;
   - `device` and `half` are now no-ops (kept for signature
     compatibility);
   - removed the dependency on `boxmot.reid.core.auto_backend.ReidAutoBackend`
     — `with_reid=True` is supported only when the caller passes
     pre-computed embeddings via `update(..., embs=...)`.
3. **`utils/ops.py`** rewritten as a numpy-only subset (`xyxy2xywh`,
   `xywh2xyxy`).
4. **`trackers/__init__.py`** trimmed: only re-exports `BotSort`.
5. **`utils/__init__.py`** trimmed: only logger setup + path constants.
6. **`trackers/basetracker.py`**: dynamic import path for `TrackState`
   redirected from the bytetrack module to our botsort `basetrack`.

## Why vendor instead of `pip install boxmot`

We needed a torch-free runtime for tracking (the rest of the inference
stack runs through `onnxruntime-gpu`). The motion + matching core of
BoT-SORT is pure numpy/scipy/cv2; only the bundled ReID backbones bring
PyTorch. By vendoring the tracker and dropping ReID we eliminate the
~2 GB of `torch + nvidia-cuda-*` wheels from this code path.

## License obligations

This codebase as a whole, when distributed alongside the vendored
BoxMOT files, must comply with AGPL-3.0. If you operate a network
service based on this software, AGPL §13 requires that you offer your
users the corresponding source.
