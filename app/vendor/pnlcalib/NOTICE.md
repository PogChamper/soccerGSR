# PnLCalib notice

This directory contains a runtime subset of PnLCalib:

- `utils/utils_calib.py`
- `utils/utils_heatmap.py`
- `utils/utils_optimize.py`

Upstream: https://github.com/mguti97/PnLCalib

The upstream project is licensed under GPL-2.0. Local changes drop the
training-time Gaussian heatmap-label generators from `utils_heatmap.py`, the
optical-flow and camera-vector helpers from `utils_optimize.py`, and the HRNet
model definitions, their config YAMLs and the `utils_field` / `utils_geometry`
/ `utils_keypoints` / `utils_lines` modules. The one addition is a NumPy port
of the torch maxpool + top-k heatmap decoder, so inference runs under
onnxruntime. See `LICENSE-GPL-2.0`.
