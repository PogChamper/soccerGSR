# soccerGSR: Game State Reconstruction in Soccer

Reconstructing the game state of a football match

![Status](https://img.shields.io/badge/status-completed-brightgreen)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Framework](https://img.shields.io/badge/framework-PyTorch-orange)
![Service](https://img.shields.io/badge/service-FastAPI-009688)

A team project by 1st-year Master's students of the "Artificial Intelligence" program at HSE University (Faculty of Computer Science). The goal of the project is to build a system for analyzing football broadcasts using computer vision.

---

## Project Goal

The main goal of the project is to design and ship an end-to-end system that takes a clip of a football match as input and performs **player detection and tracking** on it. The final solution is packaged as an interactive web service for a clear demonstration of the models.

---

## ML Service (FastAPI, async, GSR)

End-to-end Game State Reconstruction for football broadcasts. Every ML stage runs on GPU via `onnxruntime-gpu`. One GPU — one worker at a time.

### Full per-clip pipeline

| Stage     | What it does                                                                                                                                                                                                                              | Model                                                                                                                               |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Pass1     | per-frame: detection → DINOv3 batch-embed → tracking (motion + ReID) → jersey gate+OCR → team embedding sample → field keypoints/lines                                                                                                   | YOLOv5lu, DINOv3 ViT-S+/16, BoT-SORT (vendored, motion+IoU+CMC+ReID), ShuffleNetV2 (visibility), ConvNeXt-Tiny (OCR), HRNet kp/lines |
| Aggregate | per-track: jersey number from logits-mean, team via 2-component GMM on per-track mean DINOv3 embeddings (outfield outliers → referee), independent GK GMM; per-clip: PnLCalib homography per frame, foot-point projection to pitch coords | scikit-learn GaussianMixture, PnLCalib FramebyFrameCalib                                                                             |
| Pass2     | render mp4: bbox + #track\_id + J + team color + pitch minimap with projections                                                                                                                                                          | OpenCV                                                                                                                               |

### REST API (async, primary)

- **POST /forward** — accepts an mp4, returns `202 Accepted` + `job_id`. The worker processes it in the background.
- **GET /jobs/{job_id}** — status (`queued|running|done|error`), stage (`pass1|aggregate|pass2`), progress %.
- **GET /jobs/{job_id}/video** — the finished annotated mp4 (404 until done).
- **GET /jobs/{job_id}/gsr.json** — the ClipState structure: `meta`, `frames` (homography per frame), `observations` (bbox + track_id + visibility_p + team_id + pitch_xy), `tracks` (cls_name, jersey_number, team_label, ...).
- **GET /jobs?limit=&offset=&status=** — a list of recent jobs.
- **POST /forward/sync** — legacy: internally enqueues an async job and blocks until it finishes. Compatible with the old API.

### Additional endpoints

- **GET /history**, **DELETE /history** (admin JWT), **GET /stats**
- JWT auth: `/auth/register`, `/auth/login`, `/auth/me`
- Alembic migrations for `users`, `request_history`, `jobs`

### Contributors

- **OlegNotHehe** (Oleg Rokin) — EDA, YOLOv5 / v5u / x6u baselines, implementation of the main inference script, field calibration
- **PogChamper** (Oleg Baishev) — EDA, DEIMv2 (S/M/L, 640/896), Jersey OCR + Visibility Gate (VLM, ablation), ReID, traciking, team classification, the rest of the service work

---

## Installation & Run

### System requirements

- Linux / WSL2 (tested on Ubuntu 24.04)
- NVIDIA GPU + driver with CUDA 12.x compatibility (tested on RTX 4070 Ti SUPER 16GB)
- Python 3.12 (3.11 also works)

### 1. venv

```
python -m venv venv
source venv/bin/activate
```

### 2. Dependencies

```
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

`requirements.txt` pulls in onnxruntime-gpu 1.20.1 (CUDA 12), transformers 4.57
(for offline DINOv3 export only), torch 2.5.1 + torchvision 0.20.1 (needed for
the heatmap decode in PnLCalib and for the ONNX exports), scikit-learn
(GMM/KMeans), shapely, lap, loguru. **boxmot is not installed** — instead, a
trimmed, torch-free fork under `app/vendor/boxmot/` is used (see
`app/vendor/boxmot/NOTICE.md`, AGPL-3.0 license).

### 3. WSL2 nuance

`app/utils/cuda_env.bootstrap()` itself adds `/usr/lib/wsl/lib` to
`LD_LIBRARY_PATH` and calls `ort.preload_dlls(cuda=True, cudnn=True)` **before**
the first session is created. Without this, `onnxruntime-gpu >= 1.19` fails with
`CUDA failure 100`. The bootstrap is idempotent and is called from `lifespan`.

### 4. DB migrations

```
alembic upgrade head
```

### 5. Run

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

On startup it automatically:

1. Bootstraps CUDA for WSL.
2. Through `app/utils/models_registry`, pulls all models from Google Drive
(gdown) and verifies their sha256.
3. Creates the ORT sessions **before** any `import torch` (important: torch
initializes its own cudnn frontend, after which ORT can no longer create a new
CUDA session on cudnn 9.1).
4. Starts the background job-processing worker.

---

## Models

All DL models are **ONNX**. The registry and auto-download live in
`app/utils/models_registry.py`: each model is downloaded from Google Drive
(gdown) if missing and verified by sha256. The export scripts are only needed to
re-export weights from the source checkpoints.

| Model                      | File                                 | Size        | Re-export                                                      |
| -------------------------- | ------------------------------------ | ----------- | -------------------------------------------------------------- |
| DEIMv2 detector (main)     | `models/deimv2_m_896.onnx`           | 76 MB       | DEIMv2 `export_l_model.py` from `best_stg2.pth`               |
| YOLO detector (legacy)     | `models/best.onnx`                   | 213 MB      | —                                                              |
| Visibility gate            | `models/visibility_gate.onnx`        | ~5 MB       | jersey-visibility-project (ShuffleNetV2)                       |
| Jersey OCR                 | `models/jersey_ocr.onnx` (+ `.data`) | ~110+110 MB | jersey-ocr-project (ConvNeXt-Tiny)                            |
| HRNet keypoints            | `models/hrnet_kp.onnx`               | 264 MB      | `python scripts/export_hrnet_onnx.py kp`                       |
| HRNet lines                | `models/hrnet_lines.onnx`            | 264 MB      | `python scripts/export_hrnet_onnx.py lines`                    |
| DINOv3 embedder            | `models/dinov3_vits16plus.onnx`      | 115 MB      | `python scripts/export_dinov3_onnx.py` (gated on HF, see below) |

### Export PnLCalib HRNet → ONNX

```
python scripts/export_hrnet_onnx.py both
```

The script does everything itself:

1. Downloads `SV_kp` and `SV_lines` (~265 MB each) from
`github.com/mguti97/PnLCalib/releases/v1.0.0`.
2. Loads them through the vendored copy `app/vendor/pnlcalib/model/cls_hrnet*.py`.
3. Exports with opset 17, fixed input `(1, 3, 540, 960)`, and `dynamic_axes`
over batch.

### Export DINOv3 → ONNX

```
python scripts/export_dinov3_onnx.py
```

Downloads `facebook/dinov3-vits16plus-pretrain-lvd1689m` (28.7M params,
embedding 384, gated — you must accept the
[DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)
and be logged in via `huggingface-cli login`), and exports only `pooler_output`
through `torch.onnx.export`, opset 17, dynamic batch axis, input fixed at
`(B, 3, 224, 224)`. The ONNX is ~110 MB; latency on an RTX 4070 Ti SUPER:

- batch 1 → ~6 ms
- batch 8 → ~10 ms (1.3 ms/img)
- batch 22 → ~20 ms (~0.9 ms/img) — a typical frame (~22 players)

In a single pass, the DINOv3 embedding feeds both consumers: BoT-SORT receives
it via `update(..., embeddings=...)` for ReID-aware association, and the
`TeamClassifier` uses it for GMM team clustering.

---

## API Documentation

Swagger: <http://localhost:8000/docs>

### Async (recommended): POST /forward

```
# 1. submit
JOB=$(curl -s -X POST "http://localhost:8000/forward" \
  -F "image=@123.mp4" | jq -r '.job_id')

# 2. poll
watch -n 1 "curl -s http://localhost:8000/jobs/$JOB | jq"

# 3. download
curl -s "http://localhost:8000/jobs/$JOB/video"     -o gsr.mp4
curl -s "http://localhost:8000/jobs/$JOB/gsr.json"  -o gsr.json
```

`gsr.json` contains:

```
{
  "meta": {"filename":"...","width":1920,"height":1080,"fps":30,"frame_count":673,...},
  "frames": [{"frame_idx":0,"H_world2img":[[...]],"H_img2world":[[...]],"cam_params":{...}}, ...],
  "observations": [
    {"frame_idx":0,"track_id":2,"cls_id":0,"bbox_xyxy":[...],"team_id":1,"pitch_xy":[12.3,4.5], ...},
    ...
  ],
  "tracks": {
    "2": {"track_id":2,"cls_name":"player","team_label":"team_b","jersey_number":10,"jersey_confidence":0.99, ...},
    ...
  }
}
```

### Legacy sync: POST /forward/sync

Compatible with the old API: blocks until completion and returns either a
base64 mp4 (JSON) or a stream:

```
curl -X POST "http://localhost:8000/forward/sync" \
  -F "image=@video.mp4" \
  -H "X-Return-Format: stream" -o output.mp4
```

### Misc

```
curl "http://localhost:8000/jobs?limit=10&status=done"
curl "http://localhost:8000/history?limit=10&offset=0"
curl -X DELETE "http://localhost:8000/history" -H "Authorization: Bearer <admin-jwt>"
curl "http://localhost:8000/stats"
curl "http://localhost:8000/health"
```

---

## Authentication

### Registration

```
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username": "user1", "password": "password123"}'
```

### Login

```
curl -X POST "http://localhost:8000/auth/login" \
  -d "username=user1&password=password123"
```

### Creating an admin

Registration via the API never grants admin rights. To create a user or promote
one to admin (needed for `DELETE /history`):

```
PYTHONPATH=. python scripts/create_admin.py admin <password>
```

---

## Detection classes

| ID | Class      | Color       |
| --- | ---------- | ----------- |
| 0  | player     | 🟢 Green     |
| 1  | goalkeeper | 🟡 Yellow    |
| 2  | referee    | 🔴 Red       |
| 3  | ball       | 🟠 Orange    |

---

## Project structure

```
soccer-app/
├── app/
│   ├── main.py                          # FastAPI lifespan: cuda → models → ORT → torch → worker
│   ├── config.py
│   ├── api/
│   │   ├── jobs.py                      # POST /forward (async), GET /jobs/{id}/...
│   │   ├── forward.py                   # POST /forward/sync (legacy wrapper)
│   │   ├── history.py
│   │   ├── stats.py
│   │   └── auth.py
│   ├── models/
│   │   ├── database.py                  # users, request_history, jobs
│   │   └── schemas.py
│   ├── services/
│   │   ├── detector.py                  # YOLO ONNX
│   │   ├── tracker.py                   # wrapper over app/vendor/boxmot (BoT-SORT, no torch)
│   │   ├── jersey.py                    # visibility gate ONNX + ConvNeXt OCR ONNX + per-track aggregation
│   │   ├── embedder.py                  # DINOv3 ViT-S+/16 ONNX (shared ReID + team)
│   │   ├── team_classifier.py           # GMM on DINO embeddings + outlier→referee
│   │   ├── keypoints.py                 # PnLCalib HRNet kp+lines via ORT
│   │   ├── calibration.py               # FramebyFrameCalib wrapper, projection, foot-point → pitch
│   │   ├── minimap.py                   # 2D pitch + player markers overlay
│   │   ├── clip_state.py                # ClipMeta / FrameInfo / FrameObservation / TrackInfo / ClipState
│   │   ├── video_processor.py           # pass1 + aggregate + pass2 orchestrator
│   │   ├── job_worker.py                # asyncio queue + thread executor for GPU jobs
│   │   └── history_service.py
│   ├── utils/
│   │   ├── cuda_env.py                  # WSL2 LD_LIBRARY_PATH + ort.preload_dlls bootstrap
│   │   ├── models_registry.py           # central model spec/download registry (Google Drive + sha256)
│   │   └── visualizer.py                # bbox + track + jersey + team color
│   └── vendor/
│       └── pnlcalib/                    # vendored from github.com/mguti97/PnLCalib (model + utils + config)
├── scripts/
│   └── export_hrnet_onnx.py             # SV_kp/SV_lines .pt → ONNX 540×960
├── models/
│   ├── best.onnx                        # YOLO detector
│   ├── visibility_gate.onnx             # ShuffleNetV2
│   ├── jersey_ocr.onnx (+ .data)        # ConvNeXt-Tiny tens+units
│   ├── hrnet_kp.onnx                    # PnLCalib SV_kp
│   ├── hrnet_lines.onnx                 # PnLCalib SV_lines
│   └── SV_*.pt                          # source weights kept for re-export
├── alembic/
│   ├── env.py
│   └── versions/
│       ├── 2024_..._001_initial_migration.py
│       └── 2026_..._002_add_jobs_table.py
├── alembic.ini
├── requirements.txt
└── README.md
```

---

## Tech stack

- **Language:** Python 3.12 (3.11 ok)
- **Inference:** onnxruntime-gpu 1.20 (CUDA 12, cudnn 9). `torch 2.5.1` stays as a dependency only for the heatmap decode inside PnLCalib and for the offline ONNX export; torch is not loaded on the tracker's inference path.
- **Computer Vision:** OpenCV (headless), NumPy
- **ML models:**
  * YOLOv5lu — detection (player / goalkeeper / referee / ball)
  * BoT-SORT (vendored, app/vendor/boxmot, AGPL-3.0; numpy + scipy + opencv) — motion + appearance tracking, with embeddings from DINOv3
  * DINOv3 ViT-S+/16 — shared appearance embedder for ReID and team clustering
  * ShuffleNetV2 — visibility gate (jersey-visibility-project)
  * ConvNeXt-Tiny two-head — jersey OCR (jersey-ocr-project)
  * HRNet-W48 (×2) — field keypoints + lines (PnLCalib SV_kp / SV_lines)
- **Calibration:** PnLCalib FramebyFrameCalib (per-frame, classical solver)
- **Team clustering:** scikit-learn `GaussianMixture` (k=2 outfield, k=2 GK separately) on 384-d L2-normalized DINOv3 embeddings; outfield tracks with a log-likelihood below the 5th percentile are auto-promoted to `referee`.
- **Service:** FastAPI + Uvicorn, async job worker (1 GPU = 1 worker)
- **DB:** SQLite + SQLAlchemy 2 + Alembic

---

## Licenses & attribution

- **BoT-SORT** — tracking is built on the [BoxMOT](https://github.com/mikel-brostrom/boxmot) code (Mikel Broström, **AGPL-3.0**). `app/vendor/boxmot/` holds a trimmed, torch-free fork (BoT-SORT only, no ReID backbones); the full list of changes and the license text are in `app/vendor/boxmot/NOTICE.md` and `app/vendor/boxmot/LICENSE-AGPL`.
- **PnLCalib** — camera calibration is based on [mguti97/PnLCalib](https://github.com/mguti97/PnLCalib) (**GPL-2.0**); `app/vendor/pnlcalib/` vendors the HRNet definitions, the heatmap decoder, and the calibration optimization. The `SV_kp`/`SV_lines` weights are from the PnLCalib releases.
- **DINOv3** — the embedder uses the [facebook/dinov3-vits16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m) weights under the [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/) (Meta).

---

## One-year work plan

| Checkpoint                   | Deadline                                  | Stage goal                          | Key tasks                                                                                                                       |
| ---------------------------- | ----------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **1. Setup**                 | End of September 2025                      | Formalize the project               | - Create the repository   - Finalize the topic and plan   - Write `README.md`                                                  |
| **2. EDA**                   | Late October 2025                         | Explore the data                    | - Find and analyze datasets (SoccerNet, SportsMOT)   - Write EDA scripts   - Pick the main dataset to work with                |
| **3-4. Baseline**            | End of November 2025 – mid-January 2026   | Build the first working prototype   | - Train a baseline detector (YOLOv8)   - Implement a simple tracker (Kalman Filter)   - Assemble the `Detection + Tracking` pipeline |
| **5. Service**               | Mid-February 2026                         | Package the solution into a demo    | - Build a UI in Streamlit/Gradio   - Integrate the baseline models into the web service   - Demo on test videos                |
| **6-7. Improving the DL part** | End of March – mid-May 2026             | Improve tracking quality            | - Study and integrate a SOTA tracker (BoT-SORT)   - Train/adapt a Re-ID model   - Compare metrics against the baseline         |
| **Defense**                  | June 2026                                 | Finalize and present the project    | - Prepare the final report   - Build the presentation   - Demo the best version of the service                                 |

---

## Team

- **PogChamper** (Oleg Baishev) — Researcher
- **OlegNotHehe** (Oleg Rokin) — Researcher

## Supervisor

- **Mark Blumenau**