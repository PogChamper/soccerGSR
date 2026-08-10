# soccerGSR

Offline game-state reconstruction for soccer broadcast clips. You upload a
clip, the service runs it through a GPU inference pipeline and returns an
annotated MP4 (boxes, jersey numbers, a top-down minimap) and a per-frame JSON
with every player's identity, team, and position in pitch coordinates.

One machine, one GPU, one job at a time. The API is a thin FastAPI layer over
a local queue: no authentication, no multi-tenancy, loopback only. It is meant
for a trusted local machine.

The research pipeline this service was distilled from scored 55.68 GS-HOTA on
the SoccerNetGS public test split. The service has not yet reproduced that
number in a fresh video-to-score run; the exact boundary is documented in
[docs/benchmark.md](docs/benchmark.md).

## How it works

A clip is processed in two passes, so identity decisions can use evidence from
the whole clip instead of being made frame by frame.

Pass 1 walks the video once and collects raw evidence:

- DEIMv2 (a DETR-family detector) finds players, goalkeepers, referees, and
  the ball;
- BoT-SORT links detections into track fragments; people and the ball get two
  independent tracker states, so an id can never migrate between a boot and
  the ball;
- OSNet computes a 512-d appearance embedding for every person crop;
- a ShuffleNetV2 gate decides whether the jersey number is readable, and a
  two-head ConvNeXt OCR votes on the digits when it is;
- two PnLCalib HRNets detect pitch keypoints and line points.

Aggregate consolidates the whole clip in memory:

- an image-to-pitch homography is fitted per frame from the HRNet evidence;
  degenerate and jumping fits are rejected, short gaps are interpolated, long
  gaps are held a few frames from each side and honestly left uncalibrated in
  the middle, and the accepted sequence is smoothed with a Savitzky-Golay
  filter. Every frame records how its homography was obtained (`h_source`:
  `solved` / `held` / `interp` / `none`);
- per-track OSNet means are clustered into two teams (k-means) and the
  anonymous clusters are oriented by pitch position;
- fragments are merged into identities by three rules, in order: same team
  plus a confidently read jersey number, physically feasible pitch motion, and
  OSNet cosine similarity of at least `0.90`. Guards keep two different known
  numbers out of one identity and refuse any merge that implies impossible
  movement;
- jersey numbers are committed from pooled OCR votes, teleporting ball points
  are dropped, player trajectories get a median filter plus Savitzky-Golay,
  and short detection gaps are interpolated.

Pass 2 walks the video again and renders the result: boxes with ids and
numbers, the minimap (frozen, then blanked, when calibration is missing), and
`gsr.json` with everything the pipeline knows.

Two similar-sounding thresholds are different quantities: BoT-SORT associates
detections at a maximum cosine distance of `0.4`, offline merging requires a
minimum cosine similarity of `0.90`.

## Stack

- Service: FastAPI, Uvicorn, SQLAlchemy + aiosqlite (job state in SQLite),
  pydantic-settings.
- Inference: ONNX Runtime (CUDA) for all six models, OpenCV for video and
  drawing, NumPy, SciPy (Savitzky-Golay and median filters), scikit-learn
  (team k-means).
- Vendored: trimmed runtime subsets of BoxMOT (BoT-SORT, AGPL-3.0) and
  PnLCalib (calibration, GPL-2.0); see the NOTICE files under `app/vendor/`.
- Tooling: uv with a locked Python 3.12 environment, pytest, ruff.

## Requirements

- Linux or WSL2, Python 3.12, `uv` 0.11.25+
- NVIDIA GPU with a CUDA 12 driver. ONNX Runtime can fall back to CPU, but
  the pipeline is sized for GPU (a 42 s clip takes about 7.5 minutes of GPU
  time), so treat the CUDA check below as required.
- FFmpeg for H.264 output; OpenCV MP4 is the fallback.

```bash
uv sync --locked --no-dev
cp env.example .env
```

The lock pins `onnxruntime-gpu` only: the CPU and GPU wheels share one Python
namespace and must not coexist in the same environment.

## Models

Six ONNX artifacts, all verified by SHA-256:

| Registry name | Model | Role |
|---|---|---|
| `deimv2_detector` | DEIMv2-DINOv3 M @ 896 | players, goalkeepers, referees, ball |
| `visibility_gate` | ShuffleNetV2 | is the jersey number readable |
| `jersey_ocr` | ConvNeXt-Tiny, two heads | jersey digits |
| `hrnet_kp`, `hrnet_lines` | PnLCalib HRNet | pitch keypoints and lines |
| `osnet_reid` | OSNet-x1.0 (SoccerNet) | re-id embeddings for teams and merging |

Download everything that has a configured source:

```bash
uv run --locked --no-dev python -m app.utils.models_registry
```

OSNet has no public download source yet. Place the pre-exported file at
`models/osnet_x1_0_soccernet.onnx`, or point the service at it with a process
environment variable (not `.env`):

```bash
export MODELS__OSNET_REID__PATH=/absolute/path/osnet_x1_0_soccernet.onnx
```

Required SHA-256:

```text
6d7a70bb28c309d91f970dbff190755a95bfed78089aa6a9831b1824914cb078
```

Then verify the full set. Startup is fail-fast: all six models are required,
there is no reduced-quality mode.

```bash
uv run --locked --no-dev python -m app.utils.models_registry --strict
```

## Run

```bash
uv run --locked --no-dev python -m app.utils.cuda_env
uv run --locked --no-dev uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

The first command must report `CUDAExecutionProvider`. Use exactly one worker:
the queue, job ownership, and loaded GPU sessions are process-local. Readiness
turns green only after the database, the models, and the job worker are up:

```bash
curl -fsS http://127.0.0.1:8000/health/live
curl -fsS http://127.0.0.1:8000/health/ready
```

OpenAPI docs: `http://127.0.0.1:8000/docs`.

## API

Submit a clip and poll until `status` is `done` (on `error`, read
`error_message`):

```bash
JOB_ID=$(curl -fsS -X POST http://127.0.0.1:8000/jobs \
  -F video=@clip.mp4 | python -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
watch -n 5 "curl -fsS http://127.0.0.1:8000/jobs/$JOB_ID | python -m json.tool"
```

Download the outputs, list jobs, delete a finished job and its artifacts:

```bash
curl -f "http://127.0.0.1:8000/jobs/$JOB_ID/video" -o annotated.mp4
curl -f "http://127.0.0.1:8000/jobs/$JOB_ID/gsr.json" -o gsr.json
curl -fsS "http://127.0.0.1:8000/jobs?status=done&limit=20"
curl -X DELETE "http://127.0.0.1:8000/jobs/$JOB_ID"
```

Behavior worth knowing:

- uploads stream to disk and are capped by `MAX_VIDEO_SIZE_MB`; the bounded
  queue answers `503` with `Retry-After` when full;
- input files are removed after processing; outputs stay until the job is
  deleted (no TTL, no disk quota);
- on restart, complete artifact pairs are finalized and interrupted jobs with
  an intact input are requeued;
- graceful shutdown waits for the active job, because a running inference
  thread cannot be cancelled safely.

In `gsr.json`, `cls_id` is the consolidated role and `raw_cls_id` the
detector's original one; interpolated observations carry `"synthetic": true`;
each frame's `h_source` says where its homography came from.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_AUTO_DOWNLOAD` | `true` | download missing artifacts that have a source |
| `DEIMV2_CONFIDENCE_THRESHOLD` | `0.4` | detector confidence threshold |
| `DATABASE_URL` | `<project>/soccer_gsr.db` | SQLite job database |
| `ARTIFACT_DIR` | `<project>/soccer_gsr_jobs` | inputs and outputs |
| `MAX_VIDEO_SIZE_MB` | `100` | upload size limit |
| `MAX_PENDING_JOBS` | `4` | pending jobs, excluding the active one |
| `DEBUG` | `false` | SQL logging and debug mode |

## Benchmark evaluation

Convert a service result into SoccerNetGS prediction format:

```python
from benchmark.predictions import export_prediction

export_prediction(
    state_path="artifacts/SNGS-116/gsr.json",
    labels_path="/data/SoccerNetGS/test/SNGS-116/Labels-GameState.json",
    output_path="predictions/SoccerNetGS-test/service/data/SNGS-116.json",
)
```

Then run the evaluator (it needs `sn-trackeval`, which is not part of the
runtime environment):

```bash
uv run --no-project --isolated --python 3.12 --with sn-trackeval==0.4.0 \
  python -m benchmark.evaluate \
  --gt-root /data/SoccerNetGS \
  --trackers-root predictions \
  --split test \
  --tag service \
  --sequences all \
  --output metrics-test.json
```

Setup, schema rules, and the result boundary: [docs/benchmark.md](docs/benchmark.md).

## Development

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked ruff check app benchmark tests
uv run --locked ruff format --check app benchmark tests
```

The test suite is CPU-only and isolates model backends; it covers tracking,
calibration, merging, jersey pooling, smoothing, upload and queue failure
paths, worker lifecycle, readiness, model integrity, and benchmark conversion.

```text
app/api/          job and artifact endpoints
app/models/       SQLite job state
app/services/     inference, aggregation, rendering, worker
app/utils/        CUDA bootstrap and model registry
app/vendor/       trimmed BoxMOT and PnLCalib runtime code
benchmark/        SoccerNetGS conversion and evaluation
docs/             benchmark evidence and evaluation boundary
tests/            CPU regression suite
```

## Licensing

Original service code is MIT. The vendored BoxMOT subset is AGPL-3.0, the
vendored PnLCalib subset is GPL-2.0; license texts, upstream links, file
scopes, and local modifications are recorded under `app/vendor/boxmot` and
`app/vendor/pnlcalib`. A combined distribution or hosted deployment must
satisfy the applicable copyleft obligations.
