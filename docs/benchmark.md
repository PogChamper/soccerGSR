# Benchmark evaluation

The benchmark path converts the service `gsr.json` output into the SoccerNetGS
prediction schema, then evaluates those predictions with the SoccerNet TrackEval
fork. Export is a library API; the repository does not include a separate
conversion script.

## Requirements

Evaluation requires:

- one service `gsr.json` for every selected sequence;
- the matching SoccerNetGS `Labels-GameState.json` files;
- `sn-trackeval==0.4.0`, exposing `trackeval.datasets.SoccerNetGS`.

Keep the evaluator outside the service environment because its upstream package
depends on the non-headless OpenCV distribution. `uv` creates an isolated
environment for the command:

```bash
uv run --no-project --isolated --python 3.12 --with sn-trackeval==0.4.0 \
  python -c 'import trackeval; print(trackeval.__file__)'
```

The label file supplies frame identifiers, image dimensions, and category IDs.
The exporter does not read ground-truth annotations or use them to alter service
predictions.

## Export predictions

Export each sequence with `benchmark.predictions.export_prediction`:

```python
from benchmark.predictions import export_prediction

export_prediction(
    state_path="artifacts/SNGS-116/gsr.json",
    labels_path="/data/SoccerNetGS/test/SNGS-116/Labels-GameState.json",
    output_path="predictions/SoccerNetGS-test/service/data/SNGS-116.json",
)
```

Repeat the call for every sequence included in the evaluation. TrackEval expects
this directory layout:

```text
predictions/
  SoccerNetGS-test/
    service/
      data/
        SNGS-116.json
        SNGS-117.json
        ...
```

The exporter fails on mismatched frame counts or image dimensions, missing track
references, unsupported classes, non-contiguous frame metadata, and duplicate
track IDs within a frame. Tracked observations without pitch coordinates are
omitted as localization abstentions. Anonymous team clusters are exported as
`null`; only spatially oriented teams become `left` or `right`.

## Evaluate

Evaluate one sequence first:

```bash
uv run --no-project --isolated --python 3.12 --with sn-trackeval==0.4.0 \
  python -m benchmark.evaluate \
  --gt-root /data/SoccerNetGS \
  --trackers-root predictions \
  --split test \
  --tag service \
  --sequences SNGS-116 \
  --output metrics-SNGS-116.json
```

After every prediction file is present, evaluate the complete split:

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

Sequence discovery comes from ground truth. The evaluator rejects missing
selected predictions before importing TrackEval and ignores stale prediction
files for sequences outside the selection. Reported values are percentages.

## Results

Both tables below were reproduced from saved prediction files. Neither is a
fresh video-to-predictions run of the current service.

49 SoccerNetGS v1.3 public-test sequences, 36,750 frames:

| Prediction set | GS-HOTA | DetA | AssA | LocA |
|---|---:|---:|---:|---:|
| Cached research pipeline | **55.682** | 42.144 | 73.586 | 92.538 |
| Locally bundled upstream example | 23.087 | 11.114 | 48.013 | 90.506 |

This repository does not distribute the research predictions, so the first row
is not reproducible from a clean clone. The service now has its own export and
evaluation path, but no full public-test run has gone through it yet.

`SNGS-116` through `SNGS-120`, with the current team assignment, merging, jersey
aggregation, interpolation, and exporter run over saved detector, OSNet,
tracker, and calibration outputs:

| Five-sequence assembly | GS-HOTA | DetA | AssA | LocA |
|---|---:|---:|---:|---:|
| Current service assembly | 55.777 | 40.346 | 77.124 | 93.040 |
| Saved research assembly | 55.406 | 40.081 | 76.604 | 93.049 |

Both rows start from the same cached model outputs, so this compares
post-processing and export only. It covers five sequences and does not exercise
live ball tracking.

The example row comes from
[`SoccerNetGS-test.zip`](https://github.com/SoccerNet/sn-gamestate/blob/057dd144a982e00576f8ffb45bdd00c0f614c549/examples_predictions/SoccerNetGS-test.zip)
at `sn-gamestate` revision `057dd144a982e00576f8ffb45bdd00c0f614c549`; it is not
the official 2025 challenge baseline.
