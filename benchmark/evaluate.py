"""Evaluate SoccerNet Game State predictions with the official TrackEval fork.

trackeval is imported inside evaluate(), so discovery and validation work in
environments without the benchmark stack installed.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DATASET_NAME = "SoccerNetGS"
LABELS_FILENAME = "Labels-GameState.json"
METRIC_FIELDS = ("HOTA", "DetA", "AssA", "LocA")


def discover_sequences(gt_root: str | Path, split: str) -> tuple[str, ...]:
    """Return the sorted split subdirectories that contain ``Labels-GameState.json``."""

    split_root = Path(gt_root) / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Ground-truth split directory not found: {split_root}")

    sequences = tuple(
        sorted(
            child.name
            for child in split_root.iterdir()
            if child.is_dir() and (child / LABELS_FILENAME).is_file()
        )
    )
    if not sequences:
        raise ValueError(f"No {LABELS_FILENAME} files found under: {split_root}")
    return sequences


def resolve_sequences(
    gt_root: str | Path,
    split: str,
    selection: str | Sequence[str] = "all",
) -> tuple[str, ...]:
    """Resolve ``all`` or an explicit selection against ground truth, keeping input order."""

    if isinstance(selection, str):
        if selection.strip() == "all":
            return discover_sequences(gt_root, split)
        sequences = tuple(part.strip() for part in selection.split(","))
    else:
        sequences = tuple(str(part).strip() for part in selection)

    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("Sequence selection must be 'all' or a comma-separated list")
    if len(set(sequences)) != len(sequences):
        raise ValueError("Sequence selection contains duplicate names")

    split_root = Path(gt_root) / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Ground-truth split directory not found: {split_root}")
    for sequence in sequences:
        if Path(sequence).name != sequence:
            raise ValueError(f"Invalid sequence name: {sequence!r}")
        labels_path = split_root / sequence / LABELS_FILENAME
        if not labels_path.is_file():
            raise FileNotFoundError(f"Ground-truth labels not found: {labels_path}")

    return sequences


def prediction_path(
    trackers_root: str | Path,
    split: str,
    tag: str,
    sequence: str,
) -> Path:
    return Path(trackers_root) / f"{DATASET_NAME}-{split}" / tag / "data" / f"{sequence}.json"


def validate_predictions(
    trackers_root: str | Path,
    split: str,
    tag: str,
    sequences: Sequence[str],
) -> None:
    """Fail on the first selected sequence without a prediction file."""

    for sequence in sequences:
        path = prediction_path(trackers_root, split, tag, sequence)
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {path}")


def _mean_percent(values: Any) -> float:
    """Convert a TrackEval scalar or one-dimensional metric array to percent."""

    try:
        iterator = iter(values)
    except TypeError:
        mean = float(values)
    else:
        numbers = [float(value) for value in iterator]
        if not numbers:
            raise ValueError("TrackEval returned an empty metric array")
        mean = sum(numbers) / len(numbers)
    return round(100.0 * mean, 6)


def evaluate(
    *,
    gt_root: str | Path,
    trackers_root: str | Path,
    split: str,
    tag: str,
    sequences: str | Sequence[str] = "all",
) -> dict[str, Any]:
    """Run GS-HOTA evaluation and return a compact, JSON-serializable result.

    ``trackeval`` must be the SoccerNet fork exposing the ``SoccerNetGS``
    dataset. Scores are reported as percentages, matching TrackEval's tables.
    """

    selected = resolve_sequences(gt_root, split, sequences)
    validate_predictions(trackers_root, split, tag, selected)

    try:
        import trackeval
    except ImportError as exc:
        raise RuntimeError(
            "The SoccerNet TrackEval fork is required to run evaluation. "
            "Install it in the benchmark environment before invoking this command."
        ) from exc

    evaluator_config = {
        "USE_PARALLEL": False,
        "BREAK_ON_ERROR": True,
        "PRINT_RESULTS": False,
        "PRINT_CONFIG": False,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
        "TIME_PROGRESS": False,
    }
    dataset_class = trackeval.datasets.SoccerNetGS
    dataset_config = dataset_class.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(Path(gt_root)),
            "TRACKERS_FOLDER": str(Path(trackers_root)),
            "SPLIT_TO_EVAL": split,
            "TRACKERS_TO_EVAL": [tag],
            # SoccerNetGS uses the keys to select sequences and derives frame
            # information directly from Labels-GameState.json.
            "SEQ_INFO": dict.fromkeys(selected),
            "PRINT_CONFIG": False,
        }
    )

    evaluator = trackeval.Evaluator(evaluator_config)
    raw_results, _ = evaluator.evaluate([dataset_class(dataset_config)], [trackeval.metrics.HOTA()])
    hota = raw_results[DATASET_NAME][tag]["COMBINED_SEQ"]["person"]["HOTA"]
    metrics = {
        "GS-HOTA" if field == "HOTA" else field: _mean_percent(hota[field])
        for field in METRIC_FIELDS
    }
    return {
        "split": split,
        "tag": tag,
        "sequences": list(selected),
        "metrics": metrics,
    }


def write_result(result: dict[str, Any], output: str | Path) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate SoccerNet Game State JSON predictions with GS-HOTA.",
    )
    parser.add_argument(
        "--gt-root",
        required=True,
        type=Path,
        help="dataset root containing <split>/<sequence>/Labels-GameState.json",
    )
    parser.add_argument(
        "--trackers-root",
        required=True,
        type=Path,
        help="root containing SoccerNetGS-<split>/<tag>/data",
    )
    parser.add_argument(
        "--split", required=True, help="dataset split to evaluate (for example, test)"
    )
    parser.add_argument("--tag", required=True, help="tracker tag below SoccerNetGS-<split>")
    parser.add_argument(
        "--sequences",
        default="all",
        metavar="all|SEQ,SEQ,...",
        help="all labelled GT sequences or an explicit comma-separated selection (default: all)",
    )
    parser.add_argument("--output", required=True, type=Path, help="destination metrics JSON file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate(
        gt_root=args.gt_root,
        trackers_root=args.trackers_root,
        split=args.split,
        tag=args.tag,
        sequences=args.sequences,
    )
    write_result(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
