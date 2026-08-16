"""Shared helpers for the SoccerNet GSR 2025 pipeline (GS-HOTA target).

The pipeline runs in three compute passes across two conda envs plus a final
eval in the sn-gamestate venv; every stage keys its intermediates by sequence
name and by (frame index, detection index) so downstream passes can align
without matching boxes. Frame order is the GT image order (image_id ascending).
"""
import json
import pickle
from pathlib import Path

DATA_ROOT = Path("/mnt/d/datasets/soccernet2025")
OUT_ROOT = Path("/home/dxdxxd/projects/soccer-app/gsr/out")

# detector class_id (onnx label - 1) -> GT category_id and role
CLS_TO_CAT = {0: 1, 1: 2, 2: 3, 3: 4}
CLS_TO_ROLE = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
CAT_TO_ROLE = {1: "player", 2: "goalkeeper", 3: "referee", 4: "ball"}


def seq_dir(split, seq):
    return DATA_ROOT / split / seq


def list_seqs(split):
    return sorted(p.name for p in (DATA_ROOT / split).iterdir() if p.is_dir())


def load_labels(split, seq):
    return json.load(open(seq_dir(split, seq) / "Labels-GameState.json"))


def frame_index(labels):
    """Return (image_ids, file_names) in GT frame order (image_id ascending)."""
    imgs = sorted(labels["images"], key=lambda im: im["image_id"])
    return [im["image_id"] for im in imgs], [im["file_name"] for im in imgs]


def dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
