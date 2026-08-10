import numpy as np

from app.vendor.boxmot.utils.matching import iou_distance


def joint_stracks(tlista: list, tlistb: list) -> list:
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista: list, tlistb: list) -> list:
    stracks = {t.id: t for t in tlista}
    for t in tlistb:
        tid = t.id
        if tid in stracks:
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa: list, stracksb: list) -> tuple[list, list]:
    """Of each overlapping pair, drop the track that has lived the shorter time."""
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []

    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)

    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]

    return resa, resb
