import numpy as np

from app.services.detector import Detection
from app.services.tracker import BoxmotTracker


class FakeTracker:
    def __init__(self) -> None:
        self.calls = []

    def update(self, detections, frame, *, embs=None):
        self.calls.append((detections, frame, embs))
        return np.empty((0, 8), dtype=np.float32)


def _wrapper(human: FakeTracker, ball: FakeTracker) -> BoxmotTracker:
    tracker = BoxmotTracker.__new__(BoxmotTracker)
    tracker._human_impl = human
    tracker._ball_impl = ball
    tracker._track_ids = {}
    tracker._next_track_id = 1
    return tracker


def test_empty_detections_advance_both_trackers() -> None:
    human = FakeTracker()
    ball = FakeTracker()
    tracker = _wrapper(human, ball)
    frame = np.zeros((8, 12, 3), dtype=np.uint8)

    result = tracker.update([], frame)

    assert result == []
    for implementation in (human, ball):
        assert len(implementation.calls) == 1
        detections, received_frame, embeddings = implementation.calls[0]
        assert detections.shape == (0, 6)
        assert detections.dtype == np.float32
        assert received_frame is frame
        assert embeddings.shape == (0, 512)
        assert embeddings.dtype == np.float32


def test_nonempty_detections_require_embeddings() -> None:
    human = FakeTracker()
    ball = FakeTracker()
    tracker = _wrapper(human, ball)
    detection = Detection((1.0, 1.0, 4.0, 7.0), 0, "player", 0.9)

    with np.testing.assert_raises_regex(ValueError, "OSNet embeddings are required"):
        tracker.update([detection], np.zeros((8, 12, 3), dtype=np.uint8))

    assert human.calls == []
    assert ball.calls == []


def test_tracker_groups_human_roles_and_isolates_ball() -> None:
    human = FakeTracker()
    ball = FakeTracker()
    tracker = _wrapper(human, ball)
    detections = [Detection((1.0, 1.0, 4.0, 7.0), class_id, "object", 0.9) for class_id in range(4)]

    tracker.update(
        detections,
        np.zeros((8, 12, 3), dtype=np.uint8),
        np.ones((4, 512), dtype=np.float32),
    )

    np.testing.assert_array_equal(human.calls[0][0][:, 5], [0, 0, 0])
    np.testing.assert_array_equal(ball.calls[0][0][:, 5], [1])


def test_tracker_remaps_local_detection_indices_and_track_ids() -> None:
    class OutputTracker(FakeTracker):
        def __init__(self, internal_id: int) -> None:
            super().__init__()
            self.internal_id = internal_id

        def update(self, detections, frame, *, embs=None):
            super().update(detections, frame, embs=embs)
            return np.asarray(
                [
                    [*detection[:4], self.internal_id, detection[4], detection[5], index]
                    for index, detection in enumerate(detections)
                ],
                dtype=np.float32,
            ).reshape(-1, 8)

    tracker = _wrapper(OutputTracker(7), OutputTracker(7))
    detections = [
        Detection((1.0, 1.0, 4.0, 7.0), 0, "player", 0.9),
        Detection((2.0, 2.0, 3.0, 3.0), 3, "ball", 0.9),
        Detection((5.0, 1.0, 8.0, 7.0), 2, "referee", 0.9),
    ]

    result = tracker.update(
        detections,
        np.zeros((8, 12, 3), dtype=np.uint8),
        np.ones((3, 512), dtype=np.float32),
    )

    assert [track_id for _, track_id in result] == [1, 2, 1]


def test_ball_cannot_reactivate_a_lost_human_track() -> None:
    tracker = BoxmotTracker()
    y, x = np.indices((96, 128))
    gray = (((x // 8) + (y // 8)) % 2 * 255).astype(np.uint8)
    frame = np.repeat(gray[..., None], 3, axis=2)
    embedding = np.ones((1, 512), dtype=np.float32)
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    human = Detection((20.0, 10.0, 40.0, 70.0), 0, "player", 0.95)
    ball = Detection((20.0, 10.0, 40.0, 70.0), 3, "ball", 0.95)

    human_id = tracker.update([human], frame, embedding)[0][1]
    tracker.update([ball], frame, embedding)
    ball_id = tracker.update([ball], frame, embedding)[0][1]
    recovered_human_id = tracker.update([human], frame, embedding)[0][1]

    assert human_id is not None
    assert ball_id is not None
    assert ball_id != human_id
    assert recovered_human_id == human_id
