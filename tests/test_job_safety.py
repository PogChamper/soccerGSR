import asyncio
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException, UploadFile
from sqlalchemy.exc import SQLAlchemyError
from starlette.datastructures import Headers

from app.api import jobs, uploads
from app.services import job_worker


class _JobSession:
    def __init__(self, job: object) -> None:
        self._job = job
        self.deleted: object = None
        self.committed = False

    async def execute(self, _statement: object):
        return SimpleNamespace(scalar_one_or_none=lambda: self._job)

    async def delete(self, value: object) -> None:
        self.deleted = value

    async def commit(self) -> None:
        self.committed = True


class _CreateJobSession:
    def __init__(self, *, cancel_first_commit: bool = False) -> None:
        self._cancel_first_commit = cancel_first_commit
        self.deleted = False
        self.commits = 0

    def add(self, _job: object) -> None:
        pass

    async def commit(self) -> None:
        self.commits += 1
        if self._cancel_first_commit and self.commits == 1:
            raise asyncio.CancelledError

    async def rollback(self) -> None:
        pass

    async def execute(self, _statement: object) -> None:
        self.deleted = True


class _RecoverySession:
    def __init__(self, job: object) -> None:
        self._job = job
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: object) -> None:
        pass

    async def execute(self, _statement: object):
        rows = SimpleNamespace(all=lambda: [self._job])
        return SimpleNamespace(scalars=lambda: rows)

    async def commit(self) -> None:
        self.committed = True


def _inline_executor(loop: asyncio.AbstractEventLoop):
    def run_inline(_executor, function, *args):
        future = loop.create_future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    return run_inline


def test_output_video_path_is_server_owned(tmp_path: Path) -> None:
    job_id = str(uuid4())

    output = Path(job_worker.build_output_video_path(str(tmp_path), job_id))

    assert output == tmp_path / f"job_{job_id}_output.mp4"
    assert output.resolve().parent == tmp_path.resolve()


def test_api_timestamp_restores_sqlite_utc() -> None:
    value = jobs._api_timestamp(datetime(2026, 8, 8, 12, 0, 0))

    assert value is not None
    assert value.tzinfo is UTC


@pytest.mark.asyncio
async def test_get_job_includes_self_link() -> None:
    job_id = str(uuid4())
    job = SimpleNamespace(
        job_id=job_id,
        status="queued",
        stage=None,
        progress=None,
        created_at=datetime(2026, 8, 8, 12, 0, 0),
        started_at=None,
        finished_at=None,
        input_filename="clip.mp4",
        error_message=None,
    )

    response = await jobs.get_job(job_id, db=_JobSession(job))  # type: ignore[arg-type]

    assert response["links"]["self"] == f"/jobs/{job_id}"


@pytest.mark.parametrize("job_id", ["../escape", "../../tmp/video", "/tmp/video"])
def test_output_video_path_rejects_non_uuid_job_ids(tmp_path: Path, job_id: str) -> None:
    with pytest.raises(ValueError):
        job_worker.build_output_video_path(str(tmp_path), job_id)


def test_validate_video_rejects_known_oversized_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uploads.settings, "max_video_size_mb", 1)
    upload = UploadFile(
        BytesIO(b""),
        size=1024 * 1024 + 1,
        filename="clip.mp4",
        headers=Headers({"content-type": "video/mp4"}),
    )

    with pytest.raises(HTTPException) as exc_info:
        uploads.validate_video(upload)

    assert exc_info.value.status_code == 413


@pytest.mark.parametrize(
    ("filename", "expected"),
    [("../../clip.mp4", "clip.mp4"), (r"C:\\videos\\clip.mov", "clip.mov")],
)
def test_upload_name_keeps_only_client_basename(filename: str, expected: str) -> None:
    upload = UploadFile(BytesIO(b""), filename=filename)

    assert uploads.upload_name(upload) == expected


@pytest.mark.asyncio
async def test_enqueue_job_is_bounded_and_nonblocking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(job_worker.settings, "max_pending_jobs", 1)
    monkeypatch.setattr(job_worker, "_QUEUE", None)

    await job_worker.enqueue_job("first")

    assert job_worker.get_queue().maxsize == 1
    with pytest.raises(asyncio.QueueFull):
        await job_worker.enqueue_job("second")


@pytest.mark.asyncio
async def test_stop_worker_finishes_active_executor_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def process(_job_id: str) -> None:
        started.set()
        await release.wait()

    monkeypatch.setattr(job_worker, "_process_job", process)
    monkeypatch.setattr(job_worker, "_QUEUE", asyncio.Queue(maxsize=2))
    monkeypatch.setattr(job_worker, "_STOPPING", False)
    monkeypatch.setattr(job_worker, "_JOB_RUNNING", False)
    worker = asyncio.create_task(job_worker._worker_loop())
    monkeypatch.setattr(job_worker, "_WORKER_TASK", worker)
    await job_worker.enqueue_job("active")
    await started.wait()

    stopping = asyncio.create_task(job_worker.stop_worker())
    await asyncio.sleep(0)
    assert not stopping.done()

    release.set()
    await stopping
    assert job_worker._WORKER_TASK is None
    assert job_worker._QUEUE is None


@pytest.mark.asyncio
async def test_stop_worker_survives_a_persistence_failure_mid_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing() -> None:
        raise job_worker.JobPersistenceError("database gone")

    worker = asyncio.create_task(failing())
    await asyncio.sleep(0)
    monkeypatch.setattr(job_worker, "_WORKER_TASK", worker)
    monkeypatch.setattr(job_worker, "_QUEUE", asyncio.Queue())
    monkeypatch.setattr(job_worker, "_JOB_RUNNING", True)

    await job_worker.stop_worker()

    assert job_worker._WORKER_TASK is None
    assert job_worker._QUEUE is None


@pytest.mark.asyncio
async def test_create_job_queue_full_removes_job_and_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")

    async def fake_save_upload(_video: object, _job_id: str) -> str:
        return str(input_path)

    async def full_queue(_job_id: str) -> None:
        raise asyncio.QueueFull

    session = _CreateJobSession()
    video = UploadFile(BytesIO(b""), filename="clip.mp4")
    monkeypatch.setattr(jobs, "save_upload", fake_save_upload)
    monkeypatch.setattr(jobs, "enqueue_job", full_queue)

    with pytest.raises(HTTPException) as exc_info:
        await jobs.create_job(video=video, db=session)  # type: ignore[arg-type]

    assert exc_info.value.status_code == 503
    assert exc_info.value.headers == {"Retry-After": "5"}
    assert session.deleted
    assert session.commits == 2
    assert not input_path.exists()


@pytest.mark.asyncio
async def test_create_job_cancellation_removes_job_and_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")

    async def fake_save_upload(_video: object, _job_id: str) -> str:
        return str(input_path)

    session = _CreateJobSession(cancel_first_commit=True)
    video = UploadFile(BytesIO(b""), filename="clip.mp4")
    monkeypatch.setattr(jobs, "save_upload", fake_save_upload)

    with pytest.raises(asyncio.CancelledError):
        await jobs.create_job(video=video, db=session)  # type: ignore[arg-type]

    assert session.deleted
    assert session.commits == 2
    assert not input_path.exists()


@pytest.mark.asyncio
async def test_delete_finished_job_removes_all_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    json_path = tmp_path / "state.json"
    job_id = str(uuid4())
    partial_path = tmp_path / f"job_{job_id}_output.mp4"
    for path in (input_path, output_path, json_path, partial_path):
        path.write_bytes(b"data")

    job = SimpleNamespace(
        status="done",
        input_path=str(input_path),
        output_video_path=str(output_path),
    )

    session = _JobSession(job)
    monkeypatch.setattr(jobs.settings, "artifact_dir", str(tmp_path))
    monkeypatch.setattr(jobs, "gsr_json_path", lambda _job_id: str(json_path))

    response = await jobs.delete_job(job_id, db=session)  # type: ignore[arg-type]

    assert response.status_code == 204
    assert session.deleted is job
    assert session.committed
    assert not any(path.exists() for path in (input_path, output_path, json_path, partial_path))


@pytest.mark.asyncio
async def test_completed_artifacts_survive_terminal_database_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_id = str(uuid4())
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    job = SimpleNamespace(
        job_id=job_id,
        status="queued",
        input_path=str(input_path),
        input_filename="clip.mp4",
    )
    output_path = Path(job_worker.build_output_video_path(str(tmp_path), job_id))
    json_path = tmp_path / f"job_{job_id}_gsr.json"

    async def load_job(_job_id: str):
        return job

    updates = 0

    async def update_job(_job_id: str, **fields) -> None:
        nonlocal updates
        updates += 1
        if fields["status"] == "done":
            raise SQLAlchemyError("database unavailable")

    def process(_job_id, _input_path, requested_output, _loop, source_filename):
        assert source_filename == "clip.mp4"
        Path(requested_output).write_bytes(b"video")
        json_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(job_worker.settings, "artifact_dir", str(tmp_path))
    monkeypatch.setattr(job_worker, "_load_job", load_job)
    monkeypatch.setattr(job_worker, "_update_job", update_job)
    monkeypatch.setattr(job_worker, "_run_processing_blocking", process)
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "run_in_executor", _inline_executor(loop))

    with pytest.raises(job_worker.JobPersistenceError):
        await job_worker._process_job(job_id)

    assert updates == 2
    assert output_path.read_bytes() == b"video"
    assert json_path.read_text(encoding="utf-8") == "{}"
    assert not input_path.exists()


@pytest.mark.asyncio
async def test_failed_job_keeps_input_when_terminal_database_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_id = str(uuid4())
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    job = SimpleNamespace(
        job_id=job_id,
        status="queued",
        input_path=str(input_path),
        input_filename="clip.mp4",
    )
    output_path = Path(job_worker.build_output_video_path(str(tmp_path), job_id))
    json_path = tmp_path / f"job_{job_id}_gsr.json"

    async def load_job(_job_id: str):
        return job

    async def update_job(_job_id: str, **fields) -> None:
        if fields["status"] == "error":
            raise SQLAlchemyError("database unavailable")

    def process(_job_id, _input_path, requested_output, _loop, source_filename):
        assert source_filename == "clip.mp4"
        Path(requested_output).write_bytes(b"partial video")
        json_path.write_text("{}", encoding="utf-8")
        raise RuntimeError("inference failed")

    monkeypatch.setattr(job_worker.settings, "artifact_dir", str(tmp_path))
    monkeypatch.setattr(job_worker, "_load_job", load_job)
    monkeypatch.setattr(job_worker, "_update_job", update_job)
    monkeypatch.setattr(job_worker, "_run_processing_blocking", process)
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "run_in_executor", _inline_executor(loop))

    with pytest.raises(job_worker.JobPersistenceError):
        await job_worker._process_job(job_id)

    assert input_path.read_bytes() == b"video"
    assert not output_path.exists()
    assert not json_path.exists()


@pytest.mark.asyncio
async def test_recovery_commits_complete_running_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_id = str(uuid4())
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    output_path = Path(job_worker.build_output_video_path(str(tmp_path), job_id))
    output_path.write_bytes(b"video")
    json_path = tmp_path / f"job_{job_id}_gsr.json"
    json_path.write_text("{}", encoding="utf-8")
    job = SimpleNamespace(
        job_id=job_id,
        status="running",
        input_path=str(input_path),
        stage="pass2",
        progress=90.0,
        output_video_path=None,
        error_message=None,
        finished_at=None,
    )

    session = _RecoverySession(job)
    monkeypatch.setattr(job_worker.settings, "artifact_dir", str(tmp_path))
    monkeypatch.setattr(job_worker, "async_session", lambda: session)

    await job_worker._recover_stale_jobs()

    assert session.committed
    assert job.status == "done"
    assert job.stage == "done"
    assert job.progress == 100.0
    assert job.output_video_path == str(output_path)
    assert output_path.exists() and json_path.exists()
    assert not input_path.exists()


@pytest.mark.asyncio
async def test_recovery_requeues_incomplete_running_job(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    job_id = str(uuid4())
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    output_path = Path(job_worker.build_output_video_path(str(tmp_path), job_id))
    output_path.write_bytes(b"partial video")
    job = SimpleNamespace(
        job_id=job_id,
        status="running",
        input_path=str(input_path),
        stage="pass1",
        progress=25.0,
        started_at=datetime(2026, 8, 9),
        finished_at=None,
        output_video_path=None,
        error_message="stale",
    )

    queued: list[str] = []

    async def enqueue(job_id_to_enqueue: str) -> None:
        queued.append(job_id_to_enqueue)

    session = _RecoverySession(job)
    monkeypatch.setattr(job_worker.settings, "artifact_dir", str(tmp_path))
    monkeypatch.setattr(job_worker, "async_session", lambda: session)
    monkeypatch.setattr(job_worker, "enqueue_job", enqueue)

    await job_worker._recover_stale_jobs()

    assert session.committed
    assert queued == [job_id]
    assert job.status == "queued"
    assert job.stage is None
    assert job.progress is None
    assert job.started_at is None
    assert job.finished_at is None
    assert job.error_message is None
    assert input_path.exists()
    assert not output_path.exists()
