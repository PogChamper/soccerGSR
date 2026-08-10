import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

import app.main as main


@pytest.fixture
def health_app() -> FastAPI:
    application = FastAPI()
    application.add_api_route("/health/live", main.health_live)
    application.add_api_route("/health/ready", main.health_ready)
    application.state.ready = False
    application.state.startup_error = None
    return application


async def request(application: FastAPI, path: str) -> Response:
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


@pytest.fixture(autouse=True)
def running_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "worker_is_running", lambda: True)


@pytest.mark.asyncio
async def test_liveness_is_independent_of_readiness(health_app: FastAPI) -> None:
    health_app.state.startup_error = "models unavailable"

    response = await request(health_app, "/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


@pytest.mark.asyncio
async def test_readiness_returns_503_with_startup_error(health_app: FastAPI) -> None:
    health_app.state.startup_error = "RuntimeError: detector unavailable"

    response = await request(health_app, "/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "ready": False,
        "startup_error": "RuntimeError: detector unavailable",
    }


@pytest.mark.asyncio
async def test_readiness_returns_200_when_startup_completed(health_app: FastAPI) -> None:
    health_app.state.ready = True

    response = await request(health_app, "/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "ready": True,
        "startup_error": None,
    }


@pytest.mark.asyncio
async def test_startup_error_overrides_stale_ready_flag(health_app: FastAPI) -> None:
    health_app.state.ready = True
    health_app.state.startup_error = "RuntimeError: worker failed"

    response = await request(health_app, "/health/ready")

    assert response.status_code == 503
    assert response.json()["ready"] is False


@pytest.mark.asyncio
async def test_readiness_detects_stopped_worker(
    health_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    health_app.state.ready = True
    monkeypatch.setattr(main, "worker_is_running", lambda: False)

    response = await request(health_app, "/health/ready")

    assert response.status_code == 503
    assert response.json()["ready"] is False


@pytest.mark.asyncio
async def test_readiness_does_not_probe_models(
    health_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called():
        pytest.fail("readiness request must not load the detector")

    monkeypatch.setattr("app.services.detector.get_detector", fail_if_called)

    response = await request(health_app, "/health/ready")

    assert response.status_code == 503


def test_lifespan_marks_ready_only_after_worker_and_clears_it_on_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    application = FastAPI()
    events = []

    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            app_name="test",
            secret_key="test-secret",
            artifact_dir=str(tmp_path),
            detector_backend="test",
        ),
    )
    monkeypatch.setattr(main, "cuda_bootstrap", lambda: events.append("cuda") or False)
    monkeypatch.setattr("app.services.detector.get_detector", lambda: events.append("detector"))
    monkeypatch.setattr(
        "app.services.jersey.get_jersey_recognizer", lambda: events.append("jersey")
    )
    monkeypatch.setattr("app.services.embedder.get_embedder", lambda: events.append("embedder"))
    monkeypatch.setattr(
        "app.services.keypoints.get_keypoints_extractor", lambda: events.append("keypoints")
    )

    async def init_db() -> None:
        events.append("database")

    async def start_worker() -> None:
        assert application.state.ready is False
        events.append("worker_started")

    async def stop_worker() -> None:
        assert application.state.ready is False
        events.append("worker_stopped")

    async def close_db() -> None:
        events.append("database_closed")

    monkeypatch.setattr(main, "init_db", init_db)
    monkeypatch.setattr(main, "start_worker", start_worker)
    monkeypatch.setattr(main, "stop_worker", stop_worker)
    monkeypatch.setattr(main, "close_db", close_db)

    async def exercise_lifespan() -> None:
        async with main.lifespan(application):
            assert application.state.ready is True
            assert application.state.startup_error is None

    asyncio.run(exercise_lifespan())

    assert application.state.ready is False
    assert events == [
        "cuda",
        "database",
        "detector",
        "jersey",
        "embedder",
        "keypoints",
        "worker_started",
        "worker_stopped",
        "database_closed",
    ]


@pytest.mark.asyncio
async def test_required_model_failure_aborts_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    application = FastAPI()
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            app_name="test",
            secret_key="test-secret",
            artifact_dir=str(tmp_path),
        ),
    )
    monkeypatch.setattr(main, "cuda_bootstrap", lambda: False)

    async def init_db() -> None:
        return None

    def fail_detector_startup() -> None:
        raise RuntimeError("detector unavailable")

    async def close_db() -> None:
        return None

    monkeypatch.setattr(main, "init_db", init_db)
    monkeypatch.setattr(main, "close_db", close_db)
    monkeypatch.setattr("app.services.detector.get_detector", fail_detector_startup)

    with pytest.raises(RuntimeError, match="detector unavailable"):
        async with main.lifespan(application):
            pytest.fail("a required model failure must abort startup")

    assert application.state.ready is False
    assert application.state.startup_error == "RuntimeError: detector unavailable"


@pytest.mark.asyncio
async def test_osnet_failure_aborts_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    application = FastAPI()
    monkeypatch.setattr(
        main,
        "settings",
        SimpleNamespace(
            app_name="test",
            artifact_dir=str(tmp_path),
            detector_backend="test",
        ),
    )
    monkeypatch.setattr(main, "cuda_bootstrap", lambda: False)
    monkeypatch.setattr("app.services.detector.get_detector", lambda: None)
    monkeypatch.setattr("app.services.jersey.get_jersey_recognizer", lambda: None)
    monkeypatch.setattr(
        "app.services.embedder.get_embedder",
        lambda: (_ for _ in ()).throw(RuntimeError("OSNet unavailable")),
    )

    async def init_db() -> None:
        return None

    async def close_db() -> None:
        return None

    monkeypatch.setattr(main, "init_db", init_db)
    monkeypatch.setattr(main, "close_db", close_db)

    with pytest.raises(RuntimeError, match="OSNet unavailable"):
        async with main.lifespan(application):
            pytest.fail("a required OSNet failure must abort startup")

    assert application.state.ready is False
    assert application.state.startup_error == "RuntimeError: OSNet unavailable"
