from fastapi.testclient import TestClient

from api import server
from inference.model import create_model

client = TestClient(server.app)


def test_model_is_deterministic():
    model_a = create_model()
    model_b = create_model()

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert state_a[key].equal(state_b[key])


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["backend"] == "pytorch"
    assert data["device"] == "cpu"
    assert data["cuda_available"] is False
    assert data["model_version"] == server.MODEL_VERSION
    assert data["input_dimension"] == 1024
    assert data["maximum_batch_size"] == 256


def test_infer_endpoint():
    response = client.post(
        "/infer",
        json={"inputs": [[0.0] * 1024]},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["backend"] == "pytorch"
    assert data["device"] == "cpu"
    assert data["batch_size"] == 1
    assert data["model_version"] == server.MODEL_VERSION
    assert len(data["outputs"]) == 1
    assert len(data["outputs"][0]) == 1000


def test_empty_batch_is_rejected():
    response = client.post("/infer", json={"inputs": []})
    assert response.status_code == 422


def test_oversized_batch_is_rejected():
    response = client.post(
        "/infer",
        json={
            "inputs": [
                [0.0] * 1024
                for _ in range(server.MAX_BATCH_SIZE + 1)
            ]
        },
    )
    assert response.status_code == 422


def test_wrong_input_dimension_is_rejected():
    response = client.post(
        "/infer",
        json={"inputs": [[0.0] * 10]},
    )
    assert response.status_code == 422
    assert "1024 values" in response.json()["detail"]


def test_backend_exception_is_not_exposed(monkeypatch):
    def fail(_inputs):
        raise RuntimeError("sensitive internal engine path")

    monkeypatch.setattr(server.backend, "infer", fail)
    response = client.post(
        "/infer",
        json={"inputs": [[0.0] * 1024]},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Inference backend failed."
    assert "sensitive" not in response.text


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "accelserve_requests_total" in response.text
    assert "accelserve_inference_latency_seconds" in response.text
