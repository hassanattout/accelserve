from fastapi.testclient import TestClient

from api.server import app
from inference.model import create_model


def test_model_is_deterministic():
    model_a = create_model()
    model_b = create_model()

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    assert state_a.keys() == state_b.keys()

    for key in state_a:
        assert state_a[key].equal(state_b[key])


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["backend"] == "pytorch"
    assert data["device"] == "cpu"
    assert data["cuda_available"] is False


def test_infer_endpoint():
    client = TestClient(app)

    payload = {
        "inputs": [
            [0.0] * 1024
        ]
    }

    response = client.post(
        "/infer",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["backend"] == "pytorch"
    assert data["device"] == "cpu"
    assert data["batch_size"] == 1
    assert len(data["outputs"]) == 1
    assert len(data["outputs"][0]) == 1000

def test_metrics_endpoint():
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "accelserve_requests_total" in response.text
    assert "accelserve_inference_latency_seconds" in response.text
