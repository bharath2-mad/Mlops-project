import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_home_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Fitness" in res.data


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.get_json() == {"status": "ok"}


@pytest.mark.parametrize(
    "sample, expected",
    [
        ({"heart_rate": 64, "accel_mean": 1.0, "accel_std": 0.02, "cadence": 0}, "resting"),
        ({"heart_rate": 95, "accel_mean": 1.1, "accel_std": 0.3, "cadence": 105}, "walking"),
        ({"heart_rate": 155, "accel_mean": 1.8, "accel_std": 0.9, "cadence": 170}, "running"),
        ({"heart_rate": 130, "accel_mean": 1.2, "accel_std": 0.45, "cadence": 80}, "cycling"),
    ],
)
def test_predict_known_activities(client, sample, expected):
    res = client.post("/predict", json=sample)
    assert res.status_code == 200
    body = res.get_json()
    assert body["activity"] == expected
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=0.05)


def test_predict_missing_field(client):
    res = client.post("/predict", json={"heart_rate": 100})
    assert res.status_code == 400
    assert "Missing" in res.get_json()["error"]


def test_predict_non_numeric(client):
    res = client.post(
        "/predict",
        json={"heart_rate": "fast", "accel_mean": 1, "accel_std": 1, "cadence": 1},
    )
    assert res.status_code == 400
