# -*- coding: utf-8 -*-
from typing import Dict

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def init_test_client(monkeypatch) -> TestClient:
    def mock_make_inference(*args, **kwargs) -> dict[str, str]:
        return {"species": "Adelie"}

    def mock_load_model(*args, **kwargs) -> None:
        return None

    monkeypatch.setenv("MODEL_PATH", "faked/model.pkl")
    monkeypatch.setattr("model_utils.make_inference", mock_make_inference)
    monkeypatch.setattr("model_utils.load_model", mock_load_model)

    from main import app
    return TestClient(app)


def test_healthcheck(init_test_client) -> None:
    response = init_test_client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_token_correctness(init_test_client) -> None:
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json={
            "culmen_length_mm": 36.7,
            "culmen_depth_mm": 19.3,
            "flipper_length_mm": 193.0,
            "body_mass_g": 3450.0,
            "sex": 1,
            "island_Biscoe": 0,
            "island_Dream": 0,
            "island_Torgersen": 1
        }
    )
    assert response.status_code == 200
    assert "species" in response.json()


def test_token_not_correctness(init_test_client):
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer kedjkj"},
        json={
            "culmen_length_mm": 36.7,
            "culmen_depth_mm": 19.3,
            "flipper_length_mm": 193.0,
            "body_mass_g": 3450.0,
            "sex": 1,
            "island_Biscoe": 0,
            "island_Dream": 0,
            "island_Torgersen": 1
        }
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid authentication credentials"
    }


def test_token_absent(init_test_client):
    response = init_test_client.post(
        "/predictions",
        json={
            "culmen_length_mm": 36.7,
            "culmen_depth_mm": 19.3,
            "flipper_length_mm": 193.0,
            "body_mass_g": 3450.0,
            "sex": 1,
            "island_Biscoe": 0,
            "island_Dream": 0,
            "island_Torgersen": 1
        }
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Not authenticated"
    }


def test_inference(init_test_client):
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json={
            "culmen_length_mm": 36.7,
            "culmen_depth_mm": 19.3,
            "flipper_length_mm": 193.0,
            "body_mass_g": 3450.0,
            "sex": 1,
            "island_Biscoe": 0,
            "island_Dream": 0,
            "island_Torgersen": 1
        }
    )
    assert response.status_code == 200
    assert response.json()["species"] == "Adelie"
