from fastapi.testclient import TestClient
import pytest
import torch 
from src.api.main import app
from src.api.auth import create_access_token

# Initialize TestClient correctly
client = TestClient(app)

@pytest.fixture
def auth_token():
    return create_access_token({"sub": "test_user"})

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint_unauthorized():
    response = client.post("/api/v1/predict", json={"text": "test text"})
    assert response.status_code == 401

def test_predict_endpoint(auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.post(
        "/api/v1/predict",
        json={"text": "I'm feeling happy today!"},
        headers=headers
    )
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_batch_predict_endpoint(auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = client.post(
        "/api/v1/batch-predict",
        json={"texts": ["text1", "text2"]},
        headers=headers
    )
    assert response.status_code == 200
    assert "predictions" in response.json()