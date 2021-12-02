from fastapi.testclient import TestClient

from densenetapi import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert 'Welcome to the DenseNet API!' in response.text
