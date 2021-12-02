from fastapi.testclient import TestClient

from densenetapi import app

client = TestClient(app)


def test_home():
    response = client.get('/')
    assert response.status_code == 200
    assert 'Welcome to the DenseNet API!' in response.text


def test_predict():
    payload = dict(image='test_image')
    response = client.post('/predict/', json=payload)
    assert response.status_code == 200
    assert response.json() == payload
