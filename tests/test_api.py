import base64
import urllib
from io import BytesIO

from PIL import Image
from fastapi.testclient import TestClient
from pytest import fixture

from densenetapi import app


@fixture
def encoded_image_and_class():
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    image = Image.open(filename)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    bytes = buffer.getvalue()
    return base64.b64encode(bytes).decode('utf-8'), 'dog'


client = TestClient(app)


def test_home():
    response = client.get('/')
    assert response.status_code == 200
    assert 'Welcome to the DenseNet API!' in response.text


def test_predict(encoded_image_and_class):
    encoded_image, expected_class = encoded_image_and_class
    payload = dict(image=encoded_image)
    response = client.post('/predict/', json=payload)
    assert response.status_code == 200
    assert response.json() == expected_class
