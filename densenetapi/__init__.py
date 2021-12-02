import base64
from io import BytesIO

import requests
import torch
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from torchvision import transforms


class ImagePayload(BaseModel):
    image: str


imagenet_class_response = requests.get('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt')
assert imagenet_class_response.status_code == 200
classes = imagenet_class_response.text.split('\n')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
if torch.cuda.is_available():
    model.to('cuda')
app = FastAPI()


@app.get("/")
def home():
    return HTMLResponse(
        """<h1>Welcome to the DenseNet API!</h1>
        <p>To use this service, send a  POST HTTP request to {this-url}/predict</p>
        <p>The JSON payload has the following format: {"image": "BASE64_STRING_OF_IMAGE"}</p>
        <p>Example request-response pair:</p>
        <p>Send request to model with curl -X POST -d '{"image": "CAT_IMAGE_ENCODED_DATA"}
        http://localhost:5000/predict'</p>
        This will respond with {"response": "cat"}</p>
        """)


@app.post('/predict/')
def predict(payload: ImagePayload):
    image = Image.open(BytesIO(base64.b64decode(payload.image)))
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_id = torch.topk(probabilities, 1)
    return classes[top_id]


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
