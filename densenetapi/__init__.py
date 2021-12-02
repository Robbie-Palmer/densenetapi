from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


class ImagePayload(BaseModel):
    image: str


app = FastAPI()


@app.get("/")
def home():
    return HTMLResponse(
        """<h1>Welcome to the DenseNet API!</h1>
        <p>To use this service, send a  POST HTTP request to {this-url}/predict</p>
        <p>The JSON payload has the following format: {"image": "IMAGE_DATA_FOR_DENSENET_TO_CLASSIFY"}</p>
        <p>Example request-response pair:</p>
        Send request to model with curl -X POST -d '{"image": "CAT_IMAGE_DATA"} http://localhost:5000/predict'</p>
        This will respond with {"response": "cat"}</p>
        """)


@app.post('/predict/')
def predict(payload: ImagePayload):
    return payload


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
