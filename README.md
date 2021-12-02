# DenseNet API

This is a web-service for retrieving image classifications from a pre-trained DenseNet model

## Developer Setup

Recommended installation:

- Install [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Create a new python environment `create -n densenetapi python=3.9`
- Install the requirements `pip install -r requirements.txt`

## Tests

- Install testing dependencies `pip install -r ./tests/requirements.txt`
- Run tests `python -m pytest ./tests`

## Deployment

- Run `uvicorn densenetapi:app`

## Sources

- https://fastapi.tiangolo.com/
- https://pytorch.org/hub/pytorch_vision_densenet/
- https://stackoverflow.com/questions/26070547/decoding-base64-from-post-to-use-in-pil
- https://stackoverflow.com/questions/606191/convert-bytes-to-a-string
- https://stackoverflow.com/questions/48229318/how-to-convert-image-pil-into-base64-without-saving
