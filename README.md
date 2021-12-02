# DenseNet API

This is a web-service for retrieving image classifications from a pre-trained DenseNet model

## Developer Setup

Recommended installation:

- Install [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Create a new python environment `create -n densenetapi python=3.9`

## Tests

- Install testing dependencies `pip install -r ./tests/requirements.txt`
- Run tests `python -m pytest ./tests`

## Deployment

- Run `uvicorn densenetapi:app`
