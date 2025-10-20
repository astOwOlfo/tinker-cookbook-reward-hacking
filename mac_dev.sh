#!/bin/bash

docker run -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  python:3.11 \
  bash -c "pip install uv && uv sync && exec bash && source .env"