#!/bin/bash

docker run -it --rm \
  -v $(pwd):/workspace \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w /workspace \
  python:3.11 \
  bash -c "apt-get update && apt-get install -y docker.io && pip install uv && uv sync && exec bash && source .env"