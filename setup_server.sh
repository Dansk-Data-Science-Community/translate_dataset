#!/usr/bin/env bash
set -euo pipefail

echo "Installing uv via snap..."
sudo snap install astral-uv --classic

echo "Running 'uv sync'..."
uv sync

sudo apt-get update
sudo apt-get install -y build-essential python3.12-dev

echo "Done."
