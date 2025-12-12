#!/usr/bin/env bash
# run: source setup_server.sh

set -euo pipefail

echo "Installing uv via snap..."
sudo snap install astral-uv --classic

echo "Running 'uv sync'..."
uv sync

sudo apt-get update
sudo apt-get install -y build-essential python3.12-dev

echo
read -s -p "Enter your Hugging Face access token: " HF_TOKEN
echo

# Export token for this shell session
export HF_TOKEN="$HF_TOKEN"

echo "Done."
