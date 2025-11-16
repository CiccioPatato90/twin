#!/bin/bash

# Install script for Random Walk Simulation
# Creates virtual environment and installs dependencies

set -e  # Exit on error

echo "Creating virtual environment..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Installation complete!"
echo "Run './run.sh' to start the server."

