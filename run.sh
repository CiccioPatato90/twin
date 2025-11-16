#!/bin/bash

# Run script for Random Walk Simulation
# Activates virtual environment and starts the server

set -e  # Exit on error

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Running install script..."
    ./install.sh
fi

echo "Activating virtual environment and starting server..."
source .venv/bin/activate && python server.py

