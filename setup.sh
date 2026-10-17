#!/usr/bin/env bash
# Sets up the project's directory structure and installs dependencies.
# Run from the repo root: ./setup.sh

set -e

echo "Creating project directories..."
mkdir -p data          # generated synthetic datasets (gitignored - see .gitignore)
mkdir -p scripts        # data generation and modeling scripts
mkdir -p notebooks      # exploratory / sprint notebooks
mkdir -p outputs        # figures, result tables, model artifacts

echo "Installing Python dependencies..."
pip install -r requirements.txt --break-system-packages

echo "Done. Directory structure:"
find . -maxdepth 1 -type d -not -path './.git' -not -path '.'