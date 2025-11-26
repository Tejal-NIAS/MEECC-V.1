#!/bin/bash

# Set up virtual environment if not exists
if [ ! -d "env" ]; then
    python3 -m venv env
fi

# Activate virtual environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python MEECC_V.1_revision.py



