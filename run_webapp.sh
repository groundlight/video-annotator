#!/bin/bash
# Helper script to run the video-annotator webapp

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Try to activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set default port to 5001 if not specified
export PORT=${PORT:-5001}

# Enable debug mode if not already set (for testing)
export FLASK_DEBUG=${FLASK_DEBUG:-1}

# Run the webapp
python3 webapp.py

