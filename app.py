#!/usr/bin/env python3
"""Application entry point for AI Service - CarePlan Generator."""

import sys
import os

# Add src to Python path so we can import directly from packages
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Now we can import main directly
from main import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
