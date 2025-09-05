#!/usr/bin/env python3
"""
Simple app.py wrapper for uvicorn
"""

# Import everything from main
from main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
