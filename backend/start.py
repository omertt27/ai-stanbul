#!/usr/bin/env python3
import sys
import os
import uvicorn

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8000))
    
    # Start the uvicorn server
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
