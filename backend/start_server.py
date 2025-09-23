#!/usr/bin/env python3
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    print("Starting AI-stanbul backend server...")
    print(f"Current directory: {current_dir}")
    print(f"Python path: {sys.path[:3]}")
    
    # Import the FastAPI app
    from main import app
    print("✅ Successfully imported FastAPI app")
    
    # Start the server
    import uvicorn
    print("Starting uvicorn server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install fastapi uvicorn")
except Exception as e:
    print(f"❌ Error starting server: {e}")
    import traceback
    traceback.print_exc()
