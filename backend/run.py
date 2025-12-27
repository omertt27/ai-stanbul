#!/usr/bin/env python3
"""
AI Istanbul Backend - Single Entry Point

This is the ONLY way to run the backend server in development:
    python run.py

For production (Render, etc.):
    uvicorn main_modular:app --host 0.0.0.0 --port 8001 --workers 1

NEVER run `python main_modular.py` directly - it causes double app construction!
"""
import os
import sys

# Ensure we're in the backend directory
backend_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(backend_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Environment setup BEFORE any imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")


def main():
    """Start the backend server."""
    import uvicorn
    
    # Get settings
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("API_PORT", "8001")))
    reload = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              AI Istanbul Backend Server                       ║
╠══════════════════════════════════════════════════════════════╣
║  Host:    {host:<50} ║
║  Port:    {port:<50} ║
║  Reload:  {str(reload):<50} ║
║  Log:     {log_level:<50} ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    if reload:
        print("⚠️  WARNING: Running with --reload may cause duplicate initialization")
        print("   This is acceptable for active development only.\n")
    
    # Use uvicorn to run the app
    # IMPORTANT: This imports main_modular as a MODULE, not as __main__
    # So the app is constructed exactly ONCE
    uvicorn.run(
        "main_modular:app",
        host=host,
        port=port,
        reload=reload,
        workers=1,  # Single worker to prevent multi-process init issues
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
