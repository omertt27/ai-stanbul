#!/usr/bin/env python3
"""
Render-compatible startup script for AIstanbul backend
This script ensures proper port binding for Render deployment
"""

import os
import sys
import subprocess

def main():
    # Get the PORT from environment variable
    port = os.environ.get('PORT')
    
    if not port:
        print("❌ PORT environment variable not set!")
        print("Using fallback port 8000 for local development")
        port = '8000'
    
    print(f"🚀 Starting AIstanbul backend on 0.0.0.0:{port}")
    
    # Build the uvicorn command
    cmd = [
        'uvicorn',
        'backend.main:app',
        '--host', '0.0.0.0',
        '--port', str(port),
        '--log-level', 'info',
        '--access-log'
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    # Execute uvicorn
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        sys.exit(0)

if __name__ == '__main__':
    main()
