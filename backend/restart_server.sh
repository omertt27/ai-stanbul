#!/bin/bash
cd /Users/omer/Desktop/ai-stanbul/backend
export PYTHONPATH=/Users/omer/Desktop/ai-stanbul/backend:$PYTHONPATH
/Users/omer/Desktop/ai-stanbul/.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
