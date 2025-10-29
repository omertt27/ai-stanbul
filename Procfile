# Procfile for deployment platforms (Render, Heroku, Railway, etc.)
# Uses production_server.py with async, caching, rate limiting, and monitoring

web: uvicorn production_server:app --host 0.0.0.0 --port $PORT --workers 2 --log-level info
