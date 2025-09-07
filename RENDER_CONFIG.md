# Render Environment Configuration

## Environment Variables to Set in Render Dashboard:

### Required Environment Variables:
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_PLACES_API_KEY=your_google_places_api_key_here

### Python Runtime (Add this to suppress Python 3.13 issues):
PYTHON_VERSION=3.11

### CORS Configuration (CRITICAL for frontend connection):
FRONTEND_URL=https://aistanbul.vercel.app

## Render Service Configuration:

### Build Command:
pip install --upgrade pip setuptools wheel && pip install -r requirements-minimal.txt

### Pre-Deploy Command:
./pre-deploy.sh

### Start Command:
./start-render.sh

### Alternative Build Command (if issues persist):
pip install --upgrade pip && pip install -r requirements-minimal.txt

## API Endpoints (for frontend integration):

Your backend is deployed at: `https://ai-stanbul.onrender.com`

### Correct API URLs:
- Health Check: `https://ai-stanbul.onrender.com/health`
- AI Chat: `https://ai-stanbul.onrender.com/ai`
- AI Stream: `https://ai-stanbul.onrender.com/ai/stream`
- Root: `https://ai-stanbul.onrender.com/`

### ❌ WRONG URLs (causing 404):
- `https://ai-stanbul.onrender.com/ai/ai/stream` (double /ai)

## Frontend Configuration:

Update your frontend to use:
```javascript
const API_BASE_URL = 'https://ai-stanbul.onrender.com'
const STREAM_ENDPOINT = '/ai/stream'  // NOT /ai/ai/stream
```

## Troubleshooting:

If you still get build errors, try these Build Commands in order:

1. Recommended:
pip install --upgrade pip setuptools wheel && pip install -r requirements-minimal.txt

2. Basic:
pip install -r backend/requirements.txt

3. With pip upgrade:
pip install --upgrade pip && pip install -r backend/requirements.txt

4. Force reinstall:
pip install --upgrade pip && pip install --force-reinstall -r requirements-minimal.txt

5. No cache:
pip install --no-cache-dir -r requirements-minimal.txt
