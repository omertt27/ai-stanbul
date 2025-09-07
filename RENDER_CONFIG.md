# Render Environment Configuration

## Environment Variables to Set in Render Dashboard:

### Required Environment Variables:
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_PLACES_API_KEY=your_google_places_api_key_here

### Python Runtime (Add this to suppress Python 3.13 issues):
PYTHON_VERSION=3.11.6

## Render Service Configuration:

### Build Command:
pip install --upgrade pip && pip install -r backend/requirements.txt

### Start Command:
./start-render.sh

### Alternative Build Command (if issues persist):
pip install --upgrade pip && pip install -r requirements-minimal.txt

## Troubleshooting:

If you still get build errors, try these Build Commands in order:

1. Basic:
pip install -r backend/requirements.txt

2. With pip upgrade:
pip install --upgrade pip && pip install -r backend/requirements.txt

3. Minimal dependencies:
pip install -r requirements-minimal.txt

4. Force reinstall:
pip install --upgrade pip && pip install --force-reinstall -r backend/requirements.txt

5. No cache:
pip install --no-cache-dir -r backend/requirements.txt
