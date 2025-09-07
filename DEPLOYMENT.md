# AIstanbul Backend Deployment Guide

## 🚀 Multiple Deployment Options

The backend supports multiple deployment methods for maximum flexibility:

### Option 1: Using start.py (Recommended)
```bash
python start.py
```
- ✅ Automatic dependency checking
- ✅ Fallback to standalone mode if needed
- ✅ Production-ready configuration

### Option 2: Direct uvicorn
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Option 3: Python module execution
```bash
python main.py
```

## 🌐 Platform-Specific Deployment

### Render
1. **Build Command**: `pip install -r requirements.txt`
2. **Start Command**: `python start.py`
3. **Environment Variables**:
   - `PORT`: Auto-set by Render
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `GOOGLE_MAPS_API_KEY`: Your Google Maps API key

### Railway
1. **Build Command**: `pip install -r requirements.txt`
2. **Start Command**: `python start.py`
3. **Port**: Auto-detected via $PORT

### Heroku
1. **Procfile**: Already configured
2. **Buildpack**: `heroku/python`
3. **Start Command**: Automatic via Procfile

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "start.py"]
```

## 🔧 Configuration

### Required Environment Variables
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: 0.0.0.0)

### Optional Environment Variables
- `OPENAI_API_KEY`: For AI responses
- `GOOGLE_MAPS_API_KEY`: For restaurant recommendations
- `DATABASE_URL`: Custom database URL

## ✅ Health Checks

- **Endpoint**: `GET /health`
- **Response**: JSON with service status
- **Use for**: Load balancer health checks

## 🔍 Troubleshooting

### Port Binding Issues
1. Ensure `PORT` environment variable is set
2. Check if port is available
3. Verify firewall settings

### Dependency Issues
- The app includes automatic dependency installation
- Fallback modes available for missing dependencies

### Database Issues
- SQLite database created automatically
- Fallback models provided for missing dependencies

## 📊 Production Checklist

- ✅ Environment variables configured
- ✅ Health endpoint responding
- ✅ Database connection working
- ✅ API keys configured (if needed)
- ✅ Logs accessible
- ✅ Error handling active
