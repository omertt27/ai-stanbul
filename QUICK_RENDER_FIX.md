# Quick Fix for Render Deployment

## 🚨 Current Error
```
RuntimeError: Form data requires "python-multipart" to be installed.
```

## ⚡ Quick Solution

### 1. Update Render Configuration

In your Render dashboard, change:

**Build Command**:
```bash
bash backend/build_simple.sh
```

**Start Command**:
```bash
cd backend && python start.py
```

### 2. Alternative (if above doesn't work)

**Build Command**:
```bash
cd backend && pip install python-multipart==0.0.6 && pip install -r requirements.txt
```

**Start Command**:
```bash
cd backend && python start.py
```

## 🔧 What Was Fixed

1. **requirements.txt** - Fixed version pinning for Python 3.13
2. **build_simple.sh** - New reliable build script that installs python-multipart first
3. **main_standalone.py** - Added automatic dependency checking and installation
4. **start.py** - Enhanced with smart backend selection

## ✅ Expected Result

After deploying with the new configuration, you should see logs like:

```
📦 Installing python-multipart...
✅ All core dependencies verified
🚀 AI-stanbul Backend Starting...
✅ python-multipart is available
🌐 Starting server on 0.0.0.0:8000
```

## 🔄 Deploy Steps

1. **Commit and push** the updated files to GitHub
2. **Update Render build/start commands** as shown above
3. **Redeploy** your service
4. **Check logs** for success messages

That's it! Your FastAPI backend should now start without the python-multipart error. 🚀
