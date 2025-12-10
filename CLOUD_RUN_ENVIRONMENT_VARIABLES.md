# 🔐 Environment Variables Configuration for Cloud Run

## ✅ Required Environment Variables

Add these in the **Cloud Run Console** → **Edit & Deploy New Revision** → **Variables & Secrets** tab:

### 1. Core Settings
```
ENVIRONMENT=production
PORT=8080
```

### 2. API Configuration
```
LLM_API_URL=https://4r1su4zfuok0s7-8000.proxy.runpod.net/v1
```

### 3. CORS Configuration
```
CORS_ORIGINS=https://aistanbul.net,http://localhost:3000
```

### 4. Database Configuration (if using external database)
```
DATABASE_URL=your-database-url
MONGODB_URI=your-mongodb-connection-string
```

### 5. Redis Configuration (if using Redis)
```
REDIS_URL=redis://your-redis-host:6379
REDIS_HOST=your-redis-host
REDIS_PORT=6379
```

---

## 📋 Complete Environment Variables List

Copy and paste this into Cloud Run console (one per line):

```env
# Core Configuration
ENVIRONMENT=production
PORT=8080
PYTHONUNBUFFERED=1

# API Configuration
LLM_API_URL=https://4r1su4zfuok0s7-8000.proxy.runpod.net/v1
LLM_TIMEOUT=120
LLM_MAX_TOKENS=768

# CORS Configuration
CORS_ORIGINS=https://aistanbul.net,http://localhost:3000

# Security (Optional - if needed)
JWT_SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here

# Database (if using)
DATABASE_URL=postgresql://user:pass@host:5432/dbname
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/dbname

# Redis Cache (if using)
REDIS_URL=redis://host:6379
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# Feature Flags
ENABLE_ANALYTICS=true
ENABLE_CACHING=true
ENABLE_RATE_LIMITING=false

# Logging
LOG_LEVEL=INFO
```

---

## 🔒 Using Google Secret Manager (Recommended for Sensitive Data)

For sensitive values like API keys, use **Secret Manager** instead of plain environment variables.

### Step 1: Create Secrets via gcloud CLI

```bash
# Create LLM API Key secret
echo -n "your-llm-api-key" | gcloud secrets create LLM_API_KEY --data-file=-

# Create Database Password
echo -n "your-db-password" | gcloud secrets create DATABASE_PASSWORD --data-file=-

# Create Redis Password
echo -n "your-redis-password" | gcloud secrets create REDIS_PASSWORD --data-file=-

# Create JWT Secret
echo -n "your-jwt-secret" | gcloud secrets create JWT_SECRET_KEY --data-file=-
```

### Step 2: Grant Cloud Run Access to Secrets

```bash
# Get your project number
PROJECT_NUMBER=$(gcloud projects describe ai-istanbul-backend --format="value(projectNumber)")

# Grant access to each secret
gcloud secrets add-iam-policy-binding LLM_API_KEY \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding DATABASE_PASSWORD \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding REDIS_PASSWORD \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding JWT_SECRET_KEY \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Step 3: Reference Secrets in Cloud Run Console

In Cloud Run console:
1. Go to **Variables & Secrets** tab
2. Click **"Reference a secret"**
3. Select the secret
4. Choose **"Expose as environment variable"**
5. Enter the variable name (e.g., `LLM_API_KEY`)

---

## 🎯 Environment Variables by Feature

### For Transportation RAG System
```env
# Already included in core configuration
# No additional env vars needed - system works out of the box
```

### For OpenStreetMap Integration
```env
# No env vars needed - CSP is configured in middleware.py
```

### For Analytics (Amplitude)
```env
AMPLITUDE_API_KEY=your-amplitude-key
ENABLE_ANALYTICS=true
```

### For Google Analytics
```env
GA_TRACKING_ID=G-2XXEMVNC7Z
ENABLE_GOOGLE_ANALYTICS=true
```

---

## 📝 How to Add Environment Variables in Cloud Run Console

### Method 1: Via UI (Recommended for First Setup)

1. Go to **Cloud Run Console**: https://console.cloud.google.com/run
2. Select your service: `ai-istanbul-backend`
3. Click **"Edit & Deploy New Revision"**
4. Scroll to **"Container, Variables & Secrets, Connections, Security"**
5. Click **"Variables & Secrets"** tab
6. Click **"Add Variable"**
7. Enter:
   - **Name**: `ENVIRONMENT`
   - **Value**: `production`
8. Click **"Add Variable"** again for each variable
9. Click **"Deploy"** at the bottom

### Method 2: Via gcloud CLI

```bash
# Set multiple environment variables at once
gcloud run services update ai-istanbul-backend \
  --region us-central1 \
  --set-env-vars "ENVIRONMENT=production,\
LLM_API_URL=https://4r1su4zfuok0s7-8000.proxy.runpod.net/v1,\
CORS_ORIGINS=https://aistanbul.net,\
PORT=8080,\
PYTHONUNBUFFERED=1"
```

### Method 3: Via YAML File

Create `env-vars.yaml`:
```yaml
ENVIRONMENT: production
PORT: "8080"
PYTHONUNBUFFERED: "1"
LLM_API_URL: https://4r1su4zfuok0s7-8000.proxy.runpod.net/v1
CORS_ORIGINS: https://aistanbul.net,http://localhost:3000
LLM_TIMEOUT: "120"
LLM_MAX_TOKENS: "768"
```

Deploy with:
```bash
gcloud run services update ai-istanbul-backend \
  --region us-central1 \
  --env-vars-file env-vars.yaml
```

---

## 🔍 Verify Environment Variables

After deployment, verify your environment variables:

```bash
# List all environment variables
gcloud run services describe ai-istanbul-backend \
  --region us-central1 \
  --format="value(spec.template.spec.containers[0].env)"
```

---

## 🚨 Important Notes

### ⚠️ Don't Include in Git
Never commit these to Git:
- API keys
- Database passwords
- JWT secrets
- Any sensitive credentials

Add to `.gitignore`:
```
.env
.env.local
.env.production
env-vars.yaml
```

### ✅ Use Secrets for Sensitive Data
Always use **Google Secret Manager** for:
- API keys
- Database credentials
- JWT secrets
- OAuth tokens
- Any password or token

### 🔄 Update Without Redeployment
You can update environment variables without rebuilding:
```bash
gcloud run services update ai-istanbul-backend \
  --region us-central1 \
  --set-env-vars "NEW_VAR=new_value"
```

---

## 📊 Environment Variables Priority

1. **Secrets** (highest priority - most secure)
2. **Environment Variables** (set in Cloud Run)
3. **Dockerfile ENV** (defaults in Dockerfile)
4. **Application defaults** (hardcoded in code)

---

## 🎯 Minimal Required Setup

For your Transportation RAG system to work, you need **at minimum**:

```env
ENVIRONMENT=production
PORT=8080
LLM_API_URL=https://4r1su4zfuok0s7-8000.proxy.runpod.net/v1
CORS_ORIGINS=https://aistanbul.net
```

That's it! Everything else is optional.

---

## 🧪 Testing Environment Variables

Test if your env vars are working:

```bash
# Deploy with debug endpoint
# Then call:
curl https://your-backend-url/api/health

# Should return service status including env check
```

Or add a debug endpoint to your FastAPI app:

```python
@app.get("/api/debug/env")
async def debug_env():
    return {
        "environment": os.getenv("ENVIRONMENT"),
        "port": os.getenv("PORT"),
        "llm_url_configured": bool(os.getenv("LLM_API_URL")),
        # Don't return actual secret values!
    }
```

---

## 📞 Quick Reference Card

```
┌─────────────────────────────────────────────────┐
│         ESSENTIAL ENVIRONMENT VARIABLES          │
├─────────────────────────────────────────────────┤
│ ENVIRONMENT=production                          │
│ PORT=8080                                       │
│ LLM_API_URL=https://4r1su4z...proxy.runpod.net │
│ CORS_ORIGINS=https://aistanbul.net             │
└─────────────────────────────────────────────────┘
```

---

## ✅ Checklist

Before deploying, ensure:
- [ ] All required env vars are set
- [ ] Sensitive data is in Secret Manager
- [ ] CORS origins include your frontend domain
- [ ] PORT is set to 8080 (Cloud Run default)
- [ ] LLM_API_URL is correct and accessible
- [ ] Environment variables don't contain typos
- [ ] No trailing spaces in values
- [ ] URLs don't have trailing slashes

---

**Next**: After setting these, click **"Deploy"** and your backend will be live with proper configuration! 🚀
