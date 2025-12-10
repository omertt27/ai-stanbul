# 🚀 Deploy AI Istanbul Backend to Google Cloud Run

## Why Google Cloud Run?

### ✅ Advantages Over Render
- **Better Free Tier**: 2M requests/month, 360,000 GB-seconds free
- **More Memory**: Up to 8GB RAM (vs Render's 512MB free)
- **Auto-Scaling**: Scales to zero when not in use (cost-effective)
- **Fast Cold Starts**: Better performance than Render
- **Google Infrastructure**: Highly reliable, global CDN
- **Container-Based**: Full control over environment

### 💰 Cost Comparison
| Provider | Free Tier | Paid (2GB RAM) | Notes |
|----------|-----------|----------------|-------|
| **Render** | 512MB RAM | $7/month | Crashes with ML models |
| **GCP Cloud Run** | 2M requests free | ~$5-10/month | Only pay for usage |

---

## 📋 Prerequisites

1. **Google Cloud Account** (Free $300 credit for new users)
2. **gcloud CLI** installed
3. **Docker** installed (for local testing)
4. **Git** for version control

---

## 🔧 Step 1: Prepare Your Backend for Cloud Run

### 1.1 Create Dockerfile

Create `/Users/omer/Desktop/ai-stanbul/backend/Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8080/api/health || exit 1

# Run the application
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
```

### 1.2 Create .dockerignore

Create `/Users/omer/Desktop/ai-stanbul/backend/.dockerignore`:

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env
.git
.gitignore
*.log
*.db
node_modules/
.vscode/
.idea/
*.md
Dockerfile
.dockerignore
```

### 1.3 Update requirements.txt

Ensure your `backend/requirements.txt` includes all dependencies:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
aiofiles==23.2.1
redis==5.0.1
sentence-transformers==2.2.2
torch==2.1.1
numpy==1.24.3
scikit-learn==1.3.2
# Add all your other dependencies
```

---

## 🚀 Step 2: Deploy to Google Cloud Run

### 2.1 Install gcloud CLI

```bash
# macOS
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

### 2.2 Initialize gcloud

```bash
# Login to your Google account
gcloud auth login

# Create a new project (or use existing)
gcloud projects create ai-istanbul-backend --name="AI Istanbul"

# Set the project
gcloud config set project ai-istanbul-backend

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 2.3 Deploy Backend

```bash
cd /Users/omer/Desktop/ai-stanbul/backend

# Build and deploy in one command
gcloud run deploy ai-istanbul-backend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --set-env-vars "ENVIRONMENT=production" \
  --set-env-vars "LLM_API_URL=https://4r1su4zfuok0s7-8000.proxy.runpod.net/v1"
```

**Explanation:**
- `--memory 2Gi`: 2GB RAM (enough for ML models)
- `--cpu 2`: 2 vCPUs for better performance
- `--timeout 300`: 5 minutes (for LLM responses)
- `--max-instances 10`: Auto-scale up to 10 instances
- `--allow-unauthenticated`: Public API access

### 2.4 Get Your Backend URL

After deployment, you'll get a URL like:
```
https://ai-istanbul-backend-[hash]-uc.a.run.app
```

---

## 🔐 Step 3: Add Environment Variables

### 3.1 Set Secrets (Recommended)

For sensitive data like API keys:

```bash
# Create secret
echo -n "your-secret-value" | gcloud secrets create LLM_API_KEY --data-file=-

# Grant access to Cloud Run
gcloud secrets add-iam-policy-binding LLM_API_KEY \
  --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor

# Update service to use secret
gcloud run services update ai-istanbul-backend \
  --update-secrets=LLM_API_KEY=LLM_API_KEY:latest
```

### 3.2 Set Public Environment Variables

```bash
gcloud run services update ai-istanbul-backend \
  --set-env-vars "ENVIRONMENT=production,\
CORS_ORIGINS=https://aistanbul.net,\
REDIS_URL=your-redis-url"
```

---

## 🗄️ Step 4: Set Up Cloud SQL or Redis (Optional)

### Option A: Google Cloud Memorystore (Redis)

```bash
# Enable Memorystore API
gcloud services enable redis.googleapis.com

# Create Redis instance
gcloud redis instances create ai-istanbul-cache \
  --size=1 \
  --region=us-central1 \
  --tier=basic

# Get connection info
gcloud redis instances describe ai-istanbul-cache --region=us-central1
```

### Option B: Use Upstash Redis (Serverless)

Easier option: Use [Upstash](https://upstash.com/) for serverless Redis:
- Free tier: 10,000 commands/day
- No infrastructure management
- Perfect for Cloud Run

---

## 🔧 Step 5: Update Frontend Configuration

Update your frontend `.env` to use the new Cloud Run backend:

```bash
# /Users/omer/Desktop/ai-stanbul/frontend/.env
VITE_API_URL=https://ai-istanbul-backend-[your-hash]-uc.a.run.app
```

---

## 📊 Step 6: Monitor and Optimize

### 6.1 View Logs

```bash
# Stream logs
gcloud run services logs tail ai-istanbul-backend --region us-central1

# View in console
# https://console.cloud.google.com/run
```

### 6.2 Set Up Monitoring

```bash
# Enable Cloud Monitoring
gcloud services enable monitoring.googleapis.com

# View metrics in console
# CPU, Memory, Request count, Latency, etc.
```

### 6.3 Configure Alerts

Set up alerts for:
- High memory usage (>80%)
- Error rate spikes
- Cold start latency
- Request volume

---

## 💡 Optimization Tips

### 1. Reduce Cold Starts

```bash
# Keep 1 instance always warm
gcloud run services update ai-istanbul-backend \
  --min-instances 1
```

⚠️ Note: This costs ~$5-10/month but eliminates cold starts

### 2. Use Lazy Loading for ML Models

Modify your backend to load models on-demand:

```python
# Instead of loading at startup
model = None

def get_model():
    global model
    if model is None:
        model = load_embedding_model()
    return model
```

### 3. Enable Request Concurrency

```bash
gcloud run services update ai-istanbul-backend \
  --concurrency 80
```

### 4. Use Cloud CDN for Static Assets

```bash
gcloud compute backend-services update ai-istanbul-backend \
  --enable-cdn
```

---

## 🚀 Quick Deploy Script

Create `/Users/omer/Desktop/ai-stanbul/deploy-to-gcp.sh`:

```bash
#!/bin/bash

set -e

echo "🚀 Deploying AI Istanbul Backend to Google Cloud Run..."

# Configuration
PROJECT_ID="ai-istanbul-backend"
SERVICE_NAME="ai-istanbul-backend"
REGION="us-central1"

# Set project
gcloud config set project $PROJECT_ID

# Build and deploy
cd backend
gcloud run deploy $SERVICE_NAME \
  --source . \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --region $REGION \
  --format 'value(status.url)')

echo "✅ Deployment complete!"
echo "🌐 Backend URL: $SERVICE_URL"
echo ""
echo "Next steps:"
echo "1. Update frontend/.env with: VITE_API_URL=$SERVICE_URL"
echo "2. Test the API: curl $SERVICE_URL/api/health"
```

Make it executable:
```bash
chmod +x /Users/omer/Desktop/ai-stanbul/deploy-to-gcp.sh
```

---

## 🎯 Cost Estimation

### Free Tier (Generous)
- **2 million requests/month**
- **360,000 GB-seconds compute**
- **180,000 vCPU-seconds**
- **1 GB network egress**

### Typical Monthly Cost (2GB RAM, 2 vCPU)
- **Low traffic** (10k requests/day): ~$5-10/month
- **Medium traffic** (100k requests/day): ~$20-30/month
- **High traffic** (1M requests/day): ~$100-150/month

**Much cheaper than Render's $7/month for 512MB!**

---

## 📋 Pre-Deployment Checklist

- [ ] Install gcloud CLI
- [ ] Create GCP account (get $300 free credit)
- [ ] Create Dockerfile in backend folder
- [ ] Test Docker build locally
- [ ] Update requirements.txt
- [ ] Set environment variables
- [ ] Configure CORS origins
- [ ] Set up Redis (Upstash recommended)
- [ ] Test API endpoints after deployment
- [ ] Update frontend with new backend URL
- [ ] Monitor logs for errors
- [ ] Set up alerts

---

## 🐛 Troubleshooting

### Error: "Memory limit exceeded"
```bash
# Increase memory
gcloud run services update ai-istanbul-backend --memory 4Gi
```

### Error: "Service timeout"
```bash
# Increase timeout
gcloud run services update ai-istanbul-backend --timeout 600
```

### Error: "Cold start too slow"
```bash
# Keep instance warm
gcloud run services update ai-istanbul-backend --min-instances 1
```

### Check logs
```bash
gcloud run services logs read ai-istanbul-backend --limit 50
```

---

## 🎉 Benefits Summary

✅ **2GB+ RAM** - No more crashes!  
✅ **Auto-scaling** - Handle traffic spikes  
✅ **Pay per use** - Cost-effective  
✅ **Global CDN** - Fast worldwide  
✅ **Easy deployment** - One command  
✅ **Built-in monitoring** - Real-time metrics  
✅ **$300 free credit** - Try for free  

---

## 📞 Next Steps

1. **Create GCP account**: https://console.cloud.google.com
2. **Install gcloud CLI**: `brew install google-cloud-sdk`
3. **Run deployment script**: `./deploy-to-gcp.sh`
4. **Update frontend**: Change `VITE_API_URL` in `.env`
5. **Test**: Visit your new backend URL

---

**Ready to deploy?** Let me know if you want me to help with any specific step!

**Estimated time**: 15-20 minutes for first deployment ⏱️
