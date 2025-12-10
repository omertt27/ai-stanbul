# 🚀 Cloud Build Configuration for Continuous Deployment

## ✅ What You're Setting Up

**Continuous Deployment**: Every time you push to the `main` branch on GitHub, Cloud Build will:
1. Pull your code
2. Build a Docker container
3. Deploy to Cloud Run automatically

---

## 📋 Configuration Settings for Cloud Run Console

### Step 1: Source Repository
```
Repository Provider: GitHub
Repository: your-username/ai-stanbul
Branch: ^main$ (regex to match main branch)
```

### Step 2: Build Configuration
```
Build Type: Dockerfile
Source Location: /backend/Dockerfile
```

⚠️ **IMPORTANT**: Since your Dockerfile is in the `backend/` folder, set:
```
Source location: /backend/Dockerfile
```

### Step 3: Service Configuration
```
Region: us-central1 (or your preferred region)
CPU: 2 vCPU
Memory: 2 GiB
Request timeout: 300 seconds
Maximum instances: 10
Minimum instances: 0 (scale to zero for cost savings)
```

### Step 4: Authentication
```
Allow unauthenticated invocations: ✅ Yes
(This makes your API public)
```

### Step 5: Environment Variables
Add these in the Cloud Run console:

```
ENVIRONMENT=production
CORS_ORIGINS=https://aistanbul.net
LLM_API_URL=https://4r1su4zfuok0s7-8000.proxy.runpod.net/v1
PORT=8080
```

---

## 🔧 Alternative: Using cloudbuild.yaml (Advanced)

If you want more control, create `/Users/omer/Desktop/ai-stanbul/backend/cloudbuild.yaml`:

```yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '-t'
      - 'gcr.io/$PROJECT_ID/ai-istanbul-backend:$COMMIT_SHA'
      - '-t'
      - 'gcr.io/$PROJECT_ID/ai-istanbul-backend:latest'
      - '-f'
      - 'backend/Dockerfile'
      - 'backend/'
    
  # Push the container image to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'push'
      - 'gcr.io/$PROJECT_ID/ai-istanbul-backend:$COMMIT_SHA'
    
  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'ai-istanbul-backend'
      - '--image'
      - 'gcr.io/$PROJECT_ID/ai-istanbul-backend:$COMMIT_SHA'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--memory'
      - '2Gi'
      - '--cpu'
      - '2'
      - '--timeout'
      - '300'
      - '--max-instances'
      - '10'
      - '--set-env-vars'
      - 'ENVIRONMENT=production,LLM_API_URL=https://4r1su4zfuok0s7-8000.proxy.runpod.net/v1'

images:
  - 'gcr.io/$PROJECT_ID/ai-istanbul-backend:$COMMIT_SHA'
  - 'gcr.io/$PROJECT_ID/ai-istanbul-backend:latest'

options:
  machineType: 'E2_HIGHCPU_8'
  logging: CLOUD_LOGGING_ONLY

timeout: '1200s'
```

---

## 🎯 Recommended Configuration (Simple)

For your use case, use the **Dockerfile** build type with these settings:

### In Cloud Run Console:

1. **Source Repository Settings**:
   ```
   Repository: Connect your GitHub repo
   Branch: ^main$
   Build Type: Dockerfile
   Source location: /backend/Dockerfile
   ```

2. **Service Settings**:
   ```
   Service name: ai-istanbul-backend
   Region: us-central1
   CPU allocation: CPU is always allocated
   Memory: 2 GiB
   CPU: 2
   Request timeout: 300
   Maximum requests per container: 80
   Minimum instances: 0
   Maximum instances: 10
   ```

3. **Environment Variables** (Add in Cloud Run console):
   ```
   ENVIRONMENT=production
   LLM_API_URL=https://4r1su4zfuok0s7-8000.proxy.runpod.net/v1
   CORS_ORIGINS=https://aistanbul.net,http://localhost:3000
   ```

---

## ⚠️ Important: Dockerfile Location

Since your Dockerfile is in `/backend/`, you have two options:

### Option 1: Keep Dockerfile in backend/ (Recommended)
Set in Cloud Run console:
```
Source location: /backend/Dockerfile
```

Cloud Build will automatically use `/backend` as the build context.

### Option 2: Move Dockerfile to root
If Cloud Run expects Dockerfile at root:
```bash
cd /Users/omer/Desktop/ai-stanbul
mv backend/Dockerfile ./Dockerfile

# Update Dockerfile to adjust paths:
# COPY backend/requirements.txt .
# COPY backend/ .
```

---

## 🔐 Environment Variables & Secrets

### For Sensitive Data (Recommended):

1. **Create secrets in Google Secret Manager**:
```bash
echo -n "your-llm-api-key" | gcloud secrets create LLM_API_KEY --data-file=-
echo -n "your-db-password" | gcloud secrets create DATABASE_PASSWORD --data-file=-
```

2. **Grant Cloud Run access**:
```bash
PROJECT_NUMBER=$(gcloud projects describe ai-istanbul-backend --format="value(projectNumber)")

gcloud secrets add-iam-policy-binding LLM_API_KEY \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

3. **Configure in Cloud Run** (Environment Variables tab):
   - Click "Add a reference to a secret"
   - Select your secret
   - Choose "Exposed as environment variable"

---

## 🚀 Deployment Flow

Once configured, your workflow will be:

```
1. Make code changes locally
   ↓
2. Commit to Git
   git add .
   git commit -m "Update transportation system"
   ↓
3. Push to GitHub
   git push origin main
   ↓
4. Cloud Build triggers automatically
   - Builds Docker image
   - Runs tests (optional)
   - Deploys to Cloud Run
   ↓
5. New version live in ~5 minutes! ✅
```

---

## 📊 Monitor Builds

View build status:
- **Cloud Build Console**: https://console.cloud.google.com/cloud-build
- **Cloud Run Console**: https://console.cloud.google.com/run

See logs:
```bash
# View recent builds
gcloud builds list --limit=5

# View specific build logs
gcloud builds log <BUILD_ID>

# View Cloud Run logs
gcloud run services logs read ai-istanbul-backend --limit=50
```

---

## ✅ Pre-Deployment Checklist

Before clicking "Deploy":

- [ ] Dockerfile exists at `backend/Dockerfile`
- [ ] .dockerignore configured properly
- [ ] requirements.txt is up to date
- [ ] Source location set to `/backend/Dockerfile`
- [ ] Branch regex is `^main$`
- [ ] Memory set to 2GiB or more
- [ ] CPU set to 2
- [ ] Timeout set to 300 seconds
- [ ] Environment variables configured
- [ ] CORS origins include your frontend domain
- [ ] Allow unauthenticated is checked
- [ ] GitHub repository connected

---

## 🎉 Benefits of Continuous Deployment

✅ **Automatic deployments** - Push to deploy  
✅ **Version control** - Every deployment is tracked  
✅ **Rollback easy** - Revert to previous version in 1 click  
✅ **No manual steps** - No SSH, no manual Docker commands  
✅ **Build logs** - See exactly what happened  
✅ **Fast** - Deploy in ~5 minutes  

---

## 🐛 Common Issues & Solutions

### Issue: "Dockerfile not found"
**Solution**: Set source location to `/backend/Dockerfile`

### Issue: "Build timeout"
**Solution**: Increase timeout in cloudbuild.yaml or use faster machine type

### Issue: "Permission denied"
**Solution**: Grant Cloud Build service account proper IAM roles:
```bash
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT --format="value(projectNumber)")

gcloud projects add-iam-policy-binding YOUR_PROJECT \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/run.admin"
```

### Issue: "Environment variables not working"
**Solution**: Set them in Cloud Run console under "Environment Variables" tab

---

## 📞 Next Steps After Deployment

1. **Get your backend URL**:
   ```bash
   gcloud run services describe ai-istanbul-backend \
     --region us-central1 \
     --format 'value(status.url)'
   ```

2. **Update frontend .env**:
   ```bash
   VITE_API_URL=https://ai-istanbul-backend-xyz-uc.a.run.app
   ```

3. **Test API**:
   ```bash
   curl https://your-backend-url/api/health
   ```

4. **Monitor**:
   - Watch Cloud Build logs for successful deployment
   - Check Cloud Run logs for runtime errors
   - Test transportation queries in frontend

---

**Ready to deploy?** Just click "Deploy" in the Cloud Run console and watch the magic happen! 🚀

**Estimated first build time**: 5-10 minutes ⏱️
