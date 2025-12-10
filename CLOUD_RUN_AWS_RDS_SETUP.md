# 🚀 Deploy Backend to Cloud Run with AWS RDS

## Current Situation

- ✅ Database migrated to AWS RDS
- ✅ AWS RDS is accessible (security group configured)
- ❌ Cloud Run needs proper configuration for AWS RDS (external database)

---

## 🔧 How to Configure Cloud Run for AWS RDS

### **Option 1: Via Google Cloud Console (Recommended)**

1. **Go to Cloud Run Console**:
   https://console.cloud.google.com/run

2. **Select your service** (e.g., `ai-istanbul-backend`)

3. **Click "EDIT & DEPLOY NEW REVISION"**

4. **Go to "Variables & Secrets" tab**

5. **Add Environment Variable**:
   ```
   Name: DATABASE_URL
   Value: postgresql://postgres:%2AiwP%23MDmX5dn8V%3A1LExE%7C70%3AO%3E%7Ci@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres?sslmode=require
   ```

6. **Important**: Do NOT use the "Cloud SQL Connections" section
   - That's only for Google Cloud SQL
   - AWS RDS is an external database, so just use the environment variable

7. **Click "DEPLOY"**

---

### **Option 2: Via gcloud CLI**

```bash
gcloud run services update ai-istanbul-backend \
  --region=europe-west1 \
  --update-env-vars="DATABASE_URL=postgresql://postgres:%2AiwP%23MDmX5dn8V%3A1LExE%7C70%3AO%3E%7Ci@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres?sslmode=require"
```

---

### **Option 3: Update and Redeploy**

If you're deploying via `gcloud run deploy`:

```bash
cd /Users/omer/Desktop/ai-stanbul/backend

gcloud run deploy ai-istanbul-backend \
  --source . \
  --region=europe-west1 \
  --platform=managed \
  --allow-unauthenticated \
  --set-env-vars="DATABASE_URL=postgresql://postgres:%2AiwP%23MDmX5dn8V%3A1LExE%7C70%3AO%3E%7Ci@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres?sslmode=require"
```

---

## ⚠️ Important: AWS RDS Security Group for Cloud Run

Cloud Run instances have dynamic IP addresses, so you need to allow Cloud Run to access your AWS RDS:

### **Option A: Allow All IPs (Not Recommended for Production)**

In AWS RDS Security Group:
- Add rule: `0.0.0.0/0` (PostgreSQL, port 5432)
- ⚠️ This is insecure but works for testing

### **Option B: Use NAT Gateway (Recommended for Production)**

Set up a Cloud NAT in Google Cloud to give your Cloud Run service a static IP, then whitelist that IP in AWS RDS.

### **Option C: Move Database to Cloud SQL (Best Long-term)**

For a Google Cloud Run app, using Google Cloud SQL is more integrated and secure.

---

## 🧪 Quick Test Script

After updating Cloud Run, test the deployment:

```bash
# Get your Cloud Run URL
CLOUD_RUN_URL=$(gcloud run services describe ai-istanbul-backend --region=europe-west1 --format='value(status.url)')

# Test health endpoint
curl $CLOUD_RUN_URL/health

# Test database connection
curl $CLOUD_RUN_URL/api/restaurants?limit=5
```

---

## 📝 Backend .env Update

Also update `/Users/omer/Desktop/ai-stanbul/backend/.env`:

```bash
DATABASE_URL=postgresql://postgres:%2AiwP%23MDmX5dn8V%3A1LExE%7C70%3AO%3E%7Ci@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres?sslmode=require
```

---

## 🔒 Security Best Practice

For production, use **Google Secret Manager** instead of environment variables:

```bash
# Store database URL in Secret Manager
echo -n "postgresql://postgres:PASSWORD@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres?sslmode=require" | \
  gcloud secrets create database-url --data-file=-

# Update Cloud Run to use secret
gcloud run services update ai-istanbul-backend \
  --region=europe-west1 \
  --update-secrets=DATABASE_URL=database-url:latest
```

---

## 🚀 Next Steps

1. **Update Cloud Run environment variable** (Option 1 above)
2. **Update AWS RDS security group** to allow Cloud Run IPs (temporarily use 0.0.0.0/0)
3. **Test the deployment**
4. **Monitor logs**: `gcloud run logs read ai-istanbul-backend --region=europe-west1`

---

## 💡 Alternative: Use Cloud SQL Instead

If you want better integration with Cloud Run:

1. Create a Cloud SQL PostgreSQL instance
2. Migrate from AWS RDS to Cloud SQL
3. Use native Cloud SQL connection in Cloud Run

Let me know which approach you'd like to take!
