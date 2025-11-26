# Fix Database Connection and Missing Dependencies

## Status: Critical Issues to Resolve

### Issue 1: Database Connection ✅ FIXED
**Problem:** Backend was trying to connect to localhost PostgreSQL instead of Render DATABASE_URL

**Solution Applied:**
- Fixed `/Users/omer/Desktop/ai-stanbul/backend/database.py`
- Added automatic conversion from `postgres://` to `postgresql://` (required for newer SQLAlchemy versions)
- Render's DATABASE_URL starts with `postgres://` but SQLAlchemy 1.4+ requires `postgresql://`

### Issue 2: Missing Dependencies ⚠️ MIGHT RESOLVE AUTOMATICALLY
**Problem:** 2/12 services failed to load:
```
entity_extractor: No module named 'numpy'
```

**Analysis:**
- ✅ `numpy>=1.24.0,<2.0.0` is already in requirements.txt
- ✅ Build script (`backend/build.sh`) installs requirements
- ⚠️ Numpy might not be installed yet because the code deployed to Render hasn't been updated

**Theory:** Once we deploy the database.py fix, Render will rebuild and install all dependencies including numpy. This should resolve the entity_extractor error automatically.

## Steps to Fix

### 1. Deploy Database Fix to Render (REQUIRED)

The database.py fix has been applied locally. This is **THE CRITICAL FIX**:

```bash
cd /Users/omer/Desktop/ai-stanbul

# Review the change
git diff backend/database.py

# Commit and deploy
git add backend/database.py
git commit -m "fix: convert postgres:// to postgresql:// for SQLAlchemy 2.0"
git push origin main
```

### 2. Wait for Render to Rebuild

Once you push:
1. Render will automatically detect the changes
2. It will rebuild (takes 2-3 minutes)
3. It will reinstall all dependencies (including numpy)
4. Backend will restart with the fixed database connection

Watch the Render logs for:
```
✅ Using PostgreSQL database connection: postgresql://...
✅ Service Manager initialized: 12/12 services active
```

### 3. No Additional Dependency Fixes Needed

Since numpy is already in requirements.txt and the build script installs all requirements, the numpy error should resolve automatically after the rebuild.

### 3. Verify Database Connection After Deploy

Once deployed, check Render logs for:

```
✅ Using PostgreSQL database connection: postgresql://...
✅ Service Manager initialized: 12/12 services active
```

### 4. Test Full End-to-End

After both fixes are deployed:

```bash
# Test health endpoint
curl https://your-backend.onrender.com/health

# Should show:
# "pure_llm_status": "healthy"
# "database": "connected"
# "services_active": 12

# Test chat endpoint
curl -X POST https://your-backend.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "recommend a cheap Turkish restaurant in Sultanahmet",
    "language": "en"
  }'
```

Expected response:
- Real AI response (not fallback)
- Restaurant recommendations with prices as $ or $$ (not TL)
- Context from database data

## Troubleshooting

### If Database Still Shows "localhost"

Check Render environment variables:

1. Go to Render Dashboard → Your Web Service → Environment
2. Verify `DATABASE_URL` is set and starts with `postgres://`
3. If not set, add it from Render PostgreSQL database dashboard

### If Services Still Fail

Check specific service errors in Render logs:

```
⚠️ Some services failed to load: X errors
  - service_name: error message
```

Common fixes:
- **numpy error**: Add `numpy>=1.24.0` to requirements.txt
- **entity_extractor error**: Likely depends on numpy, will resolve after numpy is installed
- **Other ML libraries**: Check if scikit-learn, spacy, etc. are needed

### If SQL Syntax Error Persists

The "Textual SQL expression" error means a raw SQL string needs to be wrapped in `text()`:

Look for patterns like:
```python
# WRONG
query.order_by("column_name")

# CORRECT
from sqlalchemy import text
query.order_by(text("column_name"))
```

If you see this error in logs, share the full error traceback and I can pinpoint the exact file/line.

## Next Steps After Fix

1. ✅ Verify database connection works
2. ✅ Verify all 12 services load successfully
3. ✅ Test chat endpoint returns real AI responses
4. ✅ Verify prices show as $, $$, $$$ (not TL or ranges)
5. ✅ Test with different queries (restaurants, attractions, transportation)
6. 🎉 Production ready!

## Quick Deploy Command

```bash
cd /Users/omer/Desktop/ai-stanbul
git add backend/database.py backend/requirements.txt
git commit -m "fix: database connection and missing dependencies"
git push origin main
```

Then wait 2-3 minutes for Render to deploy and check logs.
