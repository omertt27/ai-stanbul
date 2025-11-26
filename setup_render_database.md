# Setup Render Database

## Database Created ✅

- **Name:** aistanbul_postgre
- **Hostname:** dpg-d4jg45e3jp1c73b6gas0-a
- **Region:** Frankfurt (EU Central)
- **Instance:** Basic-256mb

## Next Steps

### 1. Update Web Service Environment Variable ⚡ DO THIS FIRST

1. Go to Render Dashboard → Your Web Service (api.aistanbul.net)
2. Click "Environment" tab
3. Add or update `DATABASE_URL`:
   ```
   postgres://aistanbul_postgre_user:YOUR_PASSWORD@dpg-d4jg45e3jp1c73b6gas0-a.frankfurt-postgres.render.com/aistanbul_postgre
   ```
   (Replace YOUR_PASSWORD with the actual password from the database page)
4. Save → Wait for auto-deploy (2-3 minutes)

### 2. Initialize Database Schema (Option A - Via Render Shell)

Once web service is deployed with the new DATABASE_URL:

1. Go to your Web Service → "Shell" tab
2. Run these commands:

```bash
# Navigate to backend
cd backend

# Run migrations (if you have alembic)
python -m alembic upgrade head

# OR run init script (if you have one)
python init_db.py

# OR create tables from models
python -c "from database import engine, Base; from models import *; Base.metadata.create_all(engine); print('✅ Tables created')"
```

### 2. Initialize Database Schema (Option B - From Local)

If Option A doesn't work, do it from your local machine:

```bash
cd /Users/omer/Desktop/ai-stanbul/backend

# Set the DATABASE_URL (get the EXTERNAL URL from Render)
export DATABASE_URL="postgres://aistanbul_postgre_user:PASSWORD@dpg-d4jg45e3jp1c73b6gas0-a.frankfurt-postgres.render.com/aistanbul_postgre"

# Run migrations
python -m alembic upgrade head

# OR
python init_db.py

# OR create tables directly
python -c "from database import engine, Base; from models import *; Base.metadata.create_all(engine); print('✅ Tables created')"
```

### 3. Verify Connection

Test that your backend can connect:

```bash
curl https://api.aistanbul.net/api/health/detailed
```

Expected response:
```json
{
  "status": "healthy",
  "database": "connected",
  "pureLlm": {
    "available": true
  },
  "services_active": 12
}
```

### 4. Test Chat Endpoint

```bash
curl -X POST https://api.aistanbul.net/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "recommend a cheap Turkish restaurant in Sultanahmet", "language": "en"}'
```

Expected: Real restaurant recommendations (not just prompt template)

## Troubleshooting

### If "Database connection failed"

Check Render logs for:
- `🔒 Using PostgreSQL database connection: postgresql://dpg-d4jg45e3jp1c73b6gas0-a...`
- NOT `localhost`

If still showing localhost:
1. Double-check DATABASE_URL is set correctly in Environment tab
2. Make sure you saved changes
3. Manually redeploy if needed

### If "Table does not exist" errors

Run the database initialization (Step 2 above)

### If Chat Returns Only Prompt Template

This means:
1. Database is connected ✅
2. But tables are empty ❌

You need to:
- Initialize schema (Step 2)
- Load restaurant data (see next section)

## Load Restaurant Data

After tables are created, load your restaurant data:

```bash
cd /Users/omer/Desktop/ai-stanbul/backend

# If you have a seed script
python seed_restaurants.py

# OR if you have JSON data
python -c "
from database import SessionLocal
from models import Restaurant
import json

db = SessionLocal()
with open('data/restaurants_database.json', 'r') as f:
    data = json.load(f)
    for restaurant in data.get('restaurants', []):
        db_restaurant = Restaurant(**restaurant)
        db.add(db_restaurant)
db.commit()
print('✅ Restaurants loaded')
"
```

## Summary Checklist

- [ ] Database created on Render ✅
- [ ] DATABASE_URL added to Web Service environment
- [ ] Web Service redeployed with new DATABASE_URL
- [ ] Database schema initialized (tables created)
- [ ] Restaurant data loaded
- [ ] Health check shows "database": "connected"
- [ ] Chat endpoint returns real restaurant data
- [ ] Prices show as $ or $$ (not TL)

---

**Current Status:** Database created, now update environment variable!
