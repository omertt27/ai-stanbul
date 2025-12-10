# 🎯 Quick Migration Commands

## Before Migration

### 1. Backup Current Database
```bash
# Set your Render database URL
export RENDER_DB="postgresql://user:pass@render-host.com:5432/database"

# Backup to file
pg_dump "$RENDER_DB" -F c -f render_backup_$(date +%Y%m%d).dump

# Or SQL format
pg_dump "$RENDER_DB" > render_backup_$(date +%Y%m%d).sql
```

### 2. Test AWS RDS Connection
```bash
cd /Users/omer/Desktop/ai-stanbul
python3 test_rds_connection.py
```

---

## Migration (Choose One Method)

### Method A: Automated Script (Easiest)

```bash
cd /Users/omer/Desktop/ai-stanbul

# 1. Edit .env to add:
nano .env
# Add: RENDER_DATABASE_URL=postgresql://...

# 2. Run migration
python migrate_render_to_aws.py

# 3. Follow prompts
```

### Method B: pg_dump/pg_restore (Traditional)

```bash
# 1. Export from Render
pg_dump "postgresql://render-url" -F c -f backup.dump

# 2. Import to AWS RDS
pg_restore -h database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com \
  -U postgres -d postgres backup.dump

# 3. Or use SQL format
psql "postgresql://aws-rds-url" < backup.sql
```

### Method C: Direct Pipe (Fastest for Small DBs)

```bash
pg_dump "postgresql://render-url" | \
  psql "postgresql://database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres"
```

---

## After Migration

### 1. Verify Data
```bash
# Connect to AWS RDS
psql "postgresql://database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres"

# Check tables
\dt

# Check row counts
SELECT 'users' as table, COUNT(*) FROM users
UNION ALL
SELECT 'posts', COUNT(*) FROM posts;
-- Add all your tables

# Exit
\q
```

### 2. Update Backend Configuration
```bash
cd /Users/omer/Desktop/ai-stanbul/backend

# Edit .env
nano .env

# Replace DATABASE_URL with:
# DATABASE_URL=postgresql://user:pass@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres
```

### 3. Test Backend
```bash
cd /Users/omer/Desktop/ai-stanbul/backend

# Start server
python main.py

# In another terminal, test
curl http://localhost:8000/api/health
```

### 4. Update Cloud Run (if deployed)
```bash
# Update secret
echo -n "postgresql://..." | gcloud secrets versions add DATABASE_URL --data-file=-

# Or update env var
gcloud run services update ai-istanbul-backend \
  --set-env-vars "DATABASE_URL=postgresql://..."

# Verify
gcloud run services describe ai-istanbul-backend
```

---

## Verification Checklist

```bash
# ✅ Tables exist
psql "$AWS_RDS_URL" -c "\dt"

# ✅ Data migrated
psql "$AWS_RDS_URL" -c "SELECT COUNT(*) FROM users;"

# ✅ Sequences work
psql "$AWS_RDS_URL" -c "SELECT last_value FROM users_id_seq;"

# ✅ Backend connects
cd backend && python -c "from database import engine; print('✅ Connected!' if engine else '❌ Failed')"

# ✅ API works
curl http://localhost:8000/api/health
```

---

## Emergency Rollback

```bash
# Switch back to Render immediately
cd /Users/omer/Desktop/ai-stanbul/backend

# Edit .env - restore old DATABASE_URL
nano .env

# Restart
python main.py

# Update Cloud Run
gcloud run services update ai-istanbul-backend \
  --set-env-vars "DATABASE_URL=postgresql://old-render-url"
```

---

## One-Liner Migration (if you trust it)

```bash
cd /Users/omer/Desktop/ai-stanbul && \
  echo "RENDER_DATABASE_URL=postgresql://render-url" >> .env && \
  echo "DATABASE_URL=postgresql://database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres" >> .env && \
  python migrate_render_to_aws.py
```

---

## Useful Queries

### Compare Row Counts (Run on Both DBs)
```sql
SELECT 
    schemaname,
    tablename,
    n_live_tup as row_count
FROM pg_stat_user_tables
ORDER BY n_live_tup DESC;
```

### List All Tables with Sizes
```sql
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Check Indexes
```sql
SELECT 
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY tablename, indexname;
```

---

**Ready? Start here:**
```bash
cd /Users/omer/Desktop/ai-stanbul
python migrate_render_to_aws.py
```
