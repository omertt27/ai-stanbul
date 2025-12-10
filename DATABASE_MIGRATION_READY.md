# ✅ Database Migration: Complete Setup

## 🎯 What's Ready

### Migration Tools Created:
1. **`migrate_render_to_aws.py`** - Automated migration script
   - Connects to both databases
   - Migrates all tables, data, sequences, and indexes
   - Shows progress and generates report
   - Handles errors gracefully

2. **`DATABASE_MIGRATION_GUIDE.md`** - Comprehensive guide
   - Step-by-step instructions
   - Multiple migration methods
   - Verification procedures
   - Troubleshooting tips

3. **`MIGRATION_QUICK_COMMANDS.md`** - Quick reference
   - One-liner commands
   - Verification checklist
   - Emergency rollback

4. **`backend/.env`** - Updated configuration
   - AWS RDS connection string placeholder
   - Migration instructions

---

## 🚀 How to Migrate (3 Simple Steps)

### Step 1: Get Your Render Database URL

```bash
# From Render Dashboard:
# Go to your database → "Connection" tab
# Copy "External Database URL"
```

### Step 2: Configure Environment

```bash
cd /Users/omer/Desktop/ai-stanbul

# Edit .env
nano .env

# Add BOTH URLs:
RENDER_DATABASE_URL=postgresql://your-render-url
DATABASE_URL=postgresql://username:password@database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres
```

### Step 3: Run Migration

```bash
cd /Users/omer/Desktop/ai-stanbul

# Install dependency if needed
pip install psycopg2-binary

# Run migration
python migrate_render_to_aws.py

# Answer "yes" when prompted
```

**Expected Time**: 1-5 minutes depending on database size

---

## 📊 What Gets Migrated

✅ **All Tables**
- Table structure (columns, types, constraints)
- All rows of data
- Primary keys

✅ **Sequences**
- Auto-increment values
- Preserves next ID numbers

✅ **Indexes**
- Performance indexes
- Unique indexes

✅ **Data Integrity**
- Foreign keys
- Check constraints
- Default values

---

## ✅ Verification After Migration

### Quick Check:

```bash
# 1. Connect to AWS RDS
psql "postgresql://database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres"

# 2. List tables
\dt

# 3. Check a table
SELECT COUNT(*) FROM users;  -- Replace 'users' with your table

# 4. Exit
\q
```

### Test Backend:

```bash
cd /Users/omer/Desktop/ai-stanbul/backend

# Update .env to use AWS RDS (should already be set)
# DATABASE_URL=postgresql://...

# Start backend
python main.py

# Test API
curl http://localhost:8000/api/health
```

---

## 📝 Post-Migration Checklist

After successful migration:

- [ ] ✅ All tables visible in AWS RDS
- [ ] ✅ Row counts match Render database
- [ ] ✅ Backend connects to AWS RDS successfully
- [ ] ✅ API endpoints work correctly
- [ ] ✅ Test a few queries (create, read, update)
- [ ] ✅ Update Cloud Run environment variables
- [ ] ✅ Test production deployment
- [ ] ✅ Keep Render database as backup for 7-30 days
- [ ] ✅ Monitor application for 24-48 hours
- [ ] ✅ Delete Render database (after confirming everything works)

---

## 🔄 Alternative Migration Methods

### If Automated Script Doesn't Work:

#### Method A: pg_dump/pg_restore
```bash
# 1. Backup from Render
pg_dump "postgresql://render-url" -F c -f backup.dump

# 2. Restore to AWS RDS
pg_restore -h database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com \
  -U postgres -d postgres backup.dump
```

#### Method B: Direct Pipe (One Command)
```bash
pg_dump "postgresql://render-url" | \
  psql "postgresql://database-1.cleeowy8eb7v.eu-central-1.rds.amazonaws.com:5432/postgres"
```

---

## 🆘 Troubleshooting

### Issue: "Can't connect to Render"
```bash
# Verify URL is correct
echo $RENDER_DATABASE_URL

# Test connection
psql "$RENDER_DATABASE_URL" -c "SELECT 1;"
```

### Issue: "Can't connect to AWS RDS"
```bash
# Verify RDS is publicly accessible
# Check security group allows port 5432
# Test connection
python3 test_rds_connection.py
```

### Issue: "Table already exists"
```sql
-- Clear AWS RDS first
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
GRANT ALL ON SCHEMA public TO postgres;

-- Then re-run migration
```

### Issue: "Permission denied"
```sql
-- Grant all permissions
GRANT ALL PRIVILEGES ON DATABASE postgres TO your_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_user;
```

---

## 🔙 Emergency Rollback

If something goes wrong, switch back to Render immediately:

```bash
cd /Users/omer/Desktop/ai-stanbul/backend

# Edit .env - comment out AWS, uncomment Render
nano .env

# Restart backend
python main.py

# Update Cloud Run (if deployed)
gcloud run services update ai-istanbul-backend \
  --set-env-vars "DATABASE_URL=postgresql://render-url"
```

---

## 📊 Migration Report

After migration completes, you'll get a report file:
- `migration_report_YYYYMMDD_HHMMSS.json`

Contains:
- Tables migrated
- Rows migrated
- Sequences updated
- Indexes created
- Any errors encountered
- Duration

---

## 🎉 Success Indicators

You'll know migration succeeded when:

✅ Script shows "Migration completed successfully!"
✅ All tables visible in AWS RDS
✅ Row counts match Render database
✅ Backend starts without errors
✅ API health check returns 200
✅ Test queries work correctly
✅ Production deployment connects successfully

---

## 📚 Documentation Reference

- **Full Guide**: `DATABASE_MIGRATION_GUIDE.md`
- **Quick Commands**: `MIGRATION_QUICK_COMMANDS.md`
- **Migration Script**: `migrate_render_to_aws.py`
- **Test RDS Connection**: `test_rds_connection.py`
- **AWS Integration**: `BACKEND_AWS_INTEGRATION_COMPLETE.md`
- **Deployment Guide**: `COMPLETE_CLOUD_DEPLOYMENT_GUIDE.md`

---

## 🚀 Ready to Migrate?

1. **Backup Render database** (CRITICAL!)
2. **Configure environment variables** in `.env`
3. **Run migration script**: `python migrate_render_to_aws.py`
4. **Verify data** in AWS RDS
5. **Test backend** locally
6. **Update production** (Cloud Run)
7. **Monitor for 24-48 hours**
8. **Delete Render database** after confirming success

---

**Start Migration Now:**

```bash
cd /Users/omer/Desktop/ai-stanbul
python migrate_render_to_aws.py
```

**Good luck! 🎯**

---

## 💡 Pro Tips

1. **Run during low traffic** - Minimize impact on users
2. **Keep Render backup** - At least 30 days as safety net
3. **Test thoroughly** - Don't rush the verification
4. **Monitor logs** - Watch for database errors after switch
5. **Have rollback ready** - Know how to switch back quickly
6. **Document changes** - Note what was migrated and when
7. **Enable AWS backups** - Set up automated RDS backups
8. **Optimize after** - Add indexes if needed for performance

---

**Last Updated**: December 10, 2025
**Status**: ✅ Ready for Migration
