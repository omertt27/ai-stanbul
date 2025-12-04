# 📦 Migrate Blog Posts to PostgreSQL

## 📊 Source Data
- **File**: `/backend/blog_posts.json`
- **Posts**: ~554 lines (multiple blog posts)
- **Content**: Rich Istanbul travel content with markdown formatting

## 🎯 Target Database
- **Name**: `aistanbul_postgre`
- **Type**: PostgreSQL 18
- **Location**: Frankfurt (EU Central)
- **Storage**: 15 GB (0.64% used)
- **Instance**: Basic-256mb

## 🚀 Migration Steps

### 1. Get Database URL from Render.com

From your Render.com PostgreSQL dashboard, copy the **Internal Database URL**:
```
postgres://username:password@hostname/database
```

Or set it as environment variable in backend:
```bash
DATABASE_URL=postgres://aistanbul_postgre_user:YOUR_PASSWORD@dpg-d4jg45e3jp1c73b6gas0-a/aistanbul_postgre
```

### 2. Run Migration Script

**Option A: From your local machine**
```bash
cd /Users/omer/Desktop/ai-stanbul/backend

# Set database URL (replace with your actual URL from Render)
export DATABASE_URL="postgres://aistanbul_postgre_user:YOUR_PASSWORD@dpg-d4jg45e3jp1c73b6gas0-a/aistanbul_postgre"

# Run migration
python migrate_blog_to_postgres.py
```

**Option B: Interactive (script will ask for URL)**
```bash
cd /Users/omer/Desktop/ai-stanbul/backend
python migrate_blog_to_postgres.py
# Script will prompt: "Enter PostgreSQL Database URL:"
# Paste your database URL
```

### 3. Verify Migration

**Check via API:**
```bash
curl https://ai-stanbul.onrender.com/api/blog/posts | jq '.total'
```

**Check via psql:**
```bash
# Copy PSQL Command from Render dashboard, then:
SELECT COUNT(*) FROM blog_posts;
SELECT id, title, likes_count FROM blog_posts LIMIT 5;
```

---

## 📋 What the Script Does

1. ✅ Connects to PostgreSQL database
2. ✅ Creates `blog_posts` table if not exists
3. ✅ Loads `blog_posts.json` file
4. ✅ Parses each post (title, content, author, district, etc.)
5. ✅ Inserts posts into database
6. ✅ Handles duplicates (asks user)
7. ✅ Shows progress and summary

---

## 🔧 Script Features

- **Safety**: Asks before overwriting existing data
- **Progress**: Shows batch progress every 10 posts
- **Error Handling**: Continues if individual post fails
- **Verification**: Shows total count and samples
- **Unicode**: Handles Turkish characters correctly

---

## 📊 Expected Results

```
🔄 Starting blog posts migration...
📊 Database: postgres://aistanbul_postgre_user:***...
✅ Table 'blog_posts' ready
📁 Loaded XX posts from JSON
📊 Existing posts in database: 0
✅ Migrated 10 posts...
✅ Migrated 20 posts...
...
==================================================
🎉 Migration Complete!
==================================================
✅ Added: XX posts
⚠️  Skipped: 0 posts
📊 Total in database: XX posts
==================================================

📝 Sample posts:
  - 1: Hidden Gems of Istanbul: Secret Rooftop Gardens... (2 likes)
  - 2: Istanbul's Coffee Culture Revolution: From Tra... (1 likes)
  ...

✅ Migration successful!
```

---

## 🐛 Troubleshooting

### Connection Error
```
Error: could not connect to server
```
**Fix**: Check database URL, ensure it's the **Internal** URL if running from Render backend

### Permission Error
```
Error: permission denied for table blog_posts
```
**Fix**: Ensure user has CREATE/INSERT permissions

### File Not Found
```
Error: File not found: blog_posts.json
```
**Fix**: Run script from `/backend` directory

---

## 🧪 Test After Migration

### 1. Test Backend API
```bash
curl https://ai-stanbul.onrender.com/api/blog/posts
```

### 2. Deploy Frontend Fix
```bash
cd frontend
npm run build
vercel --prod
```

### 3. Test Frontend
Open your site and check blog page - should now show posts!

---

## 🔗 Database Connection Info

**From Backend Code:**
The backend already uses this database via `DATABASE_URL` environment variable in `backend/database.py`.

**Connection String Format:**
```
postgres://USER:PASSWORD@HOST:PORT/DATABASE
```

**Your Database:**
- Host: `dpg-d4jg45e3jp1c73b6gas0-a`
- Port: `5432`
- Database: `aistanbul_postgre`
- User: `aistanbul_postgre_user`

---

## 📝 Next Steps After Migration

1. ✅ Verify posts in database
2. ✅ Test backend API endpoint
3. ✅ Deploy frontend with blog fix
4. ✅ Test blog page on site
5. ✅ Monitor for errors

---

## 💡 Tip: Backup Before Migration

```bash
# Backup existing database (if any)
pg_dump YOUR_DATABASE_URL > backup_before_blog_migration.sql
```

---

Generated: December 4, 2025
Ready to migrate blog posts from JSON to PostgreSQL! 🚀
