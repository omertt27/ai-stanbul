# POI Fetch Scripts

This directory contains scripts for fetching and managing Istanbul POI data with **zero ongoing costs**.

## 🎯 Quick Start (3-4 Days to $0/Month Database)

### Prerequisites

```bash
# Install Python packages
pip install requests python-dotenv

# Create .env file with your API keys
cp .env.example .env
# Edit .env and add your API keys
```

### Day 1: Setup & Test (2 hours)

```bash
# Get Google Places API key from:
# https://console.cloud.google.com/

# Add to .env file
echo "GOOGLE_PLACES_API_KEY=your_key_here" > .env

# Test connection
python test_apis.py
```

### Day 2: Fetch POIs (6 hours)

```bash
# Fetch 500+ Istanbul POIs
python fetch_pois.py

# Output: pois_raw_YYYYMMDD_HHMMSS.json
```

### Day 3: Create Database (4 hours)

```bash
# Create database schema
python create_db.py

# Import POIs
python import_pois.py pois_raw_20251018_140523.json

# Verify data
python verify_db.py
```

### Day 4: Integration (see main guide)

See [ONE_TIME_POI_FETCH_GUIDE.md](../ONE_TIME_POI_FETCH_GUIDE.md) for integration steps.

## 📁 Files

| File | Purpose | When to Run |
|------|---------|-------------|
| `istanbul_zones.py` | Zone definitions | Auto-imported |
| `fetch_pois.py` | Fetch POIs from Google | Day 2 |
| `create_db.py` | Create SQLite database | Day 3 (once) |
| `import_pois.py` | Import JSON → Database | Day 3 |
| `verify_db.py` | Check database stats | Day 3 |
| `test_apis.py` | Test API connections | Day 1 |

## 🔄 Maintenance

### Refresh POI Data (Every 6 Months)

```bash
# Re-run fetch
python fetch_pois.py

# Import new data (merges with existing)
python import_pois.py pois_raw_NEW_DATE.json

# Verify
python verify_db.py
```

### Monitor Database Health

```bash
python verify_db.py
```

## 💰 Cost Savings

- **Before:** $170-220/month for continuous API access
- **After:** $0/month with cached data
- **Annual Savings:** $2,040-2,640

## 🎓 What You Get

✅ 500+ Real Istanbul POIs  
✅ Ratings, reviews, photos  
✅ <10ms query response  
✅ Offline-ready  
✅ Unlimited scaling  

## 📚 Documentation

- **Full Guide:** [ONE_TIME_POI_FETCH_GUIDE.md](../ONE_TIME_POI_FETCH_GUIDE.md)
- **Integration Dashboard:** [INTEGRATION_VISUAL_DASHBOARD.md](../INTEGRATION_VISUAL_DASHBOARD.md)
- **Launch Checklist:** [LAUNCH_CHECKLIST.md](../LAUNCH_CHECKLIST.md)

## ❓ Troubleshooting

### API Quota Exceeded

Wait 24 hours or split fetch across multiple days.

### Database Locked

Add timeout: `sqlite3.connect('places_cache.db', timeout=10)`

### Missing Photos

Normal with free tier. Download selectively or use placeholders.

## 🚀 Next Steps

1. ✅ Run scripts to build database
2. ✅ Integrate with backend (see CachedPOIService)
3. ✅ Test all app features
4. ✅ Set 6-month refresh reminder
5. ✅ Launch and save $2,000+/year!
