# 📱 GPS Navigation - Quick Reference Card

## 🎯 One-Page Summary

### Database Connection
```bash
Host: dpg-d4jg45e3jp1c73b6gas0-a.frankfurt-postgres.render.com
Database: aistanbul_postgre
User: aistanbul_postgre_user
SSL: Required (automatic)
```

### Quick Commands
```bash
# Test connection
python test_db_connection.py

# Create GPS tables
cd backend && python -c "from database import Base, engine; from models import LocationHistory, NavigationSession, RouteHistory, NavigationEvent, UserPreferences, ChatSession, ConversationHistory; Base.metadata.create_all(bind=engine)"

# Start server
cd backend && uvicorn main:app --reload
```

### GPS Navigation Tables
1. **location_history** - GPS tracking (3s intervals)
2. **navigation_sessions** - Active navigation
3. **route_history** - Completed routes
4. **navigation_events** - Event log
5. **user_preferences** - User settings
6. **chat_sessions** - Chat with navigation
7. **conversation_history** - Messages + routes

### User Flow
```
User: "How can I go to Taksim?"
  ↓
GPS location captured
  ↓
Route calculated (OSRM)
  ↓
NavigationSession created
  ↓
Turn-by-turn instructions
  ↓
Real-time GPS tracking
  ↓
Route saved on completion
```

### API Endpoints
```python
POST /api/chat
{
  "message": "How can I go to Taksim?",
  "user_location": {"latitude": 41.0082, "longitude": 28.9784}
}

POST /api/navigation/start
POST /api/navigation/update
POST /api/navigation/complete
GET  /api/location/history
```

### Database Queries
```sql
-- Active sessions
SELECT COUNT(*) FROM navigation_sessions WHERE status = 'active';

-- Routes today
SELECT COUNT(*) FROM route_history WHERE completed_at > CURRENT_DATE;

-- Popular destinations
SELECT destination_name, COUNT(*) FROM navigation_sessions GROUP BY destination_name;

-- Average rating
SELECT AVG(user_rating) FROM route_history WHERE user_rating IS NOT NULL;
```

### Deployment Steps
```bash
# 1. Deploy to Render
git push origin main

# 2. Create tables (via Render Shell)
python backend/create_gps_tables.py

# 3. Test
curl https://your-app.onrender.com/api/chat -d '{"message": "Route to Taksim"}'
```

### Documentation
- **DATABASE_SETUP_GUIDE.md** - Setup
- **GPS_DEPLOYMENT_GUIDE.md** - Deploy
- **GPS_POSTGRES_COMPLETE_SUMMARY.md** - Full summary

---

**Status:** ✅ READY  
**Updated:** December 1, 2025
