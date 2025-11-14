# Semantic Cache Quick Reference

> **Priority 3.4** | Response Caching 2.0 | Status: ✅ Ready for Deployment

---

## 🚀 Quick Deploy (3 Commands)

```bash
# 1. Deploy
./deploy_semantic_cache.sh

# 2. Start backend
cd backend && uvicorn main:app --reload

# 3. Monitor
python monitor_semantic_cache.py
```

---

## ⚙️ Configuration (.env)

```bash
SEMANTIC_CACHE_ENABLED=true
SEMANTIC_CACHE_SIMILARITY_THRESHOLD=0.85
SEMANTIC_CACHE_MAX_CACHE_SIZE=10000
SEMANTIC_CACHE_TTL=86400
```

---

## 📊 Monitoring Commands

```bash
# Real-time monitoring
python monitor_semantic_cache.py

# Check cache size
redis-cli KEYS "semantic_cache:*" | wc -l

# View stats
redis-cli INFO stats | grep keyspace

# Watch memory
redis-cli INFO memory | grep used_memory_human
```

---

## 🔧 API Endpoints

```bash
# Stats
GET /api/admin/semantic-cache/stats

# Detailed stats
GET /api/admin/semantic-cache/detailed

# Health check
GET /api/admin/semantic-cache/health

# Clear cache
POST /api/admin/semantic-cache/clear
```

---

## 🎯 Success Targets

- **Week 1:** Hit rate >20%
- **Week 2:** Hit rate >35%
- **Month 1:** Hit rate >40%, Cost reduction >40%

---

## 🔍 Troubleshooting

### Low Hit Rate (<20%)
```bash
# Lower threshold
SEMANTIC_CACHE_SIMILARITY_THRESHOLD=0.80
```

### High Memory Usage
```bash
# Reduce cache size or TTL
SEMANTIC_CACHE_MAX_CACHE_SIZE=5000
SEMANTIC_CACHE_TTL=43200  # 12 hours
```

### Cache Not Working
```bash
# Check Redis
redis-cli ping

# Check logs
tail -f logs/backend.log | grep semantic_cache

# Check config
cat backend/.env | grep SEMANTIC_CACHE
```

---

## 🔙 Quick Disable

```bash
echo "SEMANTIC_CACHE_ENABLED=false" >> backend/.env
sudo systemctl restart ai-istanbul-backend
```

---

## 📚 Full Documentation

- **Deployment:** `PRIORITY_3.4_DEPLOYMENT_GUIDE.md`
- **Summary:** `PRIORITY_3.4_COMPLETE.md`
- **Status:** `PRIORITY_3_STATUS.md`
- **Code:** `backend/services/response_cache_semantic.py`

---

## 📞 Quick Help

1. Check deployment guide troubleshooting section
2. Run monitoring script for diagnostics
3. Check Redis connection: `redis-cli ping`
4. Review logs: `grep semantic_cache logs/backend.log`

---

**Target Impact:** 40%+ cost reduction, 30-50% faster responses
