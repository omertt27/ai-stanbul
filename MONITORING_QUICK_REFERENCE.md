# 📊 Backend Monitoring - Quick Reference

## 🚀 Instant Commands

### Show Dashboard (One-Time)
```bash
python scripts/monitor_backend_performance.py
# or
./scripts/monitor.sh
```

### Watch Mode (Auto-Refresh)
```bash
python scripts/monitor_backend_performance.py --watch
# or
./scripts/monitor.sh watch
```

### Generate Report
```bash
python scripts/monitor_backend_performance.py --report
# or
./scripts/monitor.sh report
```

### View Logs
```bash
python scripts/monitor_backend_performance.py --tail 30
# or
./scripts/monitor.sh logs 30
```

### Follow Logs (Real-Time)
```bash
python scripts/monitor_backend_performance.py --tail --follow
# or
./scripts/monitor.sh follow
```

## 🎯 Common Workflows

### Development Workflow
```bash
# Terminal 1: Start backend
cd backend
uvicorn main:app --reload

# Terminal 2: Monitor in watch mode
./scripts/monitor.sh watch

# Terminal 3: Test queries
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Best restaurants in Istanbul"}'
```

### Load Testing Workflow
```bash
# Terminal 1: Run backend
cd backend && uvicorn main:app --reload

# Terminal 2: Monitor (5s refresh for rapid updates)
./scripts/monitor.sh watch 5

# Terminal 3: Run tests
python test_all_query_types.py
```

### Daily Review Workflow
```bash
# Generate and save report
./scripts/monitor.sh report logs/daily_report_$(date +%Y%m%d).txt

# Review low confidence queries
grep "Low Confidence" logs/daily_report_*.txt

# Check trends over time
ls -lt logs/daily_report_*.txt | head -5
```

## 📈 Key Metrics Thresholds

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Avg Latency | < 50ms | 50-100ms | > 100ms |
| P95 Latency | < 100ms | 100-200ms | > 200ms |
| Accuracy | > 95% | 90-95% | < 90% |
| Low Confidence Rate | < 5% | 5-10% | > 10% |
| Error Rate | < 1% | 1-3% | > 3% |

## 🔍 What to Look For

### ✅ Healthy System
```
Total Predictions:   1,500+
Avg Latency:        45ms
Accuracy:           95%+
Low Confidence:     < 5%
Errors:             0-2
Diverse intent distribution
```

### ⚠️ Warning Signs
```
Low Confidence:     > 10%
Latency:           > 100ms
Errors:            > 3
One intent dominates (> 50%)
User feedback trending negative
```

### 🚨 Critical Issues
```
Accuracy:          < 85%
P95 Latency:      > 300ms
Error Rate:       > 5%
Backend unreachable
Constant low confidence predictions
```

## 🛠️ Quick Fixes

### High Latency
1. Check system resources (CPU, memory)
2. Review database queries
3. Check network connectivity
4. Consider caching improvements

### Low Confidence
1. Review queries in dashboard
2. Add to training data
3. Retrain model
4. Update backend

### High Error Rate
1. Check backend logs: `tail -f backend/logs/*.log`
2. Review stack traces
3. Check dependencies
4. Verify model files exist

## 📁 File Locations

```
logs/ml_production/
├── metrics.jsonl          # Raw logs
└── feedback.jsonl         # User feedback

backend/logs/
└── app.log               # Backend logs

models/
└── istanbul_intent_classifier_finetuned/  # Model files
```

## 🔄 Monitoring → Improvement Loop

1. **Monitor**: `./scripts/monitor.sh watch`
2. **Identify**: Low confidence queries, misclassifications
3. **Collect**: Export training candidates
4. **Label**: Add correct intents to training data
5. **Train**: `python scripts/finetune_intent_classifier.py`
6. **Deploy**: Restart backend with new model
7. **Verify**: Monitor again to see improvements
8. **Repeat**: Continuous improvement cycle

## 💡 Pro Tips

1. **Run watch mode during peak hours** to catch issues as they happen
2. **Generate reports daily** to track trends
3. **Set up cron jobs** for automated reporting
4. **Review low confidence queries weekly** for training improvements
5. **Monitor before/after deployments** to verify improvements
6. **Keep historical reports** to track long-term trends

## 🆘 Troubleshooting

### Backend not accessible
```bash
# Check if running
lsof -i :8000

# Start backend
cd backend && uvicorn main:app --reload
```

### No metrics showing
```bash
# Send test query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}'

# Refresh dashboard
python scripts/monitor_backend_performance.py
```

### Logs not updating
```bash
# Check write permissions
ls -la logs/ml_production/

# Create directory if needed
mkdir -p logs/ml_production

# Verify monitoring is enabled in backend
grep "ML_MONITORING_AVAILABLE" backend/main.py
```

## 📞 Quick Help

```bash
# Show all options
python scripts/monitor_backend_performance.py --help
./scripts/monitor.sh help

# Check backend health
curl http://localhost:8000/health

# View raw logs
cat logs/ml_production/metrics.jsonl | jq '.'

# Count predictions by intent
cat logs/ml_production/metrics.jsonl | jq -r '.predicted_intent' | sort | uniq -c
```

---

**For full documentation**: See [BACKEND_MONITORING_GUIDE.md](BACKEND_MONITORING_GUIDE.md)
