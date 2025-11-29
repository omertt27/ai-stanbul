# 🎯 AI-Istanbul Quick Action Card

**Generated:** January 2025  
**Status:** 🔴 CRITICAL ACTIONS REQUIRED

---

## 🚨 PRIORITY 1: SECURITY (DO NOW - 5 MINUTES)

### Problem
Database credentials exposed on GitHub in commit history.

### Quick Fix
```bash
# 1. Go to Render Dashboard
open https://dashboard.render.com

# 2. Find PostgreSQL database → Click "Reset Password"
# 3. Copy new DATABASE_URL

# 4. Update backend service environment:
#    - Go to backend service
#    - Click "Environment" tab
#    - Update DATABASE_URL
#    - Save (auto-redeploys)

# 5. Update local file
vim /Users/omer/Desktop/ai-stanbul/backend/.env
# Replace DATABASE_URL with new value
```

### Verification
```bash
# Check backend logs on Render
# Should see: "✅ Database connection successful"
```

📄 **Full Guide:** `SECURITY_BREACH_QUICK_FIX.md`

---

## 🤖 PRIORITY 2: AI CHAT (TEST - 5 MINUTES)

### Problem
Chat returning fallback errors due to vLLM issues or circuit breaker.

### Status
✅ Circuit breaker disabled  
✅ vLLM restarted  
⚠️ May still serve cached error responses

### Quick Test
```bash
# Test with unique query (avoids cache)
curl -X POST "https://ai-stanbul.onrender.com/api/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Tell me about restaurants in Kadıköy at $(date +%s)\"}"

# Should get proper AI response, not fallback error
```

### If Still Getting Errors
```bash
# 1. Check vLLM is running
ssh runpod  # or access via RunPod web terminal
ps aux | grep vllm

# 2. Restart vLLM if needed
cd /workspace
./start_vllm.sh

# 3. Monitor logs
tail -f /workspace/vllm.log
```

📄 **Full Guide:** `QUICK_FIX_UPDATED.txt`

---

## 🎨 PRIORITY 3: FRONTEND CSP (AUTOMATIC)

### Problem
Unsplash images blocked by Content Security Policy.

### Status
✅ Fixed in `frontend/vercel.json`  
⏳ Waiting for Vercel to redeploy

### Verification
```bash
# After Vercel redeploys:
# 1. Open frontend in browser
# 2. Open Developer Console (F12)
# 3. Check for CSP errors (should be none)
# 4. Verify Unsplash images load
```

---

## ✅ COMPLETE CHECKLIST

### Right Now (15 minutes)
- [ ] **[5 min]** Rotate database password on Render
- [ ] **[2 min]** Update backend DATABASE_URL environment variable
- [ ] **[1 min]** Update local backend/.env file
- [ ] **[2 min]** Verify backend connects successfully
- [ ] **[5 min]** Test AI chat with unique query

### Today (1 hour)
- [ ] Monitor vLLM stability (check every 30 min)
- [ ] Test frontend after Vercel redeploys
- [ ] Verify no CSP errors in console
- [ ] Test complete user flow: frontend → backend → vLLM

### This Week (Optional)
- [ ] Clean Git history to remove exposed credentials
- [ ] Set up vLLM auto-restart script
- [ ] Implement backend cache-clear endpoint
- [ ] Add pre-commit hooks for credential detection

---

## 🔗 QUICK LINKS

### Dashboards
- **Render:** https://dashboard.render.com
- **RunPod:** https://www.runpod.io/console/pods
- **GitHub:** https://github.com/omertt27/ai-stanbul

### Endpoints to Test
```bash
# vLLM Health Check
curl https://fcn3h0wk2vf5sk-8000.proxy.runpod.net/health

# Backend Chat API
curl -X POST https://ai-stanbul.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}'

# Frontend
open https://your-frontend.vercel.app
```

### Documentation Files
- `COMPLETE_STATUS_REPORT.md` - Full status report
- `SECURITY_BREACH_QUICK_FIX.md` - Security quick fix
- `SECURITY_FIX_DATABASE_CREDENTIALS.md` - Detailed security guide
- `CIRCUIT_BREAKER_DISABLED.md` - Backend fix details
- `VLLM_404_ACTUAL_ISSUE.md` - vLLM troubleshooting
- `QUICK_FIX_UPDATED.txt` - Testing commands

---

## 🎯 SUCCESS CRITERIA

### You're done when:
✅ Old database password fails to authenticate  
✅ New password works in backend logs  
✅ AI chat returns proper responses (not fallback errors)  
✅ Frontend has no CSP errors  
✅ Unsplash images load properly  

---

## 🆘 IF SOMETHING GOES WRONG

### Backend won't connect to database
```bash
# Check DATABASE_URL format:
# postgresql://user:password@host/database

# Verify on Render:
# - Environment tab
# - DATABASE_URL matches new password
```

### vLLM returns 404 or errors
```bash
# Restart vLLM
ssh runpod
cd /workspace
./start_vllm.sh
tail -f /workspace/vllm.log
# Wait for "Application startup complete"
```

### Still getting cached errors
```bash
# Test with timestamp to avoid cache
curl -X POST https://ai-stanbul.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What time is it? $(date)\"}"
```

---

**Last Updated:** January 2025  
**Next Action:** 🔴 Rotate database password NOW  
**Estimated Time:** 15 minutes total  
**Priority:** P0 CRITICAL
