# 🔐 Admin Credentials - Quick Reference

## Current Admin Credentials

### For React Admin Dashboard (http://localhost:3000/admin)

**Username:** `KAM`  
**Password:** `klasdsaqeqw_sawq123ws`

---

## Login Instructions

1. **Navigate to Admin Dashboard:**
   ```
   http://localhost:3000/admin
   ```

2. **Enter Credentials:**
   - **Username field:** `KAM`
   - **Password field:** `klasdsaqeqw_sawq123ws`

3. **Click "Sign In"**

4. **Access Unified Dashboard:**
   - After login, you'll see 5 tabs:
     - 🎯 System Overview
     - 🤖 LLM Analytics
     - 📝 Blog Analytics
     - 💬 Feedback
     - 👥 User Analytics

---

## Credential Sources

✅ **Verified from:** `backend/.env`
- `ADMIN_USERNAME=KAM`
- `ADMIN_PASSWORD_HASH=$2b$12$6jQy4SiSSe79pYdzzGCxluFyfHgwj8VN0WtLW4O9P9r34Ex9rosM6`

✅ **Password verified with:** `verify_admin_password.py`

---

## Alternative Credentials (From Documentation)

### Option 1: Current Production (Recommended)
- **Username:** KAM
- **Password:** klasdsaqeqw_sawq123ws
- **Status:** ✅ Active (verified in .env)

### Option 2: Alternative (From FINAL_DEPLOYMENT_CONFIG.md)
- **Password:** vDWvCVTXUl2uGdF7
- **Hash:** b1d85ec52f84955ef2b6aefa0cb29421042eb1f8e9e20425d3185422e54ca039 (SHA-256)
- **Status:** ⚠️ Different hash type (not bcrypt)

---

## Security Notes

🔒 **Password Storage:** Bcrypt hashed in backend/.env  
🔐 **Authentication:** JWT token-based after login  
⏱️ **Session:** 30-minute timeout with activity extension  
🌐 **Access:** Local development (http://localhost:3000/admin)

---

## Troubleshooting

### Issue: "Invalid credentials"
**Solution:** Ensure you're using:
- Username: `KAM` (case-sensitive)
- Password: `klasdsaqeqw_sawq123ws` (exact match)

### Issue: Backend not responding
**Solution:** Check if backend is running:
```bash
cd /Users/omer/Desktop/ai-stanbul/backend
python main.py
```

### Issue: Need to reset password
**Solution:** Run the password verification script:
```bash
python3 verify_admin_password.py
```

---

## Quick Test

Copy-paste these credentials for testing:

```
Username: KAM
Password: klasdsaqeqw_sawq123ws
```

---

**Last Verified:** November 15, 2025  
**Status:** ✅ Working  
**Environment:** Development (localhost:3000)

---

Happy Testing! 🚀
