# 🔌 AI Istanbul API Endpoints Reference

**Backend URL:** https://ai-stanbul.onrender.com

---

## 🏥 Health & Status Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Basic health check |
| `/api/health/detailed` | GET | Detailed service status |
| `/api/health/pure-llm` | GET | LLM service health |
| `/api/health/circuit-breakers` | GET | Circuit breaker status |
| `/api/health/readiness` | GET | Readiness probe |
| `/api/health/liveness` | GET | Liveness probe |

---

## 🔐 Authentication Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | User registration |
| `/api/auth/login` | POST | User login |
| `/api/auth/admin-login` | POST | Admin login |
| `/api/auth/refresh` | POST | Refresh token |
| `/api/auth/logout` | POST | Logout |
| `/api/auth/profile` | GET | Get user profile |

---

## 💬 Chat Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | **Main chat endpoint (streaming)** |
| `/api/chat/pure-llm` | POST | Direct LLM chat |
| `/api/chat/ml` | POST | ML-enhanced chat |

---

## 🤖 LLM Service Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/llm/health` | GET | LLM health check |
| `/api/v1/llm/generate` | POST | Generate response |
| `/api/v1/llm/istanbul-query` | POST | Istanbul-specific query |
| `/api/v1/llm/feedback` | POST | Submit feedback |
| `/api/v1/llm/interaction` | POST | Log interaction |
| `/api/v1/llm/profile/{user_id}` | GET | Get user profile |

---

## 📊 Statistics Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/llm/stats` | GET | General statistics |
| `/api/v1/llm/stats/signals` | GET | Signal statistics |
| `/api/v1/llm/stats/performance` | GET | Performance metrics |
| `/api/v1/llm/stats/cache` | GET | Cache statistics |
| `/api/v1/llm/stats/users` | GET | User statistics |
| `/api/v1/llm/stats/errors` | GET | Error statistics |
| `/api/v1/llm/stats/hourly` | GET | Hourly statistics |
| `/api/v1/llm/stats/export` | GET | Export statistics |

---

## 🎯 Main Chat Usage

### Correct Frontend Configuration

```env
VITE_API_URL=https://ai-stanbul.onrender.com/api
VITE_API_BASE_URL=https://ai-stanbul.onrender.com/api
```

### Chat Request Example

```bash
curl -X POST https://ai-stanbul.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about Istanbul",
    "conversation_id": "test-123"
  }'
```

### Expected Response

```json
{
  "response": "Istanbul is a beautiful city...",
  "conversation_id": "test-123",
  "metadata": {
    "timestamp": "2025-11-20T20:50:00Z",
    "model": "gpt-4",
    "tokens_used": 150
  }
}
```

---

## 🔧 Fine-Tuning Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/llm/tuning/run` | POST | Run fine-tuning |
| `/api/v1/llm/tuning/report` | GET | Get tuning report |
| `/api/v1/llm/personalization/metrics` | GET | Personalization metrics |

---

## 📝 Important Notes

### 1. **No `/stream` Endpoint**
The backend uses `/api/chat` for streaming responses, not `/api/stream`.

### 2. **Frontend Integration**
Make sure your frontend calls:
- ✅ `https://ai-stanbul.onrender.com/api/chat`
- ❌ NOT `https://ai-stanbul.onrender.com/api/stream`
- ❌ NOT `https://ai-stanbul.onrender.com/api/ai/stream`

### 3. **CORS**
Currently configured to allow all origins (`*`). 
For production, update to specific domains in Render environment variables:
```
ALLOWED_ORIGINS=["https://aistanbul.net","https://www.aistanbul.net"]
```

### 4. **Authentication**
Most endpoints require authentication. Include JWT token in headers:
```
Authorization: Bearer <token>
```

---

## 🧪 Testing Commands

### Test Health
```bash
curl https://ai-stanbul.onrender.com/api/health
```

### Test Chat
```bash
curl -X POST https://ai-stanbul.onrender.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello","conversation_id":"test"}'
```

### View All Endpoints
```bash
curl https://ai-stanbul.onrender.com/docs
```

### Get OpenAPI Spec
```bash
curl https://ai-stanbul.onrender.com/openapi.json
```

---

## 🐛 Common Issues

### Issue: 404 on `/api/stream`
**Solution:** Update frontend to use `/api/chat` instead

### Issue: CORS errors
**Solution:** Check Origin header and backend ALLOWED_ORIGINS

### Issue: 401 Unauthorized
**Solution:** Add authentication token or check if endpoint requires auth

### Issue: 500 Internal Server Error
**Solution:** Check Render logs for backend errors

---

## 📱 Interactive API Documentation

Visit: https://ai-stanbul.onrender.com/docs

This provides:
- Interactive endpoint testing
- Request/response schemas
- Authentication testing
- Real-time API exploration

---

## 🔗 Quick Links

- **Frontend:** https://aistanbul.net
- **Backend:** https://ai-stanbul.onrender.com
- **API Docs:** https://ai-stanbul.onrender.com/docs
- **OpenAPI Spec:** https://ai-stanbul.onrender.com/openapi.json

---

**Last Updated:** 2025-11-20 20:52 UTC
