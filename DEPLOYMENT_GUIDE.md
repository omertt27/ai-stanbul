# 🚀 DEPLOYMENT GUIDE - Istanbul AI Travel Blog

## 📋 **RECOMMENDED DEPLOYMENT STACK**

### **Option 1: Vercel + Railway (Recommended)**
- **Frontend**: Vercel (Free tier available)
- **Backend**: Railway (PostgreSQL included)
- **Database**: Railway PostgreSQL
- **Estimated Cost**: $5-20/month

### **Option 2: Netlify + Render**
- **Frontend**: Netlify (Free tier available)
- **Backend**: Render (Free tier available)
- **Database**: Render PostgreSQL
- **Estimated Cost**: $0-15/month

## 🔧 **STEP-BY-STEP DEPLOYMENT**

### **1. FRONTEND DEPLOYMENT (Vercel)**

```bash
# Install Vercel CLI
npm i -g vercel

# In your frontend directory
cd frontend
vercel

# Follow the prompts:
# - Connect to GitHub
# - Set build command: npm run build
# - Set output directory: dist
# - Set environment variables in Vercel dashboard
```

**Vercel Environment Variables:**
```
VITE_API_BASE_URL=https://your-backend-url.railway.app
VITE_ENVIRONMENT=production
```

### **2. BACKEND DEPLOYMENT (Railway)**

1. **Connect GitHub Repository**
   - Go to railway.app
   - Connect your GitHub repo
   - Select `/backend` as root directory

2. **Environment Variables in Railway:**
```
OPENAI_API_KEY=your_openai_key
GOOGLE_PLACES_API_KEY=your_google_key
GOOGLE_GEMINI_API_KEY=your_gemini_key
DATABASE_URL=${{ Postgres.DATABASE_URL }}
ENVIRONMENT=production
ALLOWED_ORIGINS=["https://your-vercel-app.vercel.app"]
```

3. **Railway Configuration Files:**

Create `railway.toml`:
```toml
[build]
builder = "nixpacks"

[deploy]
startCommand = "python setup_prod_db.py && gunicorn main:app --host 0.0.0.0 --port $PORT --workers 4"
```

### **3. DATABASE SETUP**

Railway automatically provides PostgreSQL. Update your `database.py`:

```python
import os
from sqlalchemy import create_engine

# Use Railway's DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

# Handle PostgreSQL URL format
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)
```

## 🔐 **PRODUCTION ENVIRONMENT SETUP**

### **Required API Keys:**
1. **OpenAI API Key** - For AI chatbot functionality
   - Sign up at: https://platform.openai.com/
   - Cost: ~$10-50/month depending on usage

2. **Google Places API Key** - For restaurant/place data
   - Get at: https://console.cloud.google.com/
   - Enable: Places API, Geocoding API
   - Cost: Usually free tier is sufficient

3. **Google Gemini API Key** - For AI features
   - Get at: https://makersuite.google.com/
   - Cost: Free tier available

## 📊 **POST-DEPLOYMENT CHECKLIST**

### **Immediate Testing:**
- [ ] Visit your deployed frontend URL
- [ ] Test chatbot functionality
- [ ] Test blog posting and viewing
- [ ] Test district queries (Kadikoy, Uskudar, etc.)
- [ ] Test responsive design on mobile
- [ ] Check all navigation links work

### **Performance & Monitoring:**
- [ ] Set up Vercel Analytics
- [ ] Configure error tracking (Sentry)
- [ ] Set up uptime monitoring
- [ ] Test site speed (PageSpeed Insights)

### **SEO & Content:**
- [ ] Add proper meta tags and descriptions
- [ ] Set up sitemap.xml
- [ ] Configure robots.txt
- [ ] Add favicon and app icons

### **Legal & Compliance:**
- [ ] Add privacy policy
- [ ] Add terms of service
- [ ] GDPR compliance if serving EU users
- [ ] Cookie consent if using analytics

## 💰 **ESTIMATED COSTS**

### **Free Tier (Possible):**
- Vercel: Free
- Railway: $5/month (hobby plan)
- APIs: Free tiers available
- **Total: ~$5/month**

### **Production Ready:**
- Vercel Pro: $20/month
- Railway Pro: $20/month  
- OpenAI API: $10-50/month
- **Total: ~$50-90/month**

## 🆘 **TROUBLESHOOTING**

### **Common Issues:**
1. **CORS Errors**: Update ALLOWED_ORIGINS in backend
2. **API Failures**: Verify environment variables
3. **Database Issues**: Check DATABASE_URL format
4. **Build Failures**: Verify all dependencies in package.json

### **Support Resources:**
- Railway Documentation: https://docs.railway.app/
- Vercel Documentation: https://vercel.com/docs
- FastAPI Deployment: https://fastapi.tiangolo.com/deployment/
