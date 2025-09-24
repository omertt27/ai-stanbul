# 🚀 AI Istanbul - Deployment Checklist

## Pre-Deployment Checklist ✅

### Code Readiness
- [ ] All features tested and working locally
- [ ] Comprehensive test suite passing
- [ ] Query classification working correctly
- [ ] Like/dislike feedback system functional
- [ ] Session management working
- [ ] Admin dashboard accessible
- [ ] Environment variables properly configured

### Configuration Files
- [ ] `render.yaml` created for Render deployment
- [ ] `vercel.json` created for Vercel deployment
- [ ] `frontend/.env.production` configured
- [ ] `backend/requirements.txt` up to date
- [ ] `frontend/package.json` build scripts working

### API Keys and Credentials
- [ ] OpenAI API key ready (`OPENAI_API_KEY`)
- [ ] Google Maps API key ready (`GOOGLE_MAPS_API_KEY`)
- [ ] Google Places API key ready (`GOOGLE_PLACES_API_KEY`)
- [ ] Google Weather API key ready (`GOOGLE_WEATHER_API_KEY`)
- [ ] Database connection string ready (`DATABASE_URL`)

## Deployment Steps

### Step 1: Backend Deployment (Render)
- [ ] Push code to GitHub repository
- [ ] Connect GitHub repo to Render
- [ ] Create PostgreSQL database on Render
- [ ] Set environment variables in Render dashboard
- [ ] Deploy backend service
- [ ] Test health endpoint: `/health`
- [ ] Verify API endpoints work

### Step 2: Frontend Deployment (Vercel)
- [ ] Update `VITE_API_URL` with Render backend URL
- [ ] Push updated frontend code to GitHub
- [ ] Connect GitHub repo to Vercel
- [ ] Set environment variables in Vercel
- [ ] Deploy frontend
- [ ] Test frontend loads correctly
- [ ] Verify API calls work from frontend

### Step 3: Integration Testing
- [ ] Chat functionality works end-to-end
- [ ] All query types properly classified:
  - [ ] Transportation queries
  - [ ] Museum queries  
  - [ ] Restaurant queries
  - [ ] General tips/advice queries
- [ ] Like/dislike buttons functional
- [ ] Session saving works
- [ ] Admin dashboard accessible
- [ ] Database queries working
- [ ] CORS properly configured

## Post-Deployment Verification

### Functional Tests
- [ ] Ask transportation question: "How to get to Galata Tower?"
- [ ] Ask museum question: "Tell me about Topkapi Palace"
- [ ] Ask restaurant question: "Best restaurants in Beyoglu"
- [ ] Ask general tip: "Is tap water safe to drink in Istanbul?"
- [ ] Test like button on responses
- [ ] Test session saving after liking
- [ ] Check admin dashboard for saved sessions

### Performance Tests  
- [ ] Response times acceptable (<3 seconds)
- [ ] No 500/404 errors in logs
- [ ] Health check returning 200 OK
- [ ] Database queries executing properly

### URLs to Test
- [ ] Backend health: `https://your-app.onrender.com/health`
- [ ] Frontend: `https://your-app.vercel.app`
- [ ] Chat API: `https://your-app.onrender.com/chat`
- [ ] Admin dashboard: Access through frontend

## Environment Variables Checklist

### Backend (Render)
```
✅ OPENAI_API_KEY=sk-proj-...
✅ DATABASE_URL=postgresql://...
✅ GOOGLE_MAPS_API_KEY=AIza...
✅ GOOGLE_PLACES_API_KEY=AIza...
✅ GOOGLE_WEATHER_API_KEY=AIza...
✅ GEMINI_API_KEY=... (optional)
✅ GOOGLE_ANALYTICS_PROPERTY_ID=...
```

### Frontend (Vercel)
```
✅ VITE_API_URL=https://your-backend.onrender.com
✅ VITE_ENVIRONMENT=production
```

## Success Criteria ✅

The deployment is successful when:
- [ ] Backend health check returns 200 OK
- [ ] Frontend loads without console errors
- [ ] Chat responds to all query types correctly
- [ ] Transportation queries prioritized properly
- [ ] Museum queries get detailed responses
- [ ] Restaurant queries work as expected
- [ ] General tips routed to GPT responses
- [ ] Like/dislike functionality works
- [ ] Sessions are saved when messages are liked
- [ ] Admin dashboard shows saved sessions
- [ ] No CORS errors in browser console
- [ ] Database connections stable

## Rollback Plan 🔄

If deployment fails:
1. **Backend Issues**: 
   - Check Render logs for errors
   - Verify environment variables
   - Test database connectivity
   - Roll back to previous working version

2. **Frontend Issues**:
   - Check Vercel build logs
   - Verify API URL configuration
   - Test locally with production API
   - Roll back to previous deployment

3. **Integration Issues**:
   - Check CORS configuration
   - Verify API endpoints
   - Test authentication flow
   - Check network connectivity

## Monitoring and Maintenance 📊

Post-deployment monitoring:
- [ ] Set up Render service monitoring
- [ ] Monitor Vercel deployment status
- [ ] Check error rates in logs
- [ ] Monitor API response times
- [ ] Review user feedback through admin dashboard
- [ ] Monitor database performance

## Next Steps 🎯

After successful deployment:
1. **Custom Domain** (optional): Configure custom domain names
2. **Analytics**: Set up detailed analytics tracking
3. **Performance**: Monitor and optimize response times
4. **Security**: Review and enhance security measures
5. **Scaling**: Plan for traffic scaling if needed

---

**Deployment Date**: _________________
**Deployed By**: _____________________ 
**Backend URL**: ____________________
**Frontend URL**: ____________________
**Status**: __________________________ 
