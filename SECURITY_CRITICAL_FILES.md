# AI Istanbul - Critical Files Protection Guide
# ==============================================
# This file documents all critical files that must be protected
# Last Updated: September 27, 2025

## NEVER COMMIT OR EXPOSE THESE FILES:

### 1. Environment Variables & API Keys
- .env (contains Google Maps API key, OpenAI key, backend URLs)
- backend/.env (contains OpenAI API key, Redis URL, database credentials)
- frontend/.env (contains Google Maps API key, backend API URL)

### 2. Database Files
- app.db (main application database)
- blog_analytics.db (blog analytics data)
- Any *.db, *.sqlite, *.sqlite3 files

### 3. Credentials & Authentication
- credentials/ directory (Google service account keys)
- backend/credentials/ directory
- Any *.json files with API keys
- SSL certificates (*.pem, *.key, *.crt)

### 4. Logs & Debug Files
- ai_istanbul.log (application logs may contain sensitive data)
- server.log, server_startup.log
- Any *.log files

### 5. Test Results & Coverage Data (CRITICAL - Contains API Responses & System Data)
- All test_*.py, *_test.py, *_tester.py files
- test-*.js, *-test.js, *_test.js files
- tests/ directory and all subdirectories
- *test_results*.json (may contain API responses with sensitive data)
- *test_results*.txt (system outputs and logs)
- ai_istanbul_test_*.json (contains actual API call results)
- ai_istanbul_final_test_*.json (production test data)
- comprehensive_test_*.json (detailed system analysis)
- restaurant_google_maps_test_*.json (Google Places API responses)
- gdpr_security_report_*.json (security audit data)
- coverage.json, coverage.xml, .coverage files
- Test HTML files: test-*.html, *-test.html, mobile_test.html
- Frontend test files: frontend/test-*.js, frontend/dist/*test*.html

### 6. Source Maps & Build Files
- *.map files (expose source code structure)
- frontend/dist/ (built frontend with embedded secrets)
- frontend/build/

### 7. Development & IDE Files
- .vscode/ (may contain workspace-specific configurations)
- .idea/ (IntelliJ/PyCharm project files)

### 8. Deployment Scripts
- deploy-production.sh (contains deployment secrets)
- production_deploy.sh

### 9. AI Models & Training Data
- models/ directory
- training_data/ directory
- *.pkl, *.model, *.weights files

### 10. Personal & Business Data
- user_data/ directory
- financial_data/ directory
- contracts/ directory
- invoices/ directory

## SECURITY MEASURES IMPLEMENTED:

### Frontend Protection:
1. Anti-copy protection (websiteProtection.js)
   - Disables right-click, text selection, keyboard shortcuts
   - Detects developer tools and shows warnings
   - Adds watermarks and copyright notices
   - Prevents printing and screenshots

2. CSS Protection (anti-copy.css)
   - Disables text selection globally
   - Prevents image dragging
   - Adds copyright watermarks
   - Hides content when printing

3. Legal Protection:
   - Terms of Service page (/terms)
   - Copyright notices throughout the app
   - GDPR compliance page

### Backend Protection:
1. Environment variable security
2. Rate limiting and API protection
3. CORS configuration
4. Input validation and sanitization

### Repository Protection:
1. Comprehensive .gitignore
2. Git history cleanup for exposed secrets
3. API key rotation when compromised

## IMMEDIATE ACTIONS IF SECRETS ARE EXPOSED:

1. **Rotate all API keys immediately:**
   - OpenAI API key
   - Google Maps API key
   - Any database credentials

2. **Remove from git history:**
   ```bash
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch <file>' \
   --prune-empty --tag-name-filter cat -- --all
   ```

3. **Update .gitignore and commit:**
   ```bash
   git add .gitignore
   git commit -m "Update gitignore for security"
   ```

4. **Force push to remote:**
   ```bash
   git push origin --force --all
   ```

## MONITORING:

- Regularly audit git history for exposed secrets
- Monitor application logs for suspicious activity
- Check for unauthorized API key usage
- Review access logs for scraping attempts

## CONTACT:

For security issues: security@ai-istanbul.com
For legal matters: legal@ai-istanbul.com

---
© AI Istanbul 2025 - This document contains sensitive security information
