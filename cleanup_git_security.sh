#!/bin/bash

# AI Istanbul - Git Repository Cleanup Script
# This script removes sensitive files from git tracking

echo "🔒 Starting Git Repository Cleanup for AI Istanbul..."
echo "⚠️  This will remove sensitive files from git tracking"

# Navigate to project root
cd /Users/omer/Desktop/ai-stanbul

# Remove sensitive files from git tracking (but keep local copies)
echo "📝 Removing sensitive files from git tracking..."

# Environment files with API keys
git rm --cached .env 2>/dev/null || true
git rm --cached backend/.env 2>/dev/null || true
git rm --cached backend/.env.production 2>/dev/null || true

# Database files with user data
git rm --cached app.db 2>/dev/null || true
git rm --cached backend/app.db 2>/dev/null || true
git rm --cached backend/blog_analytics.db 2>/dev/null || true
git rm --cached backend/db/ai_stanbul.db 2>/dev/null || true

# Log files
git rm --cached ai_istanbul.log 2>/dev/null || true
git rm --cached backend/ai_istanbul.log 2>/dev/null || true
git rm --cached backend/server.log 2>/dev/null || true
git rm --cached backend/server_startup.log 2>/dev/null || true
git rm --cached backend/nohup.out 2>/dev/null || true

# Python cache files
git rm -r --cached backend/__pycache__/ 2>/dev/null || true
git rm -r --cached __pycache__/ 2>/dev/null || true

# Test results and coverage files
git rm --cached coverage.json 2>/dev/null || true
git rm -r --cached htmlcov/ 2>/dev/null || true
git rm --cached backend/*test_results*.json 2>/dev/null || true
git rm --cached backend/comprehensive_*_results_*.json 2>/dev/null || true
git rm --cached *test_results*.json 2>/dev/null || true
git rm --cached gdpr_security_report_*.json 2>/dev/null || true

# Credentials
git rm -r --cached credentials/ 2>/dev/null || true
git rm --cached backend/credentials/ 2>/dev/null || true

# Backup files
git rm --cached backend/main_backup.py 2>/dev/null || true

# Virtual environments
git rm -r --cached .venv/ 2>/dev/null || true
git rm -r --cached backend/.venv/ 2>/dev/null || true
git rm -r --cached venv/ 2>/dev/null || true
git rm -r --cached backend/venv/ 2>/dev/null || true

# Node modules
git rm -r --cached node_modules/ 2>/dev/null || true

# Uploads
git rm -r --cached uploads/ 2>/dev/null || true
git rm -r --cached backend/uploads/ 2>/dev/null || true

echo "✅ Sensitive files removed from git tracking"

# Clean git cache
echo "🧹 Cleaning git cache..."
git rm -r --cached . 2>/dev/null || true
git add .

echo "📋 Creating environment templates..."

# Create .env.template for root
cat > .env.template << 'EOF'
# AI Istanbul Environment Variables Template
# Copy this to .env and add your real API keys

# ==============================================
# GOOGLE APIs - Get from: https://console.cloud.google.com/
# ==============================================
GOOGLE_PLACES_API_KEY=your_google_places_api_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
GOOGLE_WEATHER_API_KEY=your_google_weather_api_key_here

# ==============================================
# OPENAI API - Get from: https://platform.openai.com/
# ==============================================
OPENAI_API_KEY=your_openai_api_key_here

# ==============================================
# GOOGLE ANALYTICS (Optional)
# ==============================================
GOOGLE_ANALYTICS_PROPERTY_ID=your_property_id
GOOGLE_ANALYTICS_SERVICE_ACCOUNT_PATH=./credentials/your-service-account.json

# ==============================================
# DATABASE (Optional - defaults to SQLite)
# ==============================================
DATABASE_URL=sqlite:///./app.db

# ==============================================
# REDIS (Optional - for caching)
# ==============================================
REDIS_URL=redis://localhost:6379

# ==============================================
# OTHER APIs (Optional)
# ==============================================
GEMINI_API_KEY=your_gemini_api_key_here
EOF

# Create backend/.env.template
cat > backend/.env.template << 'EOF'
# AI Istanbul Backend Environment Variables Template
# Copy this to .env and add your real API keys

DATABASE_URL=sqlite:///./app.db
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
GOOGLE_WEATHER_API_KEY=your_google_weather_api_key_here
GOOGLE_PLACES_API_KEY=your_google_places_api_key_here
GOOGLE_ANALYTICS_PROPERTY_ID=your_property_id
GOOGLE_ANALYTICS_SERVICE_ACCOUNT_PATH=./credentials/your-service-account.json
REDIS_URL=redis://localhost:6379
EOF

echo "📖 Creating credentials README..."

# Create credentials folder with README
mkdir -p credentials
cat > credentials/README.md << 'EOF'
# Credentials Directory

This directory is for storing sensitive credential files like:

- Google Analytics service account JSON files
- SSL certificates
- Other authentication files

**IMPORTANT**: This entire directory is ignored by git for security.

## Setup Instructions

1. Download your service account JSON file from Google Cloud Console
2. Place it in this directory
3. Update your .env file with the correct path

## Security Note

Never commit credential files to version control!
EOF

echo "🔍 Checking remaining tracked files..."
echo "Files still tracked by git:"
git ls-files | grep -E '\.(env|log|db|json)$|credentials|__pycache__|uploads|\.venv' || echo "✅ No sensitive files found in tracking"

echo "📊 Repository status:"
git status --porcelain | head -10

echo ""
echo "🎉 Git cleanup completed!"
echo ""
echo "📋 Next Steps:"
echo "1. Copy .env.template to .env and add your real API keys"
echo "2. Copy backend/.env.template to backend/.env and add your real API keys"
echo "3. Add your Google Analytics service account JSON to credentials/ folder"
echo "4. Commit the changes: git add . && git commit -m 'Security: Remove sensitive files and add .gitignore'"
echo ""
echo "⚠️  IMPORTANT: Make sure to test your application after updating .env files!"
