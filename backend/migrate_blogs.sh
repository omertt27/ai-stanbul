#!/bin/bash
# Quick script to migrate blog posts to Render PostgreSQL

echo "📦 Blog Posts Migration to Render PostgreSQL"
echo "=============================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📝 This will migrate blog posts from blog_posts.json to your Render PostgreSQL database"
echo ""

# Option 1: Use DATABASE_URL from Render environment
echo "Option 1: Use DATABASE_URL from Render dashboard"
echo "Go to: https://dashboard.render.com → aistanbul_postgre → Internal Database URL"
echo ""
echo "Option 2: Enter manually"
echo ""

# Prompt for database URL
read -p "Enter Render PostgreSQL URL (or press Enter to use local): " RENDER_DB_URL

if [ -z "$RENDER_DB_URL" ]; then
    echo "⚠️  No URL provided. Using DATABASE_URL from .env file..."
    export DATABASE_URL=$(grep "^DATABASE_URL=" .env | cut -d '=' -f2-)
else
    export DATABASE_URL="$RENDER_DB_URL"
fi

echo ""
echo "📊 Database: ${DATABASE_URL:0:50}..."
echo ""
echo "🚀 Starting migration..."
echo ""

# Run migration
python3 migrate_blog_to_postgres.py

echo ""
echo "✅ Done!"
