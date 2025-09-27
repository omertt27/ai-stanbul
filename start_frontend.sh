#!/bin/bash
# AI Istanbul Frontend Server Startup Script
# Uses Vite development server for proper JavaScript module support

echo "🚀 Starting AI Istanbul Frontend Server..."
echo "📁 Serving from: frontend/"
echo "🌐 URL: http://localhost:3000/"

# Stop any existing server on port 3000
echo "🛑 Stopping existing servers on port 3000..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || echo "   No existing server found"

# Copy admin dashboard to frontend directory
echo "📋 Copying admin dashboard to frontend..."
cp unified_admin_dashboard.html frontend/ 2>/dev/null || echo "   Admin dashboard already copied"
cp unified_admin_dashboard_production_*.html frontend/ 2>/dev/null || echo "   Production files already copied"

# Start the Vite development server
echo "▶️  Starting Vite development server..."
cd frontend && NODE_ENV=development ./node_modules/.bin/vite --host --port 3000 --force

echo "✅ Frontend server stopped"
