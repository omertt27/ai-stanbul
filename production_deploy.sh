#!/bin/bash

# 🚀 AIstanbul Production Deployment Script
# This script helps deploy the chatbot to production

set -e  # Exit on any error

echo "🚀 Starting AIstanbul Production Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if required files exist
check_requirements() {
    echo -e "${BLUE}📋 Checking requirements...${NC}"
    
    if [ ! -f "backend/.env" ]; then
        echo -e "${RED}❌ backend/.env file not found!${NC}"
        echo "Please copy backend/.env.production to backend/.env and fill in your values"
        exit 1
    fi
    
    if [ ! -f "frontend/.env.production" ]; then
        echo -e "${RED}❌ frontend/.env.production file not found!${NC}"
        echo "Please copy frontend/.env.production to frontend/.env.production and fill in your values"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Requirements check passed${NC}"
}

# Install dependencies
install_dependencies() {
    echo -e "${BLUE}📦 Installing dependencies...${NC}"
    
    # Backend dependencies
    echo "Installing backend dependencies..."
    cd backend
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo -e "${RED}❌ requirements.txt not found in backend directory${NC}"
        exit 1
    fi
    cd ..
    
    # Frontend dependencies
    echo "Installing frontend dependencies..."
    cd frontend
    if [ -f "package.json" ]; then
        npm install
    else
        echo -e "${RED}❌ package.json not found in frontend directory${NC}"
        exit 1
    fi
    cd ..
    
    echo -e "${GREEN}✅ Dependencies installed${NC}"
}

# Set up database
setup_database() {
    echo -e "${BLUE}🗄️ Setting up database...${NC}"
    
    cd backend
    echo "Initializing database..."
    python init_db.py
    cd ..
    
    echo -e "${GREEN}✅ Database setup completed${NC}"
}

# Build frontend
build_frontend() {
    echo -e "${BLUE}🏗️ Building frontend...${NC}"
    
    cd frontend
    npm run build
    cd ..
    
    if [ -d "frontend/dist" ]; then
        echo -e "${GREEN}✅ Frontend build completed${NC}"
    else
        echo -e "${RED}❌ Frontend build failed${NC}"
        exit 1
    fi
}

# Start services
start_services() {
    echo -e "${BLUE}🚀 Starting services...${NC}"
    
    # Check if services are already running
    if pgrep -f "uvicorn main:app" > /dev/null; then
        echo -e "${YELLOW}⚠️ Backend is already running. Stopping...${NC}"
        pkill -f "uvicorn main:app"
        sleep 2
    fi
    
    # Start backend
    echo "Starting backend server..."
    cd backend
    nohup uvicorn main:app --host 0.0.0.0 --port 8001 --workers 2 > ../backend.log 2>&1 &
    BACKEND_PID=$!
    cd ..
    
    # Wait a moment and check if backend started
    sleep 5
    if ps -p $BACKEND_PID > /dev/null; then
        echo -e "${GREEN}✅ Backend started successfully (PID: $BACKEND_PID)${NC}"
    else
        echo -e "${RED}❌ Backend failed to start${NC}"
        exit 1
    fi
    
    # Frontend is served by web server (nginx/apache)
    echo -e "${YELLOW}📝 Frontend built and ready to be served by web server${NC}"
    echo -e "${YELLOW}   Deploy the 'frontend/dist' folder to your web server${NC}"
}

# Health check
health_check() {
    echo -e "${BLUE}🏥 Performing health check...${NC}"
    
    # Check backend health
    echo "Checking backend health..."
    sleep 5  # Give backend time to fully start
    
    if curl -f http://localhost:8001/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend health check passed${NC}"
    else
        echo -e "${RED}❌ Backend health check failed${NC}"
        echo "Check backend.log for errors"
        exit 1
    fi
    
    # Test AI endpoint
    echo "Testing AI endpoint..."
    if curl -f -X POST http://localhost:8001/ai \
        -H "Content-Type: application/json" \
        -d '{"message":"Hello"}' > /dev/null 2>&1; then
        echo -e "${GREEN}✅ AI endpoint test passed${NC}"
    else
        echo -e "${YELLOW}⚠️ AI endpoint test failed (check API keys)${NC}"
    fi
}

# Main deployment flow
main() {
    echo -e "${GREEN}🇹🇷 AIstanbul Chatbot Production Deployment${NC}"
    echo "=================================================="
    
    check_requirements
    install_dependencies
    setup_database
    build_frontend
    start_services
    health_check
    
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}📋 Next Steps:${NC}"
    echo "1. Configure your web server (nginx/apache) to serve frontend/dist"
    echo "2. Set up SSL certificate for HTTPS"
    echo "3. Configure your domain DNS"
    echo "4. Update CORS origins in backend/.env"
    echo "5. Monitor logs: tail -f backend.log"
    echo ""
    echo -e "${BLUE}🔗 Service URLs:${NC}"
    echo "Backend API: http://localhost:8001"
    echo "Backend Health: http://localhost:8001/health"
    echo "Frontend: Serve frontend/dist with your web server"
    echo ""
    echo -e "${BLUE}📊 Monitoring:${NC}"
    echo "Backend logs: tail -f backend.log"
    echo "Backend process: ps aux | grep uvicorn"
    echo ""
    echo -e "${GREEN}✅ Your AIstanbul chatbot is now ready for real users!${NC}"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "start")
        start_services
        ;;
    "stop")
        echo "Stopping services..."
        pkill -f "uvicorn main:app" || echo "Backend not running"
        echo "Services stopped"
        ;;
    "health")
        health_check
        ;;
    "logs")
        tail -f backend.log
        ;;
    *)
        echo "Usage: $0 [deploy|start|stop|health|logs]"
        echo "  deploy: Full deployment (default)"
        echo "  start:  Start services only"
        echo "  stop:   Stop services"
        echo "  health: Health check"
        echo "  logs:   Show backend logs"
        exit 1
        ;;
esac
