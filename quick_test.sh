#!/bin/bash
# Quick Start Script for Multi-Language Testing
# This script helps start both backend and frontend, then runs tests

set -e  # Exit on error

echo "🌍 AI Istanbul Multi-Language Testing - Quick Start"
echo "=================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if backend is running
check_backend() {
    echo "🔍 Checking if backend is running on port 8002..."
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend is running${NC}"
        return 0
    else
        echo -e "${RED}❌ Backend is not running${NC}"
        return 1
    fi
}

# Check if frontend is running
check_frontend() {
    echo "🔍 Checking if frontend is running on port 3000..."
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Frontend is running${NC}"
        return 0
    else
        echo -e "${RED}❌ Frontend is not running${NC}"
        return 1
    fi
}

# Start backend
start_backend() {
    echo ""
    echo "🚀 Starting backend on port 8002..."
    cd backend
    
    # Check if Python virtual environment exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Start backend in background
    echo "   Running: python main_pure_llm.py"
    nohup python main_pure_llm.py > ../backend.log 2>&1 &
    BACKEND_PID=$!
    echo "   Backend PID: $BACKEND_PID"
    
    # Wait for backend to start
    echo "   Waiting for backend to start..."
    sleep 5
    
    cd ..
    
    if check_backend; then
        echo -e "${GREEN}   ✅ Backend started successfully${NC}"
        echo "$BACKEND_PID" > .backend.pid
        return 0
    else
        echo -e "${RED}   ❌ Backend failed to start. Check backend.log for errors${NC}"
        return 1
    fi
}

# Start frontend
start_frontend() {
    echo ""
    echo "🚀 Starting frontend on port 3000..."
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "   📦 Installing dependencies..."
        npm install
    fi
    
    # Start frontend in background
    echo "   Running: npm run dev"
    nohup npm run dev > ../frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo "   Frontend PID: $FRONTEND_PID"
    
    # Wait for frontend to start
    echo "   Waiting for frontend to start..."
    sleep 10
    
    cd ..
    
    if check_frontend; then
        echo -e "${GREEN}   ✅ Frontend started successfully${NC}"
        echo "$FRONTEND_PID" > .frontend.pid
        return 0
    else
        echo -e "${YELLOW}   ⚠️  Frontend may still be starting. Check frontend.log${NC}"
        return 0
    fi
}

# Run automated tests
run_tests() {
    echo ""
    echo "🧪 Running automated multi-language tests..."
    echo ""
    python3 test_multilanguage.py
}

# Main menu
main_menu() {
    echo ""
    echo "What would you like to do?"
    echo ""
    echo "1) 🏥 Check system status"
    echo "2) 🚀 Start backend (port 8002)"
    echo "3) 🎨 Start frontend (port 3000)"
    echo "4) 🚀 Start both backend and frontend"
    echo "5) 🧪 Run automated tests"
    echo "6) 🌐 Open frontend in browser"
    echo "7) 📊 View test results"
    echo "8) 🛑 Stop all services"
    echo "9) 📝 View logs"
    echo "0) ❌ Exit"
    echo ""
    read -p "Select option (0-9): " choice
    
    case $choice in
        1)
            echo ""
            check_backend
            check_frontend
            main_menu
            ;;
        2)
            if check_backend; then
                echo -e "${YELLOW}Backend is already running${NC}"
            else
                start_backend
            fi
            main_menu
            ;;
        3)
            if check_frontend; then
                echo -e "${YELLOW}Frontend is already running${NC}"
            else
                start_frontend
            fi
            main_menu
            ;;
        4)
            if ! check_backend; then
                start_backend
            fi
            if ! check_frontend; then
                start_frontend
            fi
            echo ""
            echo -e "${GREEN}✅ Both services are running!${NC}"
            echo "   Backend: http://localhost:8002"
            echo "   Frontend: http://localhost:3000"
            main_menu
            ;;
        5)
            if ! check_backend; then
                echo -e "${RED}Backend must be running first!${NC}"
                read -p "Start backend now? (y/n): " start_it
                if [ "$start_it" = "y" ]; then
                    start_backend
                    run_tests
                fi
            else
                run_tests
            fi
            main_menu
            ;;
        6)
            echo ""
            echo "🌐 Opening frontend in browser..."
            if [[ "$OSTYPE" == "darwin"* ]]; then
                open http://localhost:3000
            elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
                xdg-open http://localhost:3000
            else
                echo "   Please open http://localhost:3000 in your browser"
            fi
            main_menu
            ;;
        7)
            echo ""
            if [ -f "MULTI_LANGUAGE_TEST_RESULTS.md" ]; then
                echo "📊 Test Results:"
                echo "------------------------------------------------"
                head -50 MULTI_LANGUAGE_TEST_RESULTS.md
                echo ""
                echo "Full results in: MULTI_LANGUAGE_TEST_RESULTS.md"
            else
                echo -e "${YELLOW}No test results found. Run tests first (option 5)${NC}"
            fi
            main_menu
            ;;
        8)
            echo ""
            echo "🛑 Stopping services..."
            if [ -f ".backend.pid" ]; then
                BACKEND_PID=$(cat .backend.pid)
                kill $BACKEND_PID 2>/dev/null && echo "   ✅ Backend stopped" || echo "   ⚠️  Backend already stopped"
                rm .backend.pid
            fi
            if [ -f ".frontend.pid" ]; then
                FRONTEND_PID=$(cat .frontend.pid)
                kill $FRONTEND_PID 2>/dev/null && echo "   ✅ Frontend stopped" || echo "   ⚠️  Frontend already stopped"
                rm .frontend.pid
            fi
            # Also kill any remaining processes on these ports
            lsof -ti:8002 | xargs kill -9 2>/dev/null || true
            lsof -ti:3000 | xargs kill -9 2>/dev/null || true
            echo -e "${GREEN}   All services stopped${NC}"
            main_menu
            ;;
        9)
            echo ""
            echo "📝 Recent Logs:"
            echo ""
            echo "--- Backend Log (last 20 lines) ---"
            if [ -f "backend.log" ]; then
                tail -20 backend.log
            else
                echo "No backend log found"
            fi
            echo ""
            echo "--- Frontend Log (last 20 lines) ---"
            if [ -f "frontend.log" ]; then
                tail -20 frontend.log
            else
                echo "No frontend log found"
            fi
            main_menu
            ;;
        0)
            echo ""
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo ""
            echo -e "${RED}Invalid option${NC}"
            main_menu
            ;;
    esac
}

# Start
echo ""
echo "📋 Pre-flight check..."
check_backend
check_frontend

main_menu
