#!/bin/bash
#
# LLM API Server Startup Script for Google Cloud VM
# ==================================================
#
# This script automates the startup process for the Llama 3.1 8B API server
# on the Google Cloud VM (n4-standard-8, CPU-only).
#
# Usage on VM:
#   chmod +x start_llm_server.sh
#   ./start_llm_server.sh
#
# Or run directly:
#   bash start_llm_server.sh
#

set -e  # Exit on error

echo "============================================================"
echo "🚀 AI Istanbul - LLM API Server Startup"
echo "============================================================"
echo ""

# Configuration
PROJECT_DIR="/home/$(whoami)/ai-istanbul"
VENV_DIR="$PROJECT_DIR/venv"
LOG_FILE="/tmp/llm_api_server.log"
PID_FILE="/tmp/llm_api_server.pid"
PORT=8000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if server is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        print_warning "Server is already running with PID $OLD_PID"
        read -p "Do you want to restart it? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Stopping existing server..."
            kill "$OLD_PID" 2>/dev/null || true
            sleep 2
        else
            print_status "Exiting. To stop the server, run: kill $OLD_PID"
            exit 0
        fi
    fi
fi

# Check if port is in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "Port $PORT is already in use"
    PID_ON_PORT=$(lsof -ti:$PORT)
    print_status "Process using port: $PID_ON_PORT"
    read -p "Kill process and continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill -9 "$PID_ON_PORT" 2>/dev/null || true
        sleep 2
    else
        print_error "Cannot start server while port is in use"
        exit 1
    fi
fi

# Create project directory if it doesn't exist
if [ ! -d "$PROJECT_DIR" ]; then
    print_warning "Project directory not found: $PROJECT_DIR"
    read -p "Enter project directory path: " PROJECT_DIR
    if [ ! -d "$PROJECT_DIR" ]; then
        print_error "Directory does not exist: $PROJECT_DIR"
        exit 1
    fi
fi

print_status "Project directory: $PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    print_warning "Virtual environment not found at $VENV_DIR"
    print_status "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
print_success "Virtual environment activated"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
print_status "Python version: $PYTHON_VERSION"

# Install/upgrade required packages
print_status "Checking dependencies..."
pip install --upgrade pip -q

REQUIRED_PACKAGES=(
    "transformers>=4.36.0"
    "torch>=2.1.0"
    "fastapi>=0.104.0"
    "uvicorn>=0.24.0"
    "psutil>=5.9.0"
    "pydantic>=2.0.0"
)

print_status "Installing required packages..."
for package in "${REQUIRED_PACKAGES[@]}"; do
    print_status "  - $package"
    pip install "$package" -q
done
print_success "Dependencies installed"

# Check system resources
print_status "Checking system resources..."
CPU_COUNT=$(nproc)
TOTAL_RAM=$(free -h | awk '/^Mem:/ {print $2}')
AVAILABLE_RAM=$(free -h | awk '/^Mem:/ {print $7}')

echo ""
echo "💻 System Resources:"
echo "   - CPUs: $CPU_COUNT"
echo "   - Total RAM: $TOTAL_RAM"
echo "   - Available RAM: $AVAILABLE_RAM"
echo ""

# Check if we have enough RAM (at least 20GB recommended)
RAM_GB=$(free -g | awk '/^Mem:/ {print $2}')
if [ "$RAM_GB" -lt 20 ]; then
    print_warning "Low RAM detected (${RAM_GB}GB). Recommended: 32GB+"
    print_warning "Model loading may fail or be very slow."
fi

# Check if model file exists (cached)
HF_CACHE_DIR="$HOME/.cache/huggingface/hub"
MODEL_NAME="models--meta-llama--Meta-Llama-3.1-8B"
MODEL_CACHE_PATH="$HF_CACHE_DIR/$MODEL_NAME"

if [ -d "$MODEL_CACHE_PATH" ]; then
    print_success "Model cache found - loading will be faster"
else
    print_warning "Model not cached - first load will download ~16GB"
    print_status "This may take 10-30 minutes depending on connection"
fi

# Change to project directory
cd "$PROJECT_DIR"
print_status "Changed to project directory: $(pwd)"

# Check if llm_api_server.py exists
if [ ! -f "llm_api_server.py" ]; then
    print_error "llm_api_server.py not found in $PROJECT_DIR"
    print_error "Please ensure the server file is uploaded to the VM"
    exit 1
fi

print_success "Server file found: llm_api_server.py"

# Clear old log file
> "$LOG_FILE"
print_status "Log file: $LOG_FILE"

# Ask user for startup mode
echo ""
print_status "Server startup options:"
echo "  1. Foreground (see logs in terminal, Ctrl+C to stop)"
echo "  2. Background (runs as daemon, use 'kill PID' to stop)"
echo ""
read -p "Select option (1 or 2): " -n 1 -r STARTUP_MODE
echo ""

if [ "$STARTUP_MODE" = "2" ]; then
    # Background mode
    print_status "Starting server in background mode..."
    
    nohup python llm_api_server.py > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$PID_FILE"
    
    print_success "Server started in background"
    print_status "PID: $SERVER_PID"
    print_status "Log file: $LOG_FILE"
    echo ""
    echo "📋 Useful commands:"
    echo "   - View logs: tail -f $LOG_FILE"
    echo "   - Check status: ps -p $SERVER_PID"
    echo "   - Stop server: kill $SERVER_PID"
    echo "   - Health check: curl http://localhost:$PORT/health"
    echo ""
    
    # Wait a bit and check if server is still running
    sleep 5
    if ps -p "$SERVER_PID" > /dev/null 2>&1; then
        print_success "Server is running (PID: $SERVER_PID)"
        print_status "Waiting for model to load (this may take 2-5 minutes)..."
        print_status "Monitor progress: tail -f $LOG_FILE"
    else
        print_error "Server failed to start. Check logs: $LOG_FILE"
        exit 1
    fi
else
    # Foreground mode
    print_status "Starting server in foreground mode..."
    print_status "Press Ctrl+C to stop the server"
    echo ""
    echo "============================================================"
    echo ""
    
    python llm_api_server.py 2>&1 | tee "$LOG_FILE"
fi

echo ""
print_success "Server startup complete!"
