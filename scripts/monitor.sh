#!/bin/bash
#
# Quick Backend Monitoring Script
# Provides easy shortcuts for common monitoring tasks
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

show_help() {
    cat << EOF
${BLUE}AI Istanbul Backend Monitoring${NC}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

${GREEN}Quick Commands:${NC}

  ./scripts/monitor.sh                  Show current dashboard
  ./scripts/monitor.sh watch            Watch mode (auto-refresh)
  ./scripts/monitor.sh report           Generate detailed report
  ./scripts/monitor.sh logs             Tail recent logs
  ./scripts/monitor.sh follow           Follow logs in real-time

${GREEN}Advanced:${NC}

  ./scripts/monitor.sh watch 5          Watch with 5s refresh
  ./scripts/monitor.sh report output.txt  Save report to file
  ./scripts/monitor.sh logs 50          Show last 50 log entries

${GREEN}Multiple Terminals Setup:${NC}

  Terminal 1: cd backend && uvicorn main:app --reload
  Terminal 2: ./scripts/monitor.sh watch
  Terminal 3: python test_all_query_types.py

EOF
}

case "${1:-dashboard}" in
    help|-h|--help)
        show_help
        ;;
    
    dashboard|status)
        echo -e "${BLUE}📊 Backend Dashboard${NC}"
        python scripts/monitor_backend_performance.py
        ;;
    
    watch|-w)
        echo -e "${BLUE}🔄 Watch Mode - Press Ctrl+C to exit${NC}"
        if [ -n "$2" ]; then
            python scripts/monitor_backend_performance.py --watch --interval "$2"
        else
            python scripts/monitor_backend_performance.py --watch
        fi
        ;;
    
    report|-r)
        echo -e "${BLUE}📝 Generating Report${NC}"
        if [ -n "$2" ]; then
            python scripts/monitor_backend_performance.py --report --output "$2"
        else
            python scripts/monitor_backend_performance.py --report
        fi
        ;;
    
    logs|tail|-t)
        echo -e "${BLUE}📜 Recent Logs${NC}"
        if [ -n "$2" ]; then
            python scripts/monitor_backend_performance.py --tail "$2"
        else
            python scripts/monitor_backend_performance.py --tail 20
        fi
        ;;
    
    follow|-f)
        echo -e "${BLUE}📡 Following Logs - Press Ctrl+C to exit${NC}"
        python scripts/monitor_backend_performance.py --tail --follow
        ;;
    
    *)
        echo -e "${YELLOW}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
