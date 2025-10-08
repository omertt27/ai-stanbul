# AI Istanbul Load Testing Suite Makefile

.PHONY: help setup install test test-load test-stress test-endurance test-integration test-frontend test-all report clean

# Default target
help:
	@echo "🚀 AI Istanbul Load Testing Suite"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make setup         - Install dependencies and setup environment"
	@echo "  make install       - Install Python dependencies only"
	@echo "  make test          - Run all tests (alias for test-all)"
	@echo "  make test-all      - Run all load tests"
	@echo "  make test-load     - Run API load tests only"
	@echo "  make test-stress   - Run stress tests only"
	@echo "  make test-endurance - Run endurance tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-frontend - Run frontend performance tests only"
	@echo "  make test-prod     - Run all tests against production"
	@echo "  make report        - Generate HTML report from results"
	@echo "  make clean         - Clean test results and temporary files"
	@echo "  make status        - Show testing environment status"
	@echo ""
	@echo "Environment variables:"
	@echo "  ENV=production     - Test against production environment"
	@echo "  TESTS=load,stress  - Run specific tests only"

# Setup and installation
setup:
	@echo "🔧 Setting up load testing environment..."
	python3 setup.py

install:
	@echo "📦 Installing dependencies..."
	pip3 install -r requirements.txt
	python3 -m playwright install

# Test execution
test: test-all

test-all:
	@echo "🧪 Running all load tests..."
	python3 run_tests.py --tests all

test-load:
	@echo "📊 Running API load tests..."
	python3 run_tests.py --tests load

test-stress:
	@echo "🔥 Running stress tests..."
	python3 run_tests.py --tests stress

test-endurance:
	@echo "⏰ Running endurance tests..."
	python3 run_tests.py --tests endurance

test-integration:
	@echo "🔗 Running integration tests..."
	python3 run_tests.py --tests integration

test-frontend:
	@echo "🌐 Running frontend performance tests..."
	python3 run_tests.py --tests frontend

test-prod:
	@echo "🚀 Running tests against production..."
	python3 run_tests.py --env production

# Custom test combinations
test-quick:
	@echo "⚡ Running quick test suite..."
	python3 run_tests.py --tests load integration

test-backend:
	@echo "🖥️ Running backend-focused tests..."
	python3 run_tests.py --tests load stress endurance

# Reporting
report:
	@echo "📊 Generating comprehensive report..."
	python3 generate_report.py

# Utility commands
clean:
	@echo "🧹 Cleaning test results and temporary files..."
	rm -rf results/*.json
	rm -rf results/*.html
	rm -rf results/*.png
	rm -rf __pycache__/
	rm -rf *.pyc
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

status:
	@echo "📋 Load Testing Environment Status"
	@echo "=================================="
	@echo "Python version: $$(python3 --version)"
	@echo "Pip packages:"
	@python3 -c "import requests, matplotlib, pandas, locust, psutil; print('✅ Core dependencies installed')" 2>/dev/null || echo "❌ Some dependencies missing"
	@python3 -c "import playwright; print('✅ Playwright installed')" 2>/dev/null || echo "❌ Playwright not installed"
	@echo "Results directory: $$(ls -la results/ 2>/dev/null | wc -l) files"
	@echo "Recent test results:"
	@ls -lt results/*.json 2>/dev/null | head -5 || echo "  No recent results found"

# Development helpers
dev-server:
	@echo "🖥️ Starting development server..."
	cd .. && npm run dev

# Validation
validate:
	@echo "🔍 Validating test configuration..."
	python3 -c "import config; print('✅ Configuration valid')"
	python3 -c "import importlib; [importlib.import_module(f) for f in ['api_load_test', 'stress_test', 'endurance_test', 'integration_test', 'frontend_performance']]; print('✅ All test modules importable')"

# List available tests
list:
	@echo "📝 Available test modules:"
	python3 run_tests.py --list
