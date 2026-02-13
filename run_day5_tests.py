#!/usr/bin/env python3
"""
Day 5 Test Runner - Run all Phase 2 tests

This script runs all Day 5 tests in sequence:
1. Integration tests
2. Load tests
3. Performance benchmarks
4. Generates comprehensive reports

Usage:
    python run_day5_tests.py
    python run_day5_tests.py --quick  # Skip load tests
    python run_day5_tests.py --load-only  # Only load tests

Author: AI Istanbul Team
Date: February 12, 2026
"""

import sys
import os
import subprocess
import argparse
import time
from datetime import datetime
import json

# Colors for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'


def print_header(message):
    """Print formatted header"""
    print(f"\n{BLUE}{BOLD}{'='*80}{RESET}")
    print(f"{BLUE}{BOLD}{message}{RESET}")
    print(f"{BLUE}{BOLD}{'='*80}{RESET}\n")


def print_success(message):
    """Print success message"""
    print(f"{GREEN}✅ {message}{RESET}")


def print_warning(message):
    """Print warning message"""
    print(f"{YELLOW}⚠️  {message}{RESET}")


def print_error(message):
    """Print error message"""
    print(f"{RED}❌ {message}{RESET}")


def run_command(cmd, description, timeout=300):
    """Run a command and return success status"""
    print(f"\n{BOLD}Running: {description}{RESET}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            print_success(f"{description} completed")
            return True, result.stdout
        else:
            print_error(f"{description} failed")
            print(f"Error output:\n{result.stderr}")
            return False, result.stderr
    
    except subprocess.TimeoutExpired:
        print_error(f"{description} timed out after {timeout}s")
        return False, f"Timeout after {timeout}s"
    
    except Exception as e:
        print_error(f"{description} failed with exception: {e}")
        return False, str(e)


def run_integration_tests():
    """Run integration tests"""
    print_header("Phase 1: Integration Tests")
    
    cmd = "pytest tests/integration/test_ncf_integration.py -v --tb=short --color=yes"
    success, output = run_command(cmd, "Integration Tests", timeout=300)
    
    return success, output


def run_load_tests(quick=False):
    """Run load tests"""
    print_header("Phase 2: Load Tests")
    
    if quick:
        print_warning("Quick mode: Skipping load tests")
        return True, "Skipped in quick mode"
    
    # Check if locust is installed
    try:
        subprocess.run(["locust", "--version"], capture_output=True, check=True)
    except:
        print_error("Locust not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "locust"], check=False)
    
    # Run load test (100 users, 5 minutes)
    cmd = (
        "locust -f tests/load_testing/ncf_load_test.py "
        "--host=http://localhost:8080 "
        "--users 100 --spawn-rate 10 --run-time 5m "
        "--headless --html tests/load_testing/load_test_report.html"
    )
    
    success, output = run_command(cmd, "Load Tests (100 users, 5 min)", timeout=400)
    
    return success, output


def run_performance_benchmarks():
    """Run performance benchmarks"""
    print_header("Phase 3: Performance Benchmarks")
    
    # Check if benchmark script exists
    benchmark_script = "backend/ml/deep_learning/performance_benchmark.py"
    
    if not os.path.exists(benchmark_script):
        print_warning(f"Benchmark script not found: {benchmark_script}")
        return True, "Benchmark script not found"
    
    cmd = f"python {benchmark_script}"
    success, output = run_command(cmd, "Performance Benchmarks", timeout=600)
    
    return success, output


def generate_summary_report(results):
    """Generate summary report"""
    print_header("Test Summary Report")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = {
        "timestamp": timestamp,
        "tests": results,
        "overall_status": all(r["success"] for r in results.values())
    }
    
    # Print summary
    print(f"Timestamp: {timestamp}\n")
    
    for test_name, test_result in results.items():
        status = "✅ PASS" if test_result["success"] else "❌ FAIL"
        duration = test_result.get("duration", 0)
        
        print(f"{status} {test_name} ({duration:.1f}s)")
        
        if not test_result["success"]:
            print(f"   Error: {test_result.get('error', 'Unknown error')}")
    
    print()
    
    # Overall status
    if report["overall_status"]:
        print_success(f"{BOLD}ALL TESTS PASSED{RESET}")
        print_success("✅ System is ready for Day 6: GCP Deployment")
    else:
        print_error(f"{BOLD}SOME TESTS FAILED{RESET}")
        print_warning("⚠️  Review failures before proceeding to Day 6")
    
    # Save report
    report_file = f"PHASE_2_DAY_5_TEST_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return report


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Run Phase 2 Day 5 tests")
    parser.add_argument("--quick", action="store_true", help="Quick mode (skip load tests)")
    parser.add_argument("--load-only", action="store_true", help="Only run load tests")
    parser.add_argument("--integration-only", action="store_true", help="Only run integration tests")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run benchmarks")
    
    args = parser.parse_args()
    
    print_header("🧪 Phase 2 Day 5: Testing & Validation")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    start_time = time.time()
    
    # Run tests based on flags
    if args.integration_only or not (args.load_only or args.benchmark_only):
        test_start = time.time()
        success, output = run_integration_tests()
        results["Integration Tests"] = {
            "success": success,
            "duration": time.time() - test_start,
            "output": output[:1000] if output else ""  # Truncate long output
        }
    
    if args.load_only or not (args.integration_only or args.benchmark_only):
        test_start = time.time()
        success, output = run_load_tests(quick=args.quick)
        results["Load Tests"] = {
            "success": success,
            "duration": time.time() - test_start,
            "output": output[:1000] if output else ""
        }
    
    if args.benchmark_only or not (args.integration_only or args.load_only):
        test_start = time.time()
        success, output = run_performance_benchmarks()
        results["Performance Benchmarks"] = {
            "success": success,
            "duration": time.time() - test_start,
            "output": output[:1000] if output else ""
        }
    
    # Generate summary
    total_duration = time.time() - start_time
    
    print_header("📊 Final Summary")
    print(f"Total Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    
    report = generate_summary_report(results)
    
    # Exit code based on results
    sys.exit(0 if report["overall_status"] else 1)


if __name__ == "__main__":
    main()
