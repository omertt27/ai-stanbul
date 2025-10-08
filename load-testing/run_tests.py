#!/usr/bin/env python3
"""
AI Istanbul Load Testing Suite Runner

This script provides a convenient way to run all load tests and generate reports.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Test configurations
TESTS = {
    'load': {
        'script': 'api_load_test.py',
        'description': 'API Load Testing - Simulates realistic user load',
        'duration': '5 minutes'
    },
    'stress': {
        'script': 'stress_test.py', 
        'description': 'Stress Testing - High load simulation',
        'duration': '3 minutes'
    },
    'endurance': {
        'script': 'endurance_test.py',
        'description': 'Endurance Testing - Long-running stability test',
        'duration': '10 minutes'
    },
    'integration': {
        'script': 'integration_test.py',
        'description': 'Integration Testing - End-to-end workflows',
        'duration': '5 minutes'
    },
    'frontend': {
        'script': 'frontend_performance.py',
        'description': 'Frontend Performance - UI and interaction testing',
        'duration': '3 minutes'
    }
}

def print_banner():
    """Print the test suite banner."""
    print("=" * 60)
    print("🚀 AI ISTANBUL LOAD TESTING SUITE")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_test_info(test_name, test_config):
    """Print information about a test before running."""
    print(f"🧪 Running: {test_config['description']}")
    print(f"⏱️  Duration: {test_config['duration']}")
    print(f"📄 Script: {test_config['script']}")
    print("-" * 40)

def run_test(script_name, environment='local'):
    """Run a single test script."""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"❌ Error: Test script {script_name} not found!")
        return False
        
    try:
        # Run the test script
        result = subprocess.run([
            sys.executable, str(script_path), '--env', environment
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            if result.stdout:
                print(f"📊 Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            if result.stderr:
                print(f"💬 Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"💥 Exception running {script_name}: {str(e)}")
        return False

def generate_report():
    """Generate the comprehensive test report."""
    print("\n" + "=" * 60)
    print("📊 GENERATING COMPREHENSIVE REPORT")
    print("=" * 60)
    
    report_script = Path(__file__).parent / 'generate_report.py'
    
    if not report_script.exists():
        print("❌ Error: Report generator not found!")
        return False
        
    try:
        result = subprocess.run([
            sys.executable, str(report_script)
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ Report generated successfully")
            if result.stdout:
                print(f"📄 Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Report generation failed with return code {result.returncode}")
            if result.stderr:
                print(f"💬 Error: {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Report generation timed out")
        return False
    except Exception as e:
        print(f"💥 Exception generating report: {str(e)}")
        return False

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='AI Istanbul Load Testing Suite Runner')
    parser.add_argument('--tests', nargs='+', choices=list(TESTS.keys()) + ['all'], 
                       default=['all'], help='Tests to run (default: all)')
    parser.add_argument('--env', choices=['local', 'production'], default='local',
                       help='Environment to test against (default: local)')
    parser.add_argument('--no-report', action='store_true', 
                       help='Skip report generation')
    parser.add_argument('--list', action='store_true',
                       help='List available tests and exit')
    
    args = parser.parse_args()
    
    # List tests if requested
    if args.list:
        print("Available tests:")
        for test_name, test_config in TESTS.items():
            print(f"  {test_name:12} - {test_config['description']}")
        return
    
    # Determine which tests to run
    if 'all' in args.tests:
        tests_to_run = list(TESTS.keys())
    else:
        tests_to_run = args.tests
    
    print_banner()
    print(f"🎯 Environment: {args.env.upper()}")
    print(f"📝 Tests to run: {', '.join(tests_to_run)}")
    print()
    
    # Check if requirements are installed
    try:
        import requests
        import matplotlib
        import pandas
    except ImportError as e:
        print(f"❌ Missing dependencies! Please install requirements:")
        print(f"   pip install -r requirements.txt")
        print(f"   Missing: {str(e)}")
        return 1
    
    # Run each test
    results = {}
    start_time = datetime.now()
    
    for test_name in tests_to_run:
        if test_name not in TESTS:
            print(f"⚠️  Unknown test: {test_name}, skipping...")
            continue
            
        test_config = TESTS[test_name]
        print_test_info(test_name, test_config)
        
        success = run_test(test_config['script'], args.env)
        results[test_name] = success
        
        print()
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("=" * 60)
    print("📋 TEST EXECUTION SUMMARY")
    print("=" * 60)
    print(f"⏱️  Total Duration: {duration}")
    print(f"📊 Tests Run: {len(results)}")
    print(f"✅ Passed: {sum(results.values())}")
    print(f"❌ Failed: {len(results) - sum(results.values())}")
    print()
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:12} - {status}")
    
    # Generate report if requested
    if not args.no_report and any(results.values()):
        print()
        generate_report()
    
    # Return appropriate exit code
    return 0 if all(results.values()) else 1

if __name__ == '__main__':
    sys.exit(main())
