#!/usr/bin/env python3
"""
Test runner script for the concurrent RAG optimization system.
Runs all test suites with proper configuration and reporting.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    print(f"Exit code: {result.returncode}")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run test suites for concurrent RAG optimization")
    parser.add_argument("--suite", choices=[
        "unit", "integration", "performance", "security", "all"
    ], default="all", help="Test suite to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["PYTHONPATH"] = str(Path.cwd())
    
    # Base pytest command
    pytest_cmd = "python -m pytest"
    
    if args.verbose:
        pytest_cmd += " -v"
    
    if args.coverage:
        pytest_cmd += " --cov=app --cov-report=html --cov-report=term"
    
    if args.parallel:
        pytest_cmd += " -n auto"
    
    # Test suite configurations
    test_suites = {
        "unit": {
            "description": "Unit Tests",
            "files": [
                "tests/test_session_manager.py",
                "tests/test_auth.py",
                "tests/test_resource_manager.py",
                "tests/test_job_manager.py",
                "tests/test_error_handling.py",
                "tests/test_monitoring.py",
                "tests/test_security_isolation.py"
            ]
        },
        "integration": {
            "description": "Integration Tests",
            "files": [
                "tests/test_integration_workflows.py",
                "tests/test_api_endpoints_comprehensive.py"
            ]
        },
        "performance": {
            "description": "Performance Tests",
            "files": [
                "tests/test_performance_optimization.py",
                "tests/test_performance_benchmarks.py"
            ]
        },
        "security": {
            "description": "Security Tests",
            "files": [
                "tests/test_security_penetration.py"
            ]
        }
    }
    
    success = True
    
    if args.suite == "all":
        # Run all test suites
        for suite_name, suite_config in test_suites.items():
            files = " ".join(suite_config["files"])
            command = f"{pytest_cmd} {files}"
            
            if not run_command(command, f"{suite_config['description']} ({suite_name})"):
                success = False
    else:
        # Run specific test suite
        suite_config = test_suites[args.suite]
        files = " ".join(suite_config["files"])
        command = f"{pytest_cmd} {files}"
        
        success = run_command(command, suite_config["description"])
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    if success:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()