#!/usr/bin/env python3
"""
Integration test for Celery worker infrastructure.
This script tests that workers can start, process tasks, and report health.
"""
import time
import subprocess
import sys
import signal
import os
from app.workers.monitor import worker_monitor
from app.shared.redis_client import redis_client


def check_redis():
    """Check if Redis is running."""
    try:
        return redis_client.health_check()
    except:
        return False


def start_test_worker():
    """Start a single test worker."""
    print("Starting test worker...")
    
    # Start a single maintenance worker for testing
    cmd = [
        "celery", "-A", "app.workers.celery_app", "worker",
        "--hostname=test_worker@%h",
        "--queues=maintenance",
        "--concurrency=1",
        "--loglevel=info"
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process


def test_worker_health_reporting():
    """Test that worker health reporting works."""
    print("Testing worker health reporting...")
    
    # Wait for worker to start and report health
    max_wait = 30
    wait_time = 0
    
    while wait_time < max_wait:
        workers = worker_monitor.get_all_workers()
        if workers:
            print(f"✓ Found {len(workers)} worker(s)")
            for name, worker in workers.items():
                print(f"  - {name}: {worker.status.value} (queue: {worker.queue})")
            return True
        
        time.sleep(1)
        wait_time += 1
        print(f"  Waiting for workers... ({wait_time}/{max_wait})")
    
    print("✗ No workers found within timeout")
    return False


def test_system_health():
    """Test system health monitoring."""
    print("Testing system health monitoring...")
    
    health = worker_monitor.get_system_health()
    print(f"System status: {health['system_status']}")
    print(f"Workers: {health['workers']['total']} total")
    print(f"Redis health: {health['redis_health']}")
    
    return health['system_status'] in ['healthy', 'warning']


def test_maintenance_task():
    """Test that maintenance tasks can be executed."""
    print("Testing maintenance task execution...")
    
    try:
        from app.workers.maintenance_worker import system_health_check
        
        # Execute health check task directly
        result = system_health_check()
        print(f"✓ Health check task completed: {result}")
        return True
    except Exception as e:
        print(f"✗ Health check task failed: {e}")
        return False


def main():
    """Run integration tests."""
    print("=" * 60)
    print("CELERY WORKER INTEGRATION TEST")
    print("=" * 60)
    
    # Check prerequisites
    if not check_redis():
        print("✗ Redis is not running. Please start Redis first.")
        print("Run: redis-server or ./setup_redis.sh")
        return 1
    
    print("✓ Redis is running")
    
    # Start test worker
    worker_process = None
    try:
        worker_process = start_test_worker()
        
        # Give worker time to start
        print("Waiting for worker to start...")
        time.sleep(5)
        
        # Run tests
        tests = [
            ("Worker Health Reporting", test_worker_health_reporting),
            ("System Health Monitoring", test_system_health),
            ("Maintenance Task Execution", test_maintenance_task)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n--- {test_name} ---")
            try:
                if test_func():
                    print(f"✓ {test_name} PASSED")
                    passed += 1
                else:
                    print(f"✗ {test_name} FAILED")
            except Exception as e:
                print(f"✗ {test_name} ERROR: {e}")
        
        print(f"\n--- RESULTS ---")
        print(f"Passed: {passed}/{total}")
        
        if passed == total:
            print("🎉 All tests passed!")
            return 0
        else:
            print("❌ Some tests failed")
            return 1
    
    finally:
        # Clean up worker process
        if worker_process:
            print("\nStopping test worker...")
            try:
                worker_process.terminate()
                worker_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker_process.kill()
                worker_process.wait()
            print("✓ Test worker stopped")


if __name__ == "__main__":
    sys.exit(main())