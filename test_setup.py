#!/usr/bin/env python3
"""
Test script to verify the project infrastructure setup.
"""

import sys
import os
import importlib.util

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing module imports...")
    
    try:
        # Test shared modules
        from app.shared.models import UserSession, Job, JobStatus
        from app.shared.config import config
        from app.shared.redis_client import redis_client
        print("✓ Shared modules imported successfully")
        
        # Test worker modules
        from app.workers.celery_app import celery_app
        from app.workers.embedding_worker import process_document_embedding
        from app.workers.query_worker import process_user_query
        from app.workers.maintenance_worker import cleanup_expired_sessions
        print("✓ Worker modules imported successfully")
        
        # Test API module
        from app.api.main import app
        print("✓ API module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_dependencies():
    """Test that required dependencies are available."""
    print("\nTesting dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'celery',
        'redis',
        'websockets',
        'pydantic',
        'jose',
        'passlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'jose':
                import jose
            elif package == 'passlib':
                import passlib
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def test_redis_connection():
    """Test Redis connection."""
    print("\nTesting Redis connection...")
    
    try:
        from app.shared.redis_client import redis_client
        
        if redis_client.health_check():
            print("✓ Redis connection successful")
            return True
        else:
            print("✗ Redis connection failed")
            print("Make sure Redis is running. Run: ./setup_redis.sh")
            return False
            
    except Exception as e:
        print(f"✗ Redis test error: {e}")
        return False

def test_directory_structure():
    """Test that directory structure is correct."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'app/api',
        'app/workers', 
        'app/shared'
    ]
    
    required_files = [
        'app/__init__.py',
        'app/api/__init__.py',
        'app/api/main.py',
        'app/workers/__init__.py',
        'app/workers/celery_app.py',
        'app/workers/embedding_worker.py',
        'app/workers/query_worker.py',
        'app/workers/maintenance_worker.py',
        'app/shared/__init__.py',
        'app/shared/models.py',
        'app/shared/config.py',
        'app/shared/redis_client.py',
        'requirements.txt',
        'setup_redis.sh',
        'start_dev.sh'
    ]
    
    all_good = True
    
    for directory in required_dirs:
        if os.path.isdir(directory):
            print(f"✓ {directory}/")
        else:
            print(f"✗ {directory}/ - MISSING")
            all_good = False
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_good = False
    
    return all_good

def main():
    """Run all tests."""
    print("=== Concurrent RAG System Setup Test ===\n")
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Dependencies", test_dependencies),
        ("Module Imports", test_imports),
        ("Redis Connection", test_redis_connection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n=== Test Summary ===")
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The project infrastructure is set up correctly.")
        print("\nNext steps:")
        print("1. Run ./setup_redis.sh to install and configure Redis")
        print("2. Run ./start_dev.sh to see how to start the services")
        print("3. Start implementing the remaining tasks")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()