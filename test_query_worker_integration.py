#!/usr/bin/env python3
"""
Integration test for query worker functionality.
This test verifies that the query worker can be imported and basic functions work.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_query_worker_imports():
    """Test that query worker can be imported successfully."""
    try:
        from app.workers.query_worker import (
            create_user_security_filters,
            generate_cache_key,
            validate_query_security,
            QuerySecurityError
        )
        print("✓ Query worker imports successful")
        return True
    except Exception as e:
        print(f"✗ Query worker import failed: {e}")
        return False

def test_security_filters():
    """Test security filter creation."""
    try:
        from app.workers.query_worker import create_user_security_filters
        
        # Test valid security filters
        filters = create_user_security_filters("test_user", ["group1", "group2"])
        assert filters is not None
        assert len(filters.filters) == 3  # 1 user + 2 groups
        assert filters.condition == "or"
        
        print("✓ Security filters creation works")
        return True
    except Exception as e:
        print(f"✗ Security filters test failed: {e}")
        return False

def test_cache_key_generation():
    """Test cache key generation."""
    try:
        from app.workers.query_worker import generate_cache_key
        
        # Test cache key generation
        key1 = generate_cache_key("user1", ["group1"], "test query")
        key2 = generate_cache_key("user1", ["group1"], "test query")
        key3 = generate_cache_key("user2", ["group1"], "test query")
        
        assert key1 == key2  # Same inputs should produce same key
        assert key1 != key3  # Different users should produce different keys
        assert key1.startswith("query_cache:")
        
        print("✓ Cache key generation works")
        return True
    except Exception as e:
        print(f"✗ Cache key generation test failed: {e}")
        return False

def test_query_validation():
    """Test query security validation."""
    try:
        from app.workers.query_worker import validate_query_security, QuerySecurityError
        
        # Test valid query
        validate_query_security("test_user", ["group1"], "What is AI?")
        
        # Test invalid queries
        try:
            validate_query_security("", ["group1"], "test")
            assert False, "Should have raised QuerySecurityError"
        except QuerySecurityError:
            pass  # Expected
        
        try:
            validate_query_security("user", [], "test")
            assert False, "Should have raised QuerySecurityError"
        except QuerySecurityError:
            pass  # Expected
        
        try:
            validate_query_security("user", ["group1"], "")
            assert False, "Should have raised QuerySecurityError"
        except QuerySecurityError:
            pass  # Expected
        
        print("✓ Query validation works")
        return True
    except Exception as e:
        print(f"✗ Query validation test failed: {e}")
        return False

def test_celery_task_registration():
    """Test that Celery tasks are properly registered."""
    try:
        from app.workers.celery_app import celery_app
        
        # Check if query tasks are registered
        registered_tasks = celery_app.tasks.keys()
        
        expected_tasks = [
            "process_user_query",
            "get_query_status", 
            "cleanup_query_cache"
        ]
        
        for task in expected_tasks:
            if task not in registered_tasks:
                print(f"✗ Task {task} not registered")
                return False
        
        print("✓ Celery tasks properly registered")
        return True
    except Exception as e:
        print(f"✗ Celery task registration test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("=== Query Worker Integration Tests ===\n")
    
    tests = [
        test_query_worker_imports,
        test_security_filters,
        test_cache_key_generation,
        test_query_validation,
        test_celery_task_registration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("❌ Some integration tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())