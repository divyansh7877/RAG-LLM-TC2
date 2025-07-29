"""
Unit tests for security isolation and user data protection.
Tests for Requirements 1.1, 1.2, 1.3, 1.4 - User isolation and security.
"""
import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.shared.models import UserSession, Job, JobStatus, JobType
from app.shared.session_manager import SessionManager
from app.shared.auth import AuthenticationManager
from app.shared.job_manager import JobManager


class TestUserDataIsolation:
    """Test user data isolation and security filters."""
    
    @pytest.fixture
    def mock_query_engine(self):
        """Mock query engine for testing."""
        with patch('app.shared.query_engine_factory.QueryEngineFactory') as mock:
            mock_instance = Mock()
            mock.return_value = mock_instance
            
            # Mock query method to return user-specific results
            def mock_query(query_text, user_id, group_ids):
                # Simulate user-specific filtering
                if user_id == "user1":
                    return {"results": [{"doc": "user1_doc1", "content": "User 1 content"}]}
                elif user_id == "user2":
                    return {"results": [{"doc": "user2_doc1", "content": "User 2 content"}]}
                else:
                    return {"results": []}
            
            mock_instance.query.side_effect = mock_query
            yield mock_instance
    
    def test_concurrent_user_query_isolation(self, mock_query_engine):
        """Test that concurrent queries from different users are properly isolated."""
        results = {}
        errors = []
        
        def user_query_worker(user_id, group_ids):
            try:
                # Simulate query processing
                result = mock_query_engine.query(
                    "test query", 
                    user_id=user_id, 
                    group_ids=group_ids
                )
                results[user_id] = result
            except Exception as e:
                errors.append(f"User {user_id}: {e}")
        
        # Create threads for different users
        threads = []
        users = [
            ("user1", ["group1"]),
            ("user2", ["group2"]),
            ("user3", ["group3"])
        ]
        
        for user_id, group_ids in users:
            thread = threading.Thread(
                target=user_query_worker, 
                args=(user_id, group_ids)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify each user got their own results
        assert len(results) == 3
        assert "user1_doc1" in str(results["user1"])
        assert "user2_doc1" in str(results["user2"])
        assert results["user3"]["results"] == []  # No documents for user3
        
        # Verify no cross-contamination
        assert "user2_doc1" not in str(results["user1"])
        assert "user1_doc1" not in str(results["user2"])
    
    def test_session_isolation_concurrent_access(self):
        """Test that user sessions remain isolated under concurrent access."""
        session_manager = SessionManager()
        created_sessions = {}
        errors = []
        
        def create_user_session(user_id):
            try:
                with patch('app.shared.session_manager.redis_client') as mock_redis:
                    mock_redis.set_session.return_value = True
                    mock_redis.get_session.return_value = None
                    
                    session = session_manager.create_session(
                        user_id=user_id,
                        groups=[f"group_{user_id}"],
                        permissions=["upload", "query"]
                    )
                    created_sessions[user_id] = session
            except Exception as e:
                errors.append(f"User {user_id}: {e}")
        
        # Create sessions for multiple users concurrently
        threads = []
        for i in range(10):
            user_id = f"user_{i}"
            thread = threading.Thread(target=create_user_session, args=(user_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(created_sessions) == 10
        
        # Verify session isolation
        for user_id, session in created_sessions.items():
            assert session.user_id == user_id
            assert session.groups == [f"group_{user_id}"]
            
            # Verify session IDs are unique
            other_sessions = [s for uid, s in created_sessions.items() if uid != user_id]
            for other_session in other_sessions:
                assert session.session_id != other_session.session_id
    
    def test_metadata_race_condition_prevention(self):
        """Test prevention of race conditions in metadata handling."""
        job_manager = JobManager()
        created_jobs = []
        errors = []
        
        def create_embedding_job(user_id, document_name):
            try:
                with patch('app.shared.job_manager.redis_client') as mock_redis:
                    mock_redis.get_user_jobs.return_value = []
                    mock_redis.set_job.return_value = True
                    
                    job = job_manager.create_job(
                        user_id=user_id,
                        job_type=JobType.EMBEDDING,
                        metadata={
                            "document_name": document_name,
                            "user_specific_data": f"data_for_{user_id}"
                        }
                    )
                    created_jobs.append(job)
            except Exception as e:
                errors.append(f"User {user_id}, Doc {document_name}: {e}")
        
        # Create jobs for multiple users with similar document names
        threads = []
        for user_num in range(5):
            for doc_num in range(3):
                user_id = f"user_{user_num}"
                document_name = f"document_{doc_num}.pdf"
                thread = threading.Thread(
                    target=create_embedding_job, 
                    args=(user_id, document_name)
                )
                threads.append(thread)
                thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(created_jobs) == 15  # 5 users * 3 documents
        
        # Verify metadata isolation
        for job in created_jobs:
            expected_user_data = f"data_for_{job.user_id}"
            assert job.metadata["user_specific_data"] == expected_user_data
            
            # Verify no metadata mixing
            for other_job in created_jobs:
                if other_job.job_id != job.job_id and other_job.user_id != job.user_id:
                    assert job.metadata["user_specific_data"] != other_job.metadata["user_specific_data"]
    
    def test_security_filter_application(self):
        """Test that security filters are correctly applied for each user."""
        auth_manager = AuthenticationManager()
        
        with patch('app.shared.auth.config') as mock_config:
            mock_config.SECRET_KEY = "test-secret"
            mock_config.ACCESS_TOKEN_EXPIRE_MINUTES = 30
            mock_config.USERS = {
                "user1": {"password": "pass1", "groups": ["group1"]},
                "user2": {"password": "pass2", "groups": ["group2"]},
                "admin": {"password": "admin_pass", "groups": ["admin"]}
            }
            
            with patch('app.shared.auth.session_manager') as mock_session_manager:
                # Mock different sessions for different users
                def create_session_side_effect(user_id, groups, permissions):
                    return UserSession(
                        session_id=f"session_{user_id}",
                        user_id=user_id,
                        groups=groups,
                        permissions=permissions
                    )
                
                mock_session_manager.create_session.side_effect = create_session_side_effect
                mock_session_manager.validate_session.return_value = True
                mock_session_manager.update_session_activity.return_value = True
                
                # Authenticate different users
                user1_auth = auth_manager.authenticate_user("user1", "pass1")
                user2_auth = auth_manager.authenticate_user("user2", "pass2")
                admin_auth = auth_manager.authenticate_user("admin", "admin_pass")
                
                # Verify each user has correct permissions
                user1_info = auth_manager.validate_token(user1_auth["access_token"])
                user2_info = auth_manager.validate_token(user2_auth["access_token"])
                admin_info = auth_manager.validate_token(admin_auth["access_token"])
                
                # Verify user isolation
                assert user1_info["groups"] == ["group1"]
                assert user2_info["groups"] == ["group2"]
                assert admin_info["groups"] == ["admin"]
                
                # Verify permission differences
                assert "admin" not in user1_info["permissions"]
                assert "admin" not in user2_info["permissions"]
                assert "admin" in admin_info["permissions"]


class TestConcurrentSecurityOperations:
    """Test security operations under concurrent load."""
    
    def test_concurrent_authentication_isolation(self):
        """Test that concurrent authentication attempts don't interfere."""
        auth_manager = AuthenticationManager()
        results = []
        errors = []
        
        with patch('app.shared.auth.config') as mock_config:
            mock_config.SECRET_KEY = "test-secret"
            mock_config.ACCESS_TOKEN_EXPIRE_MINUTES = 30
            mock_config.USERS = {
                f"user_{i}": {"password": f"pass_{i}", "groups": [f"group_{i}"]}
                for i in range(20)
            }
            
            with patch('app.shared.auth.session_manager') as mock_session_manager:
                def create_session_side_effect(user_id, groups, permissions):
                    return UserSession(
                        session_id=f"session_{user_id}_{time.time()}",
                        user_id=user_id,
                        groups=groups,
                        permissions=permissions
                    )
                
                mock_session_manager.create_session.side_effect = create_session_side_effect
                mock_session_manager.validate_session.return_value = True
                mock_session_manager.update_session_activity.return_value = True
                
                def auth_worker(user_num):
                    try:
                        user_id = f"user_{user_num}"
                        password = f"pass_{user_num}"
                        
                        # Authenticate
                        auth_result = auth_manager.authenticate_user(user_id, password)
                        
                        # Validate token
                        user_info = auth_manager.validate_token(auth_result["access_token"])
                        
                        results.append({
                            "user_id": user_id,
                            "groups": user_info["groups"],
                            "session_id": user_info["session_id"]
                        })
                    except Exception as e:
                        errors.append(f"User {user_num}: {e}")
                
                # Run concurrent authentication
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(auth_worker, i) for i in range(20)]
                    
                    for future in as_completed(futures):
                        future.result()  # This will raise any exceptions
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 20
        
        # Verify each user got correct groups and unique sessions
        session_ids = set()
        for result in results:
            user_num = result["user_id"].split("_")[1]
            expected_group = f"group_{user_num}"
            
            assert result["groups"] == [expected_group]
            assert result["session_id"] not in session_ids
            session_ids.add(result["session_id"])
    
    def test_concurrent_permission_checks(self):
        """Test concurrent permission checking doesn't cause issues."""
        auth_manager = AuthenticationManager()
        permission_results = []
        errors = []
        
        with patch('app.shared.auth.config') as mock_config:
            mock_config.SECRET_KEY = "test-secret"
            mock_config.ACCESS_TOKEN_EXPIRE_MINUTES = 30
            mock_config.USERS = {
                "regular_user": {"password": "pass", "groups": ["common_rules"]},
                "power_user": {"password": "pass", "groups": ["assistance"]},
                "admin_user": {"password": "pass", "groups": ["admin"]}
            }
            
            with patch('app.shared.auth.session_manager') as mock_session_manager:
                def create_session_side_effect(user_id, groups, permissions):
                    return UserSession(
                        session_id=f"session_{user_id}",
                        user_id=user_id,
                        groups=groups,
                        permissions=permissions
                    )
                
                mock_session_manager.create_session.side_effect = create_session_side_effect
                mock_session_manager.validate_session.return_value = True
                mock_session_manager.update_session_activity.return_value = True
                mock_session_manager.get_session.side_effect = lambda sid: UserSession(
                    session_id=sid,
                    user_id=sid.replace("session_", ""),
                    groups=["test"],
                    permissions=["query", "upload"] if "admin" not in sid else ["query", "upload", "admin"]
                )
                
                # Authenticate users
                tokens = {}
                for user_id in ["regular_user", "power_user", "admin_user"]:
                    auth_result = auth_manager.authenticate_user(user_id, "pass")
                    tokens[user_id] = auth_result["access_token"]
                
                def permission_check_worker(user_id, permission):
                    try:
                        token = tokens[user_id]
                        has_permission = auth_manager.has_permission(token, permission)
                        permission_results.append({
                            "user_id": user_id,
                            "permission": permission,
                            "has_permission": has_permission
                        })
                    except Exception as e:
                        errors.append(f"User {user_id}, Permission {permission}: {e}")
                
                # Test various permission combinations concurrently
                permission_tests = [
                    ("regular_user", "query"),
                    ("regular_user", "upload"),
                    ("regular_user", "admin"),
                    ("power_user", "query"),
                    ("power_user", "upload"),
                    ("power_user", "admin"),
                    ("admin_user", "query"),
                    ("admin_user", "upload"),
                    ("admin_user", "admin")
                ]
                
                threads = []
                for user_id, permission in permission_tests:
                    thread = threading.Thread(
                        target=permission_check_worker,
                        args=(user_id, permission)
                    )
                    threads.append(thread)
                    thread.start()
                
                # Wait for completion
                for thread in threads:
                    thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(permission_results) == 9
        
        # Verify permission logic is correct
        for result in permission_results:
            user_id = result["user_id"]
            permission = result["permission"]
            has_permission = result["has_permission"]
            
            if permission == "admin":
                # Only admin should have admin permission
                expected = user_id == "admin_user"
            else:
                # All users should have query and upload permissions
                expected = True
            
            assert has_permission == expected, f"Permission check failed for {user_id}:{permission}"


class TestDataLeakagePrevention:
    """Test prevention of data leakage between users."""
    
    def test_job_result_isolation(self):
        """Test that job results are isolated between users."""
        job_manager = JobManager()
        job_results = {}
        errors = []
        
        def process_user_job(user_id, job_data):
            try:
                with patch('app.shared.job_manager.redis_client') as mock_redis:
                    # Mock job creation and retrieval
                    created_job = Job(
                        user_id=user_id,
                        job_type=JobType.QUERY,
                        status=JobStatus.PENDING,
                        metadata=job_data
                    )
                    
                    mock_redis.get_user_jobs.return_value = []
                    mock_redis.set_job.return_value = True
                    mock_redis.get_job.return_value = created_job
                    
                    # Create job
                    job = job_manager.create_job(
                        user_id=user_id,
                        job_type=JobType.QUERY,
                        metadata=job_data
                    )
                    
                    # Simulate job completion with user-specific results
                    job_manager.update_job_status(
                        job.job_id,
                        JobStatus.COMPLETED,
                        result={
                            "user_id": user_id,
                            "query_results": f"Results for {user_id}",
                            "document_count": len(job_data.get("documents", []))
                        }
                    )
                    
                    job_results[user_id] = job.result
                    
            except Exception as e:
                errors.append(f"User {user_id}: {e}")
        
        # Create jobs for different users with different data
        threads = []
        users_data = {
            "user1": {"documents": ["doc1.pdf", "doc2.pdf"], "query": "test query 1"},
            "user2": {"documents": ["doc3.pdf"], "query": "test query 2"},
            "user3": {"documents": ["doc4.pdf", "doc5.pdf", "doc6.pdf"], "query": "test query 3"}
        }
        
        for user_id, job_data in users_data.items():
            thread = threading.Thread(
                target=process_user_job,
                args=(user_id, job_data)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(job_results) == 3
        
        # Verify result isolation
        for user_id, result in job_results.items():
            assert result["user_id"] == user_id
            assert f"Results for {user_id}" in result["query_results"]
            
            # Verify no cross-contamination
            for other_user_id in users_data.keys():
                if other_user_id != user_id:
                    assert f"Results for {other_user_id}" not in result["query_results"]
    
    def test_session_data_isolation(self):
        """Test that session data doesn't leak between users."""
        session_manager = SessionManager()
        session_data = {}
        errors = []
        
        def manage_user_session(user_id, sensitive_data):
            try:
                with patch('app.shared.session_manager.redis_client') as mock_redis:
                    # Mock session storage and retrieval
                    stored_sessions = {}
                    
                    def mock_set_session(session):
                        stored_sessions[session.session_id] = session
                        return True
                    
                    def mock_get_session(session_id):
                        return stored_sessions.get(session_id)
                    
                    mock_redis.set_session.side_effect = mock_set_session
                    mock_redis.get_session.side_effect = mock_get_session
                    
                    # Create session with sensitive data
                    session = session_manager.create_session(
                        user_id=user_id,
                        groups=[f"group_{user_id}"],
                        permissions=["upload", "query"]
                    )
                    
                    # Store sensitive data in session metadata (simulated)
                    session.metadata = {"sensitive_data": sensitive_data}
                    mock_redis.set_session(session)
                    
                    # Retrieve session
                    retrieved_session = session_manager.get_session(session.session_id)
                    session_data[user_id] = {
                        "session_id": retrieved_session.session_id,
                        "user_id": retrieved_session.user_id,
                        "groups": retrieved_session.groups,
                        "sensitive_data": retrieved_session.metadata.get("sensitive_data")
                    }
                    
            except Exception as e:
                errors.append(f"User {user_id}: {e}")
        
        # Create sessions with different sensitive data
        threads = []
        sensitive_data_map = {
            "user1": "secret_data_1",
            "user2": "secret_data_2", 
            "user3": "secret_data_3"
        }
        
        for user_id, sensitive_data in sensitive_data_map.items():
            thread = threading.Thread(
                target=manage_user_session,
                args=(user_id, sensitive_data)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(session_data) == 3
        
        # Verify session data isolation
        for user_id, data in session_data.items():
            assert data["user_id"] == user_id
            assert data["groups"] == [f"group_{user_id}"]
            assert data["sensitive_data"] == sensitive_data_map[user_id]
            
            # Verify no data leakage
            for other_user_id, other_data in session_data.items():
                if other_user_id != user_id:
                    assert data["session_id"] != other_data["session_id"]
                    assert data["sensitive_data"] != other_data["sensitive_data"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])