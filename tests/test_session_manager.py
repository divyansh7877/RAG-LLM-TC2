"""
Unit tests for SessionManager with thread safety and isolation testing.
"""
import pytest
import threading
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.shared.session_manager import (
    SessionManager, SessionError, SessionNotFoundError, 
    SessionExpiredError, SessionValidationError
)
from app.shared.models import UserSession
from app.shared.redis_client import RedisConnectionError


class TestSessionManager:
    """Test suite for SessionManager."""
    
    @pytest.fixture
    def session_manager(self):
        """Create a fresh SessionManager instance for testing."""
        return SessionManager()
    
    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client for testing."""
        with patch('app.shared.session_manager.redis_client') as mock:
            yield mock
    
    @pytest.fixture
    def sample_session(self):
        """Create a sample session for testing."""
        return UserSession(
            session_id=str(uuid.uuid4()),
            user_id="test_user",
            groups=["test_group"],
            permissions=["upload", "query"],
            created_at=datetime.now(),
            last_activity=datetime.now(),
            is_active=True
        )
    
    def test_session_manager_initialization(self, session_manager):
        """Test SessionManager initialization."""
        assert session_manager is not None
        assert hasattr(session_manager, '_lock')
        assert hasattr(session_manager, '_active_sessions')
        assert isinstance(session_manager._active_sessions, set)
    
    def test_generate_session_id(self, session_manager):
        """Test session ID generation."""
        session_id1 = session_manager._generate_session_id()
        session_id2 = session_manager._generate_session_id()
        
        # Session IDs should be unique
        assert session_id1 != session_id2
        
        # Session IDs should be strings
        assert isinstance(session_id1, str)
        assert isinstance(session_id2, str)
        
        # Session IDs should be reasonable length (SHA256 hex = 64 chars)
        assert len(session_id1) == 64
        assert len(session_id2) == 64
    
    def test_validate_session_data_valid(self, session_manager, sample_session):
        """Test session data validation with valid session."""
        assert session_manager._validate_session_data(sample_session) is True
    
    def test_validate_session_data_invalid_missing_fields(self, session_manager):
        """Test session data validation with missing required fields."""
        # Missing session_id
        session = UserSession(
            session_id="",
            user_id="test_user",
            groups=["test_group"],
            permissions=["upload", "query"]
        )
        assert session_manager._validate_session_data(session) is False
        
        # Missing user_id
        session = UserSession(
            session_id="test_session",
            user_id="",
            groups=["test_group"],
            permissions=["upload", "query"]
        )
        assert session_manager._validate_session_data(session) is False
    
    def test_validate_session_data_inactive(self, session_manager, sample_session):
        """Test session data validation with inactive session."""
        sample_session.is_active = False
        assert session_manager._validate_session_data(sample_session) is False
    
    def test_validate_session_data_expired(self, session_manager, sample_session):
        """Test session data validation with expired session."""
        # Set last activity to 25 hours ago (beyond 24-hour expiration)
        sample_session.last_activity = datetime.now() - timedelta(hours=25)
        assert session_manager._validate_session_data(sample_session) is False
    
    def test_validate_session_data_invalid_permissions(self, session_manager, sample_session):
        """Test session data validation with invalid permissions."""
        sample_session.permissions = ["invalid_permission"]
        assert session_manager._validate_session_data(sample_session) is False
    
    def test_create_session_success(self, session_manager, mock_redis_client):
        """Test successful session creation."""
        mock_redis_client.set_session.return_value = True
        
        session = session_manager.create_session(
            user_id="test_user",
            groups=["test_group"],
            permissions=["upload", "query"]
        )
        
        assert session is not None
        assert session.user_id == "test_user"
        assert session.groups == ["test_group"]
        assert session.permissions == ["upload", "query"]
        assert session.is_active is True
        assert session.session_id in session_manager._active_sessions
        
        # Verify Redis was called
        mock_redis_client.set_session.assert_called_once()
    
    def test_create_session_redis_failure(self, session_manager, mock_redis_client):
        """Test session creation with Redis failure."""
        mock_redis_client.set_session.return_value = False
        
        with pytest.raises(SessionError, match="Failed to store session in Redis"):
            session_manager.create_session(
                user_id="test_user",
                groups=["test_group"]
            )
    
    def test_create_session_default_permissions(self, session_manager, mock_redis_client):
        """Test session creation with default permissions."""
        mock_redis_client.set_session.return_value = True
        
        session = session_manager.create_session(
            user_id="test_user",
            groups=["test_group"]
        )
        
        assert session.permissions == ["upload", "query"]
    
    def test_get_session_success(self, session_manager, mock_redis_client, sample_session):
        """Test successful session retrieval."""
        mock_redis_client.get_session.return_value = sample_session
        
        retrieved_session = session_manager.get_session(sample_session.session_id)
        
        assert retrieved_session is not None
        assert retrieved_session.session_id == sample_session.session_id
        assert retrieved_session.user_id == sample_session.user_id
        assert sample_session.session_id in session_manager._active_sessions
        
        mock_redis_client.get_session.assert_called_once_with(sample_session.session_id)
    
    def test_get_session_not_found(self, session_manager, mock_redis_client):
        """Test session retrieval when session not found."""
        mock_redis_client.get_session.return_value = None
        
        session = session_manager.get_session("nonexistent_session")
        
        assert session is None
        assert "nonexistent_session" not in session_manager._active_sessions
    
    def test_get_session_invalid_data(self, session_manager, mock_redis_client, sample_session):
        """Test session retrieval with invalid session data."""
        # Make session invalid
        sample_session.is_active = False
        mock_redis_client.get_session.return_value = sample_session
        mock_redis_client.delete_session.return_value = True
        
        session = session_manager.get_session(sample_session.session_id)
        
        assert session is None
        # Should have attempted to clean up invalid session
        mock_redis_client.delete_session.assert_called_once_with(sample_session.session_id)
    
    def test_validate_session_valid(self, session_manager, mock_redis_client, sample_session):
        """Test session validation with valid session."""
        mock_redis_client.get_session.return_value = sample_session
        
        is_valid = session_manager.validate_session(sample_session.session_id)
        
        assert is_valid is True
    
    def test_validate_session_invalid(self, session_manager, mock_redis_client):
        """Test session validation with invalid session."""
        mock_redis_client.get_session.return_value = None
        
        is_valid = session_manager.validate_session("invalid_session")
        
        assert is_valid is False
    
    def test_update_session_activity_success(self, session_manager, mock_redis_client, sample_session):
        """Test successful session activity update."""
        original_activity = sample_session.last_activity
        mock_redis_client.get_session.return_value = sample_session
        mock_redis_client.set_session.return_value = True
        
        # Wait a moment to ensure timestamp difference
        time.sleep(0.01)
        
        success = session_manager.update_session_activity(sample_session.session_id)
        
        assert success is True
        # Activity should be updated (we can't check exact time due to mocking)
        mock_redis_client.set_session.assert_called_once()
    
    def test_update_session_activity_not_found(self, session_manager, mock_redis_client):
        """Test session activity update when session not found."""
        mock_redis_client.get_session.return_value = None
        
        success = session_manager.update_session_activity("nonexistent_session")
        
        assert success is False
    
    def test_delete_session_success(self, session_manager, mock_redis_client):
        """Test successful session deletion."""
        session_id = "test_session"
        session_manager._active_sessions.add(session_id)
        mock_redis_client.delete_session.return_value = True
        
        success = session_manager.delete_session(session_id)
        
        assert success is True
        assert session_id not in session_manager._active_sessions
        mock_redis_client.delete_session.assert_called_once_with(session_id)
    
    def test_delete_session_not_found(self, session_manager, mock_redis_client):
        """Test session deletion when session not found."""
        mock_redis_client.delete_session.return_value = False
        
        success = session_manager.delete_session("nonexistent_session")
        
        assert success is False
    
    def test_get_user_sessions(self, session_manager, mock_redis_client, sample_session):
        """Test getting all sessions for a user."""
        user_sessions = [sample_session]
        mock_redis_client.get_user_sessions.return_value = user_sessions
        
        sessions = session_manager.get_user_sessions("test_user")
        
        assert len(sessions) == 1
        assert sessions[0].session_id == sample_session.session_id
        assert sample_session.session_id in session_manager._active_sessions
    
    def test_invalidate_user_sessions(self, session_manager, mock_redis_client, sample_session):
        """Test invalidating all sessions for a user."""
        user_sessions = [sample_session]
        mock_redis_client.get_user_sessions.return_value = user_sessions
        mock_redis_client.delete_session.return_value = True
        
        invalidated = session_manager.invalidate_user_sessions("test_user")
        
        assert invalidated == 1
        mock_redis_client.delete_session.assert_called_once_with(sample_session.session_id)
    
    def test_cleanup_expired_sessions(self, session_manager, mock_redis_client):
        """Test cleanup of expired sessions."""
        mock_redis_client.cleanup_expired_sessions.return_value = 3
        mock_redis_client.get_all_active_sessions.return_value = []
        
        cleaned = session_manager.cleanup_expired_sessions()
        
        assert cleaned == 3
        mock_redis_client.cleanup_expired_sessions.assert_called_once()
    
    def test_get_session_stats(self, session_manager, mock_redis_client, sample_session):
        """Test getting session statistics."""
        all_sessions = [sample_session]
        mock_redis_client.get_all_active_sessions.return_value = all_sessions
        
        stats = session_manager.get_session_stats()
        
        assert stats["total_sessions"] == 1
        assert stats["active_sessions"] == 1
        assert stats["users_with_sessions"] == 1
        assert "test_user" in stats["sessions_by_user"]
        assert stats["sessions_by_user"]["test_user"] == 1
    
    def test_health_check(self, session_manager, mock_redis_client):
        """Test health check functionality."""
        mock_redis_client.health_check.return_value = {
            "redis_sessions": True,
            "errors": []
        }
        
        health = session_manager.health_check()
        
        assert health["session_manager"] is True
        assert health["redis_connection"] is True
        assert health["thread_safety"] is True
        assert len(health["errors"]) == 0


class TestSessionManagerThreadSafety:
    """Test suite for SessionManager thread safety."""
    
    @pytest.fixture
    def session_manager(self):
        """Create a fresh SessionManager instance for testing."""
        return SessionManager()
    
    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client for testing."""
        with patch('app.shared.session_manager.redis_client') as mock:
            # Configure mock to simulate successful operations
            mock.set_session.return_value = True
            mock.get_session.return_value = None
            mock.delete_session.return_value = True
            mock.get_user_sessions.return_value = []
            mock.cleanup_expired_sessions.return_value = 0
            mock.get_all_active_sessions.return_value = []
            yield mock
    
    def test_concurrent_session_creation(self, session_manager, mock_redis_client):
        """Test concurrent session creation for thread safety."""
        num_threads = 10
        sessions_created = []
        errors = []
        
        def create_session_worker(worker_id):
            try:
                session = session_manager.create_session(
                    user_id=f"user_{worker_id}",
                    groups=[f"group_{worker_id}"],
                    permissions=["upload", "query"]
                )
                sessions_created.append(session)
            except Exception as e:
                errors.append(e)
        
        # Create and start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=create_session_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(sessions_created) == num_threads
        
        # Verify all session IDs are unique
        session_ids = [s.session_id for s in sessions_created]
        assert len(set(session_ids)) == num_threads
        
        # Verify all sessions are in active cache
        for session in sessions_created:
            assert session.session_id in session_manager._active_sessions
    
    def test_concurrent_session_operations(self, session_manager, mock_redis_client):
        """Test concurrent mixed session operations."""
        num_operations = 20
        results = []
        errors = []
        
        # Create some initial sessions
        initial_sessions = []
        for i in range(5):
            session = UserSession(
                session_id=f"session_{i}",
                user_id=f"user_{i}",
                groups=[f"group_{i}"],
                permissions=["upload", "query"]
            )
            initial_sessions.append(session)
            session_manager._active_sessions.add(session.session_id)
        
        # Configure mock to return these sessions
        def mock_get_session(session_id):
            for session in initial_sessions:
                if session.session_id == session_id:
                    return session
            return None
        
        mock_redis_client.get_session.side_effect = mock_get_session
        
        def mixed_operations_worker(worker_id):
            try:
                operation = worker_id % 4
                
                if operation == 0:  # Create session
                    session = session_manager.create_session(
                        user_id=f"new_user_{worker_id}",
                        groups=[f"new_group_{worker_id}"]
                    )
                    results.append(f"created_{session.session_id}")
                
                elif operation == 1:  # Get session
                    session_id = f"session_{worker_id % 5}"
                    session = session_manager.get_session(session_id)
                    results.append(f"get_{session_id}_{session is not None}")
                
                elif operation == 2:  # Update activity
                    session_id = f"session_{worker_id % 5}"
                    success = session_manager.update_session_activity(session_id)
                    results.append(f"update_{session_id}_{success}")
                
                elif operation == 3:  # Validate session
                    session_id = f"session_{worker_id % 5}"
                    valid = session_manager.validate_session(session_id)
                    results.append(f"validate_{session_id}_{valid}")
                
            except Exception as e:
                errors.append(e)
        
        # Execute concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(mixed_operations_worker, i) for i in range(num_operations)]
            
            for future in as_completed(futures):
                future.result()  # This will raise any exceptions
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_operations
    
    def test_concurrent_session_isolation(self, session_manager, mock_redis_client):
        """Test that concurrent operations maintain session isolation."""
        num_users = 5
        sessions_per_user = 3
        all_sessions = {}
        errors = []
        
        def create_user_sessions(user_id):
            try:
                user_sessions = []
                for i in range(sessions_per_user):
                    session = session_manager.create_session(
                        user_id=user_id,
                        groups=[f"group_{user_id}"],
                        permissions=["upload", "query"]
                    )
                    user_sessions.append(session)
                all_sessions[user_id] = user_sessions
            except Exception as e:
                errors.append(e)
        
        # Create sessions for multiple users concurrently
        threads = []
        for i in range(num_users):
            user_id = f"user_{i}"
            thread = threading.Thread(target=create_user_sessions, args=(user_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify session isolation
        assert len(all_sessions) == num_users
        
        for user_id, sessions in all_sessions.items():
            assert len(sessions) == sessions_per_user
            
            # Verify all sessions belong to correct user
            for session in sessions:
                assert session.user_id == user_id
                assert session.groups == [f"group_{user_id}"]
            
            # Verify session IDs are unique within user
            session_ids = [s.session_id for s in sessions]
            assert len(set(session_ids)) == sessions_per_user
        
        # Verify all session IDs are globally unique
        all_session_ids = []
        for sessions in all_sessions.values():
            all_session_ids.extend([s.session_id for s in sessions])
        assert len(set(all_session_ids)) == num_users * sessions_per_user
    
    def test_concurrent_cleanup_operations(self, session_manager, mock_redis_client):
        """Test concurrent cleanup operations don't interfere with each other."""
        cleanup_results = []
        errors = []
        
        # Configure mock to return different cleanup counts
        cleanup_counts = [0, 1, 2, 0, 3, 1, 0, 2]
        mock_redis_client.cleanup_expired_sessions.side_effect = cleanup_counts
        
        def cleanup_worker(worker_id):
            try:
                cleaned = session_manager.cleanup_expired_sessions()
                cleanup_results.append(cleaned)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent cleanup operations
        threads = []
        for i in range(len(cleanup_counts)):
            thread = threading.Thread(target=cleanup_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(cleanup_results) == len(cleanup_counts)
        
        # Verify cleanup was called the expected number of times
        assert mock_redis_client.cleanup_expired_sessions.call_count == len(cleanup_counts)
    
    def test_redis_connection_error_handling(self, session_manager):
        """Test thread safety when Redis connection errors occur."""
        errors = []
        results = []
        
        with patch('app.shared.session_manager.redis_client') as mock_redis:
            # Configure mock to raise connection errors
            mock_redis.set_session.side_effect = RedisConnectionError("Connection failed")
            mock_redis.get_session.side_effect = RedisConnectionError("Connection failed")
            
            def operation_worker(worker_id):
                try:
                    if worker_id % 2 == 0:
                        # Try to create session
                        session_manager.create_session(
                            user_id=f"user_{worker_id}",
                            groups=["test"]
                        )
                        results.append(f"create_success_{worker_id}")
                    else:
                        # Try to get session
                        session = session_manager.get_session(f"session_{worker_id}")
                        results.append(f"get_success_{worker_id}")
                except SessionError as e:
                    # Expected error due to Redis connection failure
                    errors.append(str(e))
                except Exception as e:
                    # Unexpected error
                    errors.append(f"Unexpected: {e}")
            
            # Run concurrent operations that will fail
            threads = []
            for i in range(10):
                thread = threading.Thread(target=operation_worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
        
        # All operations should have failed with SessionError
        assert len(results) == 0  # No successful operations
        assert len(errors) == 10  # All operations should have failed
        
        # All errors should be related to session service unavailability
        for error in errors:
            assert ("Session service temporarily unavailable" in error or 
                    "Session operation failed" in error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])