"""
Tests for query processing API endpoints.
"""
import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from datetime import datetime

from app.api.main import app
from app.shared.models import UserSession, Query, Job, JobStatus, JobType, QueryRequest
from app.shared.redis_client import redis_client
from app.shared.job_manager import job_manager
from app.shared.middleware import get_current_user


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    """Clear dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


class TestQuerySubmission:
    """Test query submission endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user_session(self):
        """Create mock user session."""
        return UserSession(
            session_id="test-session-123",
            user_id="test-user",
            groups=["test-group", "common_rules"],
            permissions=["upload", "query", "delete"]
        )
    
    @patch('app.shared.job_manager.job_manager.create_job')
    @patch('app.workers.query_worker.process_user_query.delay')
    @patch('app.shared.redis_client.redis_client.set_json')
    @patch('app.shared.redis_client.redis_client.set_job')
    def test_submit_query_success(self, mock_set_job, mock_set_json, mock_delay, mock_create_job, client, mock_user_session):
        """Test successful query submission."""
        # Setup authentication
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        
        # Setup mocks
        mock_job = Job(
            job_id="test-job-123",
            user_id="test-user",
            job_type=JobType.QUERY,
            status=JobStatus.PENDING
        )
        mock_create_job.return_value = mock_job
        
        mock_task = Mock()
        mock_task.id = "celery-task-123"
        mock_delay.return_value = mock_task
        
        # Make request
        response = client.post(
            "/api/query",
            json={"query_text": "What is the main topic of the documents?"}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "query_id" in data
        assert data["job_id"] == "test-job-123"
        assert data["message"] == "Query submitted successfully"
        assert data["status"] == "pending"
        
        # Verify mocks were called
        mock_create_job.assert_called_once()
        mock_delay.assert_called_once()
        mock_set_json.assert_called()
        mock_set_job.assert_called_once()
    
    def test_submit_query_no_permission(self, client):
        """Test query submission without query permission."""
        # User without query permission
        user_session = UserSession(
            session_id="test-session-123",
            user_id="test-user",
            groups=["test-group"],
            permissions=["upload"]  # No query permission
        )
        
        app.dependency_overrides[get_current_user] = lambda: user_session
        
        response = client.post(
            "/api/query",
            json={"query_text": "What is the main topic?"}
        )
        
        assert response.status_code == 403
        assert "query permission" in response.json()["error"]["message"]
    
    def test_submit_query_empty_text(self, client, mock_user_session):
        """Test query submission with empty query text."""
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        
        response = client.post(
            "/api/query",
            json={"query_text": ""}
        )
        
        assert response.status_code == 400
        assert "Query text cannot be empty" in response.json()["error"]["message"]
    
    def test_submit_query_text_too_long(self, client, mock_user_session):
        """Test query submission with text exceeding length limit."""
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        
        # Create a query text longer than 2000 characters
        long_query = "x" * 2001
        
        response = client.post(
            "/api/query",
            json={"query_text": long_query}
        )
        
        assert response.status_code == 400
        assert "Query text too long" in response.json()["error"]["message"]
    
    def test_submit_query_invalid_json(self, client, mock_user_session):
        """Test query submission with invalid JSON."""
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        
        response = client.post(
            "/api/query",
            json={"invalid_field": "value"}
        )
        
        assert response.status_code == 422
        assert "validation" in response.json()["error"]["code"].lower()


class TestQueryRetrieval:
    """Test query result retrieval endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user_session(self):
        """Create mock user session."""
        return UserSession(
            session_id="test-session-123",
            user_id="test-user",
            groups=["test-group", "common_rules"],
            permissions=["upload", "query", "delete"]
        )
    
    @pytest.fixture
    def sample_completed_query(self):
        """Create sample completed query data."""
        return {
            "query_id": "query-123",
            "user_id": "test-user",
            "query_text": "What is the main topic?",
            "status": "completed",
            "created_at": "2025-01-21T10:00:00Z",
            "processing_time": 2.5,
            "job_id": "job-123",
            "result": {
                "answer": "The main topic is artificial intelligence and machine learning.",
                "sources": ["document1.pdf (Page 1)", "document2.pdf (Page 3)"],
                "result_count": 2,
                "cached": False,
                "query_metadata": {
                    "similarity_threshold": 0.7,
                    "max_retrieved_nodes": 10,
                    "user_groups": ["test-group", "common_rules"]
                }
            }
        }
    
    @patch('app.shared.redis_client.redis_client.get_json')
    @patch('app.shared.job_manager.job_manager.get_job')
    def test_get_query_result_completed(self, mock_get_job, mock_get_json, client, mock_user_session, sample_completed_query):
        """Test getting completed query result."""
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        mock_get_json.return_value = sample_completed_query
        
        # Mock job info
        mock_job = Job(
            job_id="job-123",
            user_id="test-user",
            job_type=JobType.QUERY,
            status=JobStatus.COMPLETED,
            progress=1.0
        )
        mock_get_job.return_value = mock_job
        
        response = client.get("/api/query/query-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["query_id"] == "query-123"
        assert data["status"] == "completed"
        assert data["answer"] == "The main topic is artificial intelligence and machine learning."
        assert len(data["sources"]) == 2
        assert data["result_count"] == 2
        assert data["cached"] is False
        assert data["job_info"]["status"] == "completed"
    
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_get_query_result_not_found(self, mock_get_json, client, mock_user_session):
        """Test getting non-existent query result."""
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        mock_get_json.return_value = None
        
        response = client.get("/api/query/nonexistent-query")
        
        assert response.status_code == 404
        assert "Query not found" in response.json()["error"]["message"]
    
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_get_query_result_access_denied(self, mock_get_json, client, mock_user_session):
        """Test getting query result from different user."""
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        
        # Query belongs to different user
        other_user_query = {
            "query_id": "query-123",
            "user_id": "other-user",
            "query_text": "What is the main topic?",
            "status": "completed"
        }
        mock_get_json.return_value = other_user_query
        
        response = client.get("/api/query/query-123")
        
        assert response.status_code == 403
        assert "Access denied" in response.json()["error"]["message"]


class TestQueryStatus:
    """Test query status endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user_session(self):
        """Create mock user session."""
        return UserSession(
            session_id="test-session-123",
            user_id="test-user",
            groups=["test-group", "common_rules"],
            permissions=["upload", "query", "delete"]
        )
    
    @patch('app.shared.redis_client.redis_client.get_json')
    @patch('celery.result.AsyncResult')
    def test_get_query_status_with_task(self, mock_async_result, mock_get_json, client, mock_user_session):
        """Test getting query status with Celery task information."""
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        
        sample_query_with_task = {
            "query_id": "query-123",
            "user_id": "test-user",
            "query_text": "What is the main topic?",
            "status": "processing",
            "progress": 0.6,
            "status_message": "Processing query...",
            "created_at": "2025-01-21T10:00:00Z",
            "started_at": "2025-01-21T10:00:05Z",
            "task_id": "celery-task-123",
            "last_updated": time.time()
        }
        mock_get_json.return_value = sample_query_with_task
        
        # Mock Celery task result
        mock_task_result = Mock()
        mock_task_result.state = "PROGRESS"
        mock_task_result.info = {"progress": 0.6, "status": "Processing query..."}
        mock_async_result.return_value = mock_task_result
        
        response = client.get("/api/query/query-123/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["query_id"] == "query-123"
        assert data["status"] == "processing"
        assert data["progress"] == 0.6
        assert data["status_message"] == "Processing query..."
        assert data["task_status"]["state"] == "PROGRESS"
        assert data["task_status"]["info"]["progress"] == 0.6
    
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_get_query_status_not_found(self, mock_get_json, client, mock_user_session):
        """Test getting status for non-existent query."""
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        mock_get_json.return_value = None
        
        response = client.get("/api/query/nonexistent-query/status")
        
        assert response.status_code == 404
        assert "Query not found" in response.json()["error"]["message"]


class TestQueryListing:
    """Test query listing endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user_session(self):
        """Create mock user session."""
        return UserSession(
            session_id="test-session-123",
            user_id="test-user",
            groups=["test-group", "common_rules"],
            permissions=["upload", "query", "delete"]
        )
    
    @patch('app.shared.redis_client.redis_client.get_connection')
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_list_queries_success(self, mock_get_json, mock_get_connection, client, mock_user_session):
        """Test successful query listing."""
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        
        sample_queries = [
            {
                "query_id": "query-1",
                "user_id": "test-user",
                "query_text": "What is the main topic of the documents?",
                "status": "completed",
                "created_at": "2025-01-21T10:00:00Z",
                "completed_at": "2025-01-21T10:00:30Z",
                "processing_time": 30.0,
                "result": {
                    "result_count": 3,
                    "cached": False
                }
            },
            {
                "query_id": "query-2",
                "user_id": "test-user",
                "query_text": "What are the key findings in the research?",
                "status": "processing",
                "created_at": "2025-01-21T10:05:00Z"
            },
            {
                "query_id": "query-3",
                "user_id": "other-user",  # Different user
                "query_text": "Other user's query",
                "status": "completed",
                "created_at": "2025-01-21T10:10:00Z"
            }
        ]
        
        # Mock Redis connection and keys
        mock_client = Mock()
        mock_client.keys.return_value = [
            b"query:query-1",
            b"query:query-2",
            b"query:query-3"
        ]
        mock_get_connection.return_value.__enter__.return_value = mock_client
        
        # Mock query data retrieval
        def mock_get_json_side_effect(key):
            if "query-1" in key:
                return sample_queries[0]
            elif "query-2" in key:
                return sample_queries[1]
            elif "query-3" in key:
                return sample_queries[2]
            return None
        
        mock_get_json.side_effect = mock_get_json_side_effect
        
        response = client.get("/api/queries")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2  # Only user's queries
        assert len(data["queries"]) == 2
        
        # Check that queries are sorted by creation time (newest first)
        assert data["queries"][0]["query_id"] == "query-2"  # More recent
        assert data["queries"][1]["query_id"] == "query-1"  # Older
    
    def test_list_queries_invalid_status_filter(self, client, mock_user_session):
        """Test query listing with invalid status filter."""
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        
        response = client.get("/api/queries?status=invalid")
        
        assert response.status_code == 400
        assert "Invalid status filter" in response.json()["error"]["message"]


class TestQueryDeletion:
    """Test query deletion endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user_session(self):
        """Create mock user session."""
        return UserSession(
            session_id="test-session-123",
            user_id="test-user",
            groups=["test-group", "common_rules"],
            permissions=["upload", "query", "delete"]
        )
    
    @patch('app.shared.redis_client.redis_client.get_json')
    @patch('app.shared.redis_client.redis_client.delete')
    @patch('celery.result.AsyncResult')
    @patch('app.shared.job_manager.job_manager.get_job')
    @patch('app.shared.redis_client.redis_client.set_job')
    def test_delete_query_with_running_task(self, mock_set_job, mock_get_job, mock_async_result, mock_delete, mock_get_json, client, mock_user_session):
        """Test deleting query with running Celery task."""
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        
        sample_query_with_task = {
            "query_id": "query-123",
            "user_id": "test-user",
            "query_text": "What is the main topic of the documents?",
            "status": "processing",
            "task_id": "celery-task-123",
            "job_id": "job-123"
        }
        mock_get_json.return_value = sample_query_with_task
        mock_delete.return_value = True
        
        # Mock Celery task cancellation
        mock_task_result = Mock()
        mock_async_result.return_value = mock_task_result
        
        # Mock job cancellation
        mock_job = Job(
            job_id="job-123",
            user_id="test-user",
            job_type=JobType.QUERY,
            status=JobStatus.PROCESSING
        )
        mock_get_job.return_value = mock_job
        
        response = client.delete("/api/query/query-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Query deleted successfully"
        assert data["query_id"] == "query-123"
        
        # Verify task was cancelled
        mock_task_result.revoke.assert_called_once_with(terminate=True)
        
        # Verify job was cancelled
        mock_set_job.assert_called_once()
        
        # Verify query was deleted from Redis
        mock_delete.assert_called_once()
    
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_delete_query_not_found(self, mock_get_json, client, mock_user_session):
        """Test deleting non-existent query."""
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        mock_get_json.return_value = None
        
        response = client.delete("/api/query/nonexistent-query")
        
        assert response.status_code == 404
        assert "Query not found" in response.json()["error"]["message"]
    
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_delete_query_access_denied(self, mock_get_json, client, mock_user_session):
        """Test deleting query from different user."""
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        
        other_user_query = {
            "query_id": "query-123",
            "user_id": "other-user",
            "query_text": "Other user's query",
            "status": "completed"
        }
        mock_get_json.return_value = other_user_query
        
        response = client.delete("/api/query/query-123")
        
        assert response.status_code == 403
        assert "Access denied" in response.json()["error"]["message"]


class TestQueryCacheInfo:
    """Test query cache information endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_user_session(self):
        """Create mock user session."""
        return UserSession(
            session_id="test-session-123",
            user_id="test-user",
            groups=["test-group", "common_rules"],
            permissions=["upload", "query", "delete"]
        )
    
    @patch('app.shared.redis_client.redis_client.get_json')
    @patch('app.workers.query_worker.generate_cache_key')
    def test_get_query_cache_info_with_cache(self, mock_generate_cache_key, mock_get_json, client, mock_user_session):
        """Test getting cache info for query with cached result."""
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        mock_generate_cache_key.return_value = "query_cache:test123"
        
        # Mock query data
        query_data = {
            "query_id": "query-123",
            "user_id": "test-user",
            "query_text": "What is the main topic?",
            "status": "completed",
            "result": {
                "cached": True
            }
        }
        
        # Mock cache data
        cache_data = {
            "result": {"answer": "Test answer"},
            "cached_at": 1642780800.0,  # 2022-01-21 12:00:00
            "expires_at": 1642784400.0   # 2022-01-21 13:00:00
        }
        
        def mock_get_json_side_effect(key):
            if "query:" in key:
                return query_data
            elif "query_cache:" in key:
                return cache_data
            return None
        
        mock_get_json.side_effect = mock_get_json_side_effect
        
        response = client.get("/api/query/query-123/cache")
        
        assert response.status_code == 200
        data = response.json()
        assert data["query_id"] == "query-123"
        assert data["cache_key"] == "query_cache:test123"
        assert data["was_cached"] is True
        assert data["cache_available"] is True
        assert data["cache_expires_at"] is not None
        assert data["cache_created_at"] is not None
    
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_get_query_cache_info_not_found(self, mock_get_json, client, mock_user_session):
        """Test getting cache info for non-existent query."""
        app.dependency_overrides[get_current_user] = lambda: mock_user_session
        mock_get_json.return_value = None
        
        response = client.get("/api/query/nonexistent-query/cache")
        
        assert response.status_code == 404
        assert "Query not found" in response.json()["error"]["message"]


class TestQueryUserIsolation:
    """Test user isolation in query operations."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def user1_session(self):
        """Create user1 session."""
        return UserSession(
            session_id="user1-session",
            user_id="user1",
            groups=["group1"],
            permissions=["upload", "query", "delete"]
        )
    
    @pytest.fixture
    def user2_session(self):
        """Create user2 session."""
        return UserSession(
            session_id="user2-session",
            user_id="user2",
            groups=["group2"],
            permissions=["upload", "query", "delete"]
        )
    
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_user_cannot_access_other_user_query(self, mock_get_json, client, user2_session):
        """Test that user2 cannot access user1's query."""
        app.dependency_overrides[get_current_user] = lambda: user2_session
        
        # Query belongs to user1
        user1_query = {
            "query_id": "user1-query",
            "user_id": "user1",
            "query_text": "User1's query",
            "status": "completed"
        }
        mock_get_json.return_value = user1_query
        
        response = client.get("/api/query/user1-query")
        
        assert response.status_code == 403
        assert "Access denied" in response.json()["error"]["message"]
    
    @patch('app.shared.redis_client.redis_client.get_connection')
    def test_user_only_sees_own_queries_in_listing(self, mock_get_connection, client, user1_session):
        """Test that user only sees their own queries in listing."""
        app.dependency_overrides[get_current_user] = lambda: user1_session
        
        # Mock Redis connection to return all query keys
        mock_client = Mock()
        mock_client.keys.return_value = [
            b"query:user1-query",
            b"query:user2-query"
        ]
        mock_get_connection.return_value.__enter__.return_value = mock_client
        
        # Mock query data - only user1's query should be included
        def mock_get_json_side_effect(key):
            if "user1-query" in key:
                return {
                    "query_id": "user1-query",
                    "user_id": "user1",
                    "query_text": "User1's query",
                    "status": "completed",
                    "created_at": "2025-01-21T10:00:00Z"
                }
            elif "user2-query" in key:
                return {
                    "query_id": "user2-query",
                    "user_id": "user2",  # Different user
                    "query_text": "User2's query",
                    "status": "completed",
                    "created_at": "2025-01-21T10:05:00Z"
                }
            return None
        
        with patch('app.shared.redis_client.redis_client.get_json', side_effect=mock_get_json_side_effect):
            response = client.get("/api/queries")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1  # Only user1's query
        assert data["queries"][0]["query_id"] == "user1-query"


if __name__ == "__main__":
    pytest.main([__file__])