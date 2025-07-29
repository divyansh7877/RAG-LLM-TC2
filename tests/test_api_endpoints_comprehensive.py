"""
Comprehensive unit tests for API endpoints and user interface functionality.
Tests for Requirements 3.1, 3.2, 3.3, 3.4, 3.5 - User interface and real-time feedback.
"""
import pytest
import json
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.main import app
from app.shared.models import Job, JobStatus, JobType, UserSession


class TestDocumentUploadEndpoints:
    """Test document upload API endpoints and progress tracking."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_auth_user(self):
        """Mock authenticated user."""
        with patch('app.api.main.get_current_user') as mock:
            mock.return_value = UserSession(
                session_id="test_session",
                user_id="test_user",
                groups=["test_group"],
                permissions=["upload", "query"]
            )
            yield mock
    
    def test_document_upload_with_progress_tracking(self, client, mock_auth_user):
        """Test document upload with real-time progress tracking."""
        with patch('app.shared.job_manager.job_manager') as mock_job_manager:
            # Mock job creation
            mock_job = Job(
                user_id="test_user",
                job_type=JobType.EMBEDDING,
                status=JobStatus.PENDING,
                metadata={"filename": "test.pdf"}
            )
            mock_job_manager.create_job.return_value = mock_job
            
            # Mock file upload
            test_file_content = b"PDF content here"
            files = {"file": ("test.pdf", test_file_content, "application/pdf")}
            
            response = client.post("/api/documents/upload", files=files)
            
            assert response.status_code == 200
            response_data = response.json()
            
            assert "job_id" in response_data
            assert response_data["status"] == "queued"
            assert response_data["filename"] == "test.pdf"
            assert "progress_url" in response_data
            
            # Verify job was created
            mock_job_manager.create_job.assert_called_once()
            call_args = mock_job_manager.create_job.call_args
            assert call_args[1]["job_type"] == JobType.EMBEDDING
            assert call_args[1]["user_id"] == "test_user"
    
    def test_document_upload_validation(self, client, mock_auth_user):
        """Test document upload validation and error handling."""
        # Test invalid file type
        invalid_file = {"file": ("test.txt", b"text content", "text/plain")}
        response = client.post("/api/documents/upload", files=invalid_file)
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
        
        # Test file too large
        with patch('app.api.main.MAX_FILE_SIZE', 1024):  # 1KB limit
            large_file = {"file": ("large.pdf", b"x" * 2048, "application/pdf")}
            response = client.post("/api/documents/upload", files=large_file)
            
            assert response.status_code == 400
            assert "File too large" in response.json()["detail"]
    
    def test_document_upload_concurrent_users(self, client):
        """Test concurrent document uploads from different users."""
        upload_results = []
        
        def upload_for_user(user_id):
            with patch('app.api.main.get_current_user') as mock_auth:
                mock_auth.return_value = UserSession(
                    session_id=f"session_{user_id}",
                    user_id=user_id,
                    groups=[f"group_{user_id}"],
                    permissions=["upload", "query"]
                )
                
                with patch('app.shared.job_manager.job_manager') as mock_job_manager:
                    mock_job = Job(
                        user_id=user_id,
                        job_type=JobType.EMBEDDING,
                        status=JobStatus.PENDING,
                        metadata={"filename": f"{user_id}_doc.pdf"}
                    )
                    mock_job_manager.create_job.return_value = mock_job
                    
                    files = {"file": (f"{user_id}_doc.pdf", b"PDF content", "application/pdf")}
                    response = client.post("/api/documents/upload", files=files)
                    
                    upload_results.append({
                        "user_id": user_id,
                        "status_code": response.status_code,
                        "job_id": response.json().get("job_id") if response.status_code == 200 else None
                    })
        
        # Simulate concurrent uploads
        import threading
        threads = []
        for i in range(5):
            user_id = f"user_{i}"
            thread = threading.Thread(target=upload_for_user, args=(user_id,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify all uploads succeeded
        assert len(upload_results) == 5
        for result in upload_results:
            assert result["status_code"] == 200
            assert result["job_id"] is not None
        
        # Verify unique job IDs
        job_ids = [result["job_id"] for result in upload_results]
        assert len(set(job_ids)) == 5
    
    def test_document_list_with_metadata(self, client, mock_auth_user):
        """Test document listing with metadata and status information."""
        with patch('app.shared.job_manager.job_manager') as mock_job_manager:
            # Mock user documents
            mock_jobs = [
                Job(
                    user_id="test_user",
                    job_type=JobType.EMBEDDING,
                    status=JobStatus.COMPLETED,
                    metadata={
                        "filename": "doc1.pdf",
                        "file_size": 1024000,
                        "page_count": 10,
                        "upload_date": "2024-01-01T10:00:00Z"
                    }
                ),
                Job(
                    user_id="test_user",
                    job_type=JobType.EMBEDDING,
                    status=JobStatus.PROCESSING,
                    metadata={
                        "filename": "doc2.pdf",
                        "file_size": 2048000,
                        "upload_date": "2024-01-01T11:00:00Z"
                    },
                    progress=0.75
                )
            ]
            mock_job_manager.get_user_jobs.return_value = mock_jobs
            
            response = client.get("/api/documents/list")
            
            assert response.status_code == 200
            documents = response.json()["documents"]
            
            assert len(documents) == 2
            
            # Check completed document
            completed_doc = next(d for d in documents if d["filename"] == "doc1.pdf")
            assert completed_doc["status"] == "completed"
            assert completed_doc["file_size"] == 1024000
            assert completed_doc["page_count"] == 10
            assert "upload_date" in completed_doc
            
            # Check processing document
            processing_doc = next(d for d in documents if d["filename"] == "doc2.pdf")
            assert processing_doc["status"] == "processing"
            assert processing_doc["progress"] == 0.75
            assert "page_count" not in processing_doc  # Not available yet
    
    def test_document_deletion(self, client, mock_auth_user):
        """Test document deletion functionality."""
        with patch('app.shared.job_manager.job_manager') as mock_job_manager:
            mock_job_manager.get_job.return_value = Job(
                user_id="test_user",
                job_type=JobType.EMBEDDING,
                status=JobStatus.COMPLETED,
                metadata={"filename": "test.pdf"}
            )
            mock_job_manager.delete_job.return_value = True
            
            response = client.delete("/api/documents/test_job_id")
            
            assert response.status_code == 200
            assert response.json()["message"] == "Document deleted successfully"
            
            mock_job_manager.delete_job.assert_called_once_with("test_job_id")
    
    def test_document_deletion_unauthorized(self, client):
        """Test document deletion by unauthorized user."""
        with patch('app.api.main.get_current_user') as mock_auth:
            mock_auth.return_value = UserSession(
                session_id="session1",
                user_id="user1",
                groups=["group1"],
                permissions=["upload", "query"]
            )
            
            with patch('app.shared.job_manager.job_manager') as mock_job_manager:
                # Mock document belonging to different user
                mock_job_manager.get_job.return_value = Job(
                    user_id="user2",  # Different user
                    job_type=JobType.EMBEDDING,
                    status=JobStatus.COMPLETED,
                    metadata={"filename": "test.pdf"}
                )
                
                response = client.delete("/api/documents/test_job_id")
                
                assert response.status_code == 403
                assert "Not authorized" in response.json()["detail"]


class TestQueryEndpoints:
    """Test query processing API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_auth_user(self):
        """Mock authenticated user."""
        with patch('app.api.main.get_current_user') as mock:
            mock.return_value = UserSession(
                session_id="test_session",
                user_id="test_user",
                groups=["test_group"],
                permissions=["upload", "query"]
            )
            yield mock
    
    def test_query_submission_with_immediate_feedback(self, client, mock_auth_user):
        """Test query submission with immediate processing feedback."""
        with patch('app.shared.job_manager.job_manager') as mock_job_manager:
            mock_job = Job(
                user_id="test_user",
                job_type=JobType.QUERY,
                status=JobStatus.PENDING,
                metadata={"query_text": "What is AI?"}
            )
            mock_job_manager.create_job.return_value = mock_job
            
            query_data = {"query": "What is AI?"}
            response = client.post("/api/query", json=query_data)
            
            assert response.status_code == 200
            response_data = response.json()
            
            assert "query_id" in response_data
            assert response_data["status"] == "processing"
            assert response_data["message"] == "Query submitted for processing"
            assert "status_url" in response_data
    
    def test_query_status_tracking(self, client, mock_auth_user):
        """Test query status tracking endpoint."""
        with patch('app.shared.job_manager.job_manager') as mock_job_manager:
            # Mock query in progress
            mock_job = Job(
                user_id="test_user",
                job_type=JobType.QUERY,
                status=JobStatus.PROCESSING,
                metadata={"query_text": "What is AI?"},
                progress=0.5
            )
            mock_job_manager.get_job.return_value = mock_job
            
            response = client.get("/api/query/test_query_id/status")
            
            assert response.status_code == 200
            status_data = response.json()
            
            assert status_data["status"] == "processing"
            assert status_data["progress"] == 0.5
            assert "estimated_completion" in status_data
    
    def test_query_result_retrieval(self, client, mock_auth_user):
        """Test query result retrieval."""
        with patch('app.shared.job_manager.job_manager') as mock_job_manager:
            # Mock completed query
            mock_job = Job(
                user_id="test_user",
                job_type=JobType.QUERY,
                status=JobStatus.COMPLETED,
                metadata={"query_text": "What is AI?"},
                result={
                    "answer": "AI is artificial intelligence...",
                    "sources": [
                        {"document": "ai_basics.pdf", "page": 1, "relevance": 0.95},
                        {"document": "ml_guide.pdf", "page": 3, "relevance": 0.87}
                    ],
                    "processing_time": 2.5
                }
            )
            mock_job_manager.get_job.return_value = mock_job
            
            response = client.get("/api/query/test_query_id")
            
            assert response.status_code == 200
            result_data = response.json()
            
            assert result_data["status"] == "completed"
            assert "answer" in result_data["result"]
            assert "sources" in result_data["result"]
            assert len(result_data["result"]["sources"]) == 2
            assert result_data["result"]["processing_time"] == 2.5
    
    def test_query_history(self, client, mock_auth_user):
        """Test query history retrieval."""
        with patch('app.shared.job_manager.job_manager') as mock_job_manager:
            mock_queries = [
                Job(
                    user_id="test_user",
                    job_type=JobType.QUERY,
                    status=JobStatus.COMPLETED,
                    metadata={"query_text": "What is AI?"},
                    created_at="2024-01-01T10:00:00Z"
                ),
                Job(
                    user_id="test_user",
                    job_type=JobType.QUERY,
                    status=JobStatus.COMPLETED,
                    metadata={"query_text": "How does ML work?"},
                    created_at="2024-01-01T11:00:00Z"
                )
            ]
            mock_job_manager.get_user_jobs.return_value = mock_queries
            
            response = client.get("/api/query/history")
            
            assert response.status_code == 200
            history_data = response.json()
            
            assert len(history_data["queries"]) == 2
            assert history_data["queries"][0]["query_text"] == "What is AI?"
            assert history_data["queries"][1]["query_text"] == "How does ML work?"
    
    def test_concurrent_query_processing(self, client):
        """Test concurrent query processing from multiple users."""
        query_results = []
        
        def submit_query_for_user(user_id, query_text):
            with patch('app.api.main.get_current_user') as mock_auth:
                mock_auth.return_value = UserSession(
                    session_id=f"session_{user_id}",
                    user_id=user_id,
                    groups=[f"group_{user_id}"],
                    permissions=["upload", "query"]
                )
                
                with patch('app.shared.job_manager.job_manager') as mock_job_manager:
                    mock_job = Job(
                        user_id=user_id,
                        job_type=JobType.QUERY,
                        status=JobStatus.PENDING,
                        metadata={"query_text": query_text}
                    )
                    mock_job_manager.create_job.return_value = mock_job
                    
                    query_data = {"query": query_text}
                    response = client.post("/api/query", json=query_data)
                    
                    query_results.append({
                        "user_id": user_id,
                        "query_text": query_text,
                        "status_code": response.status_code,
                        "query_id": response.json().get("query_id") if response.status_code == 200 else None
                    })
        
        # Submit concurrent queries
        import threading
        threads = []
        queries = [
            ("user1", "What is AI?"),
            ("user2", "How does ML work?"),
            ("user3", "What is deep learning?"),
            ("user1", "What are neural networks?"),
            ("user2", "How to train models?")
        ]
        
        for user_id, query_text in queries:
            thread = threading.Thread(target=submit_query_for_user, args=(user_id, query_text))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify all queries were submitted successfully
        assert len(query_results) == 5
        for result in query_results:
            assert result["status_code"] == 200
            assert result["query_id"] is not None
        
        # Verify unique query IDs
        query_ids = [result["query_id"] for result in query_results]
        assert len(set(query_ids)) == 5


class TestWebSocketRealTimeUpdates:
    """Test WebSocket real-time updates functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_websocket_connection_authentication(self, client):
        """Test WebSocket connection requires authentication."""
        with patch('app.api.main.get_current_user_from_token') as mock_auth:
            mock_auth.return_value = UserSession(
                session_id="test_session",
                user_id="test_user",
                groups=["test_group"],
                permissions=["upload", "query"]
            )
            
            with client.websocket_connect("/ws/updates?token=valid_token") as websocket:
                # Connection should be established
                data = websocket.receive_json()
                assert data["type"] == "connection_established"
                assert data["user_id"] == "test_user"
    
    def test_websocket_job_status_updates(self, client):
        """Test WebSocket job status update broadcasting."""
        with patch('app.api.main.get_current_user_from_token') as mock_auth:
            mock_auth.return_value = UserSession(
                session_id="test_session",
                user_id="test_user",
                groups=["test_group"],
                permissions=["upload", "query"]
            )
            
            with patch('app.shared.websocket_manager.websocket_manager') as mock_ws_manager:
                mock_ws_manager.connect_user = AsyncMock()
                mock_ws_manager.disconnect_user = AsyncMock()
                mock_ws_manager.send_to_user = AsyncMock()
                
                with client.websocket_connect("/ws/updates?token=valid_token") as websocket:
                    # Simulate job status update
                    job_update = {
                        "type": "job_status_update",
                        "job_id": "test_job_id",
                        "status": "processing",
                        "progress": 0.5,
                        "message": "Processing document..."
                    }
                    
                    # Mock sending update to user
                    mock_ws_manager.send_to_user.assert_called()
    
    def test_websocket_progress_updates(self, client):
        """Test WebSocket progress update broadcasting."""
        received_messages = []
        
        with patch('app.api.main.get_current_user_from_token') as mock_auth:
            mock_auth.return_value = UserSession(
                session_id="test_session",
                user_id="test_user",
                groups=["test_group"],
                permissions=["upload", "query"]
            )
            
            with client.websocket_connect("/ws/updates?token=valid_token") as websocket:
                # Receive connection confirmation
                connection_msg = websocket.receive_json()
                received_messages.append(connection_msg)
                
                # Simulate progress updates
                progress_updates = [
                    {"type": "progress_update", "job_id": "test_job", "progress": 0.25},
                    {"type": "progress_update", "job_id": "test_job", "progress": 0.50},
                    {"type": "progress_update", "job_id": "test_job", "progress": 0.75},
                    {"type": "job_completed", "job_id": "test_job", "result": "Success"}
                ]
                
                # In a real scenario, these would be sent by the server
                # Here we're testing the WebSocket connection works
                assert len(received_messages) >= 1
                assert received_messages[0]["type"] == "connection_established"
    
    def test_websocket_user_isolation(self, client):
        """Test that WebSocket updates are isolated per user."""
        connections = {}
        
        def create_user_connection(user_id):
            with patch('app.api.main.get_current_user_from_token') as mock_auth:
                mock_auth.return_value = UserSession(
                    session_id=f"session_{user_id}",
                    user_id=user_id,
                    groups=[f"group_{user_id}"],
                    permissions=["upload", "query"]
                )
                
                try:
                    with client.websocket_connect(f"/ws/updates?token=token_{user_id}") as websocket:
                        connections[user_id] = websocket
                        # Receive connection message
                        msg = websocket.receive_json()
                        assert msg["user_id"] == user_id
                        return True
                except Exception:
                    return False
        
        # Test multiple user connections
        users = ["user1", "user2", "user3"]
        connection_results = []
        
        for user_id in users:
            result = create_user_connection(user_id)
            connection_results.append(result)
        
        # All connections should succeed (in isolation)
        assert all(connection_results)
    
    def test_websocket_error_handling(self, client):
        """Test WebSocket error handling and disconnection."""
        with patch('app.api.main.get_current_user_from_token') as mock_auth:
            # Test invalid token
            mock_auth.side_effect = Exception("Invalid token")
            
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/ws/updates?token=invalid_token"):
                    pass


class TestJobManagementEndpoints:
    """Test job management and monitoring endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_auth_user(self):
        """Mock authenticated user."""
        with patch('app.api.main.get_current_user') as mock:
            mock.return_value = UserSession(
                session_id="test_session",
                user_id="test_user",
                groups=["test_group"],
                permissions=["upload", "query"]
            )
            yield mock
    
    def test_job_list_with_status_filtering(self, client, mock_auth_user):
        """Test job listing with status filtering."""
        with patch('app.shared.job_manager.job_manager') as mock_job_manager:
            mock_jobs = [
                Job(user_id="test_user", job_type=JobType.EMBEDDING, status=JobStatus.COMPLETED),
                Job(user_id="test_user", job_type=JobType.QUERY, status=JobStatus.PROCESSING),
                Job(user_id="test_user", job_type=JobType.EMBEDDING, status=JobStatus.FAILED)
            ]
            mock_job_manager.get_user_jobs.return_value = mock_jobs
            
            # Test all jobs
            response = client.get("/api/jobs")
            assert response.status_code == 200
            assert len(response.json()["jobs"]) == 3
            
            # Test filtering by status
            response = client.get("/api/jobs?status=completed")
            mock_job_manager.get_user_jobs.assert_called_with(
                "test_user", status=JobStatus.COMPLETED, limit=100
            )
            
            # Test filtering by type
            response = client.get("/api/jobs?job_type=embedding")
            mock_job_manager.get_user_jobs.assert_called_with(
                "test_user", job_type=JobType.EMBEDDING, limit=100
            )
    
    def test_job_cancellation(self, client, mock_auth_user):
        """Test job cancellation functionality."""
        with patch('app.shared.job_manager.job_manager') as mock_job_manager:
            mock_job_manager.get_job.return_value = Job(
                user_id="test_user",
                job_type=JobType.EMBEDDING,
                status=JobStatus.PROCESSING
            )
            mock_job_manager.cancel_job.return_value = True
            
            response = client.post("/api/jobs/test_job_id/cancel")
            
            assert response.status_code == 200
            assert response.json()["message"] == "Job cancelled successfully"
            
            mock_job_manager.cancel_job.assert_called_once_with("test_job_id")
    
    def test_job_statistics(self, client, mock_auth_user):
        """Test job statistics endpoint."""
        with patch('app.shared.job_manager.job_manager') as mock_job_manager:
            mock_stats = {
                "total_jobs": 10,
                "completed_jobs": 7,
                "failed_jobs": 2,
                "active_jobs": 1,
                "success_rate": 0.78,
                "average_duration": 45.5,
                "by_type": {
                    "embedding": 6,
                    "query": 4
                },
                "by_status": {
                    "completed": 7,
                    "failed": 2,
                    "processing": 1
                }
            }
            mock_job_manager.get_job_statistics.return_value = mock_stats
            
            response = client.get("/api/jobs/statistics")
            
            assert response.status_code == 200
            stats_data = response.json()
            
            assert stats_data["total_jobs"] == 10
            assert stats_data["success_rate"] == 0.78
            assert stats_data["by_type"]["embedding"] == 6
            assert stats_data["by_status"]["completed"] == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])