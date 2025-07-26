"""
Tests for document management API endpoints.
"""
import pytest
import tempfile
import shutil
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
import io

from app.api.main import app
from app.shared.models import UserSession, Document, Job, JobStatus, JobType
from app.shared.redis_client import redis_client
from app.shared.job_manager import job_manager
from app.shared.middleware import get_current_user


class TestDocumentUpload:
    """Test document upload endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def authenticated_client(self, mock_user_session):
        """Create test client with mocked authentication."""
        def mock_get_current_user():
            return mock_user_session
        
        app.dependency_overrides[get_current_user] = mock_get_current_user
        client = TestClient(app)
        yield client
        # Clean up
        app.dependency_overrides.clear()
    
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
    def sample_pdf_content(self):
        """Create sample PDF content for testing."""
        # This is a minimal PDF content for testing
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n179\n%%EOF"
    
    @patch('app.shared.job_manager.job_manager.create_job')
    @patch('app.workers.embedding_worker.process_document_embedding.delay')
    def test_upload_documents_success(self, mock_delay, mock_create_job, authenticated_client, mock_user_session, sample_pdf_content):
        """Test successful document upload."""
        # Setup mocks
        
        mock_job = Job(
            job_id="test-job-123",
            user_id="test-user",
            job_type=JobType.EMBEDDING,
            status=JobStatus.PENDING
        )
        mock_create_job.return_value = mock_job
        
        mock_task = Mock()
        mock_task.id = "celery-task-123"
        mock_delay.return_value = mock_task
        
        # Create test file
        files = [
            ("files", ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf"))
        ]
        
        # Make request
        response = authenticated_client.post(
            "/api/documents/upload",
            files=files,
            data={"group_id": "test-group"}
        )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["files_count"] == 1
        assert data["filenames"] == ["test.pdf"]
        assert data["status"] == "queued"
        
        # Verify mocks were called
        mock_create_job.assert_called_once()
        mock_delay.assert_called_once()
    
    @patch('app.shared.middleware.get_current_user')
    def test_upload_documents_no_upload_permission(self, mock_get_user, client, sample_pdf_content):
        """Test upload with insufficient permissions."""
        # User without upload permission
        user_session = UserSession(
            session_id="test-session-123",
            user_id="test-user",
            groups=["test-group"],
            permissions=["query"]  # No upload permission
        )
        mock_get_user.return_value = user_session
        
        files = [
            ("files", ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf"))
        ]
        
        response = client.post(
            "/api/documents/upload",
            files=files,
            data={"group_id": "test-group"}
        )
        
        assert response.status_code == 403
        assert "upload permission" in response.json()["error"]["message"]
    
    @patch('app.shared.middleware.get_current_user')
    def test_upload_documents_invalid_group(self, mock_get_user, client, mock_user_session, sample_pdf_content):
        """Test upload to group user doesn't have access to."""
        mock_get_user.return_value = mock_user_session
        
        files = [
            ("files", ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf"))
        ]
        
        response = client.post(
            "/api/documents/upload",
            files=files,
            data={"group_id": "invalid-group"}
        )
        
        assert response.status_code == 403
        assert "does not have access to group" in response.json()["error"]["message"]
    
    @patch('app.shared.middleware.get_current_user')
    def test_upload_documents_invalid_file_type(self, mock_get_user, client, mock_user_session):
        """Test upload with non-PDF file."""
        mock_get_user.return_value = mock_user_session
        
        files = [
            ("files", ("test.txt", io.BytesIO(b"test content"), "text/plain"))
        ]
        
        response = client.post(
            "/api/documents/upload",
            files=files,
            data={"group_id": "test-group"}
        )
        
        assert response.status_code == 400
        assert "Only PDF files are supported" in response.json()["error"]["message"]
    
    @patch('app.shared.middleware.get_current_user')
    def test_upload_documents_file_too_large(self, mock_get_user, client, mock_user_session):
        """Test upload with file exceeding size limit."""
        mock_get_user.return_value = mock_user_session
        
        # Create a large file (simulate 51MB)
        large_content = b"x" * (51 * 1024 * 1024)
        
        # Create a mock file with size property
        mock_file = Mock()
        mock_file.filename = "large.pdf"
        mock_file.size = 51 * 1024 * 1024
        mock_file.file = io.BytesIO(large_content)
        
        files = [
            ("files", ("large.pdf", io.BytesIO(large_content), "application/pdf"))
        ]
        
        # Mock the file size check in the endpoint
        with patch('builtins.hasattr', return_value=True), \
             patch.object(type(files[0][1][1]), 'size', 51 * 1024 * 1024, create=True):
            response = client.post(
                "/api/documents/upload",
                files=files,
                data={"group_id": "test-group"}
            )
        
        assert response.status_code == 413
        assert "File too large" in response.json()["error"]["message"]
    
    @patch('app.shared.middleware.get_current_user')
    def test_upload_documents_too_many_files(self, mock_get_user, client, mock_user_session, sample_pdf_content):
        """Test upload with too many files."""
        mock_get_user.return_value = mock_user_session
        
        # Create 11 files (exceeds limit of 10)
        files = [
            ("files", (f"test{i}.pdf", io.BytesIO(sample_pdf_content), "application/pdf"))
            for i in range(11)
        ]
        
        response = client.post(
            "/api/documents/upload",
            files=files,
            data={"group_id": "test-group"}
        )
        
        assert response.status_code == 400
        assert "Maximum 10 files allowed" in response.json()["error"]["message"]


class TestDocumentListing:
    """Test document listing endpoint."""
    
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
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            {
                "document_id": "doc-1",
                "user_id": "test-user",
                "group_id": "test-group",
                "filename": "test1.pdf",
                "file_size": 1024,
                "upload_date": "2025-01-21T10:00:00Z",
                "processing_status": "completed",
                "page_count": 5,
                "chunk_count": 20
            },
            {
                "document_id": "doc-2",
                "user_id": "test-user",
                "group_id": "common_rules",
                "filename": "test2.pdf",
                "file_size": 2048,
                "upload_date": "2025-01-21T11:00:00Z",
                "processing_status": "processing",
                "page_count": 3,
                "chunk_count": 15
            }
        ]
    
    @patch('app.shared.middleware.get_current_user')
    @patch('app.shared.redis_client.redis_client.get_connection')
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_list_documents_success(self, mock_get_json, mock_get_connection, mock_get_user, client, mock_user_session, sample_documents):
        """Test successful document listing."""
        mock_get_user.return_value = mock_user_session
        
        # Mock Redis connection and keys
        mock_client = Mock()
        mock_client.keys.return_value = [
            b"document:test-user:test-group:doc-1",
            b"document:test-user:common_rules:doc-2"
        ]
        mock_get_connection.return_value.__enter__.return_value = mock_client
        
        # Mock document data retrieval
        def mock_get_json_side_effect(key):
            if "doc-1" in key:
                return sample_documents[0]
            elif "doc-2" in key:
                return sample_documents[1]
            return None
        
        mock_get_json.side_effect = mock_get_json_side_effect
        
        response = client.get("/api/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        assert len(data["documents"]) == 2
        assert data["documents"][0]["filename"] == "test2.pdf"  # Sorted by upload date, newest first
        assert data["documents"][1]["filename"] == "test1.pdf"
    
    @patch('app.shared.middleware.get_current_user')
    @patch('app.shared.redis_client.redis_client.get_connection')
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_list_documents_with_group_filter(self, mock_get_json, mock_get_connection, mock_get_user, client, mock_user_session, sample_documents):
        """Test document listing with group filter."""
        mock_get_user.return_value = mock_user_session
        
        # Mock Redis connection and keys for specific group
        mock_client = Mock()
        mock_client.keys.return_value = [b"document:test-user:test-group:doc-1"]
        mock_get_connection.return_value.__enter__.return_value = mock_client
        
        mock_get_json.return_value = sample_documents[0]
        
        response = client.get("/api/documents?group_id=test-group")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert data["documents"][0]["group_id"] == "test-group"
    
    @patch('app.shared.middleware.get_current_user')
    def test_list_documents_invalid_group(self, mock_get_user, client, mock_user_session):
        """Test document listing with invalid group filter."""
        mock_get_user.return_value = mock_user_session
        
        response = client.get("/api/documents?group_id=invalid-group")
        
        assert response.status_code == 403
        assert "does not have access to group" in response.json()["error"]["message"]
    
    @patch('app.shared.middleware.get_current_user')
    @patch('app.shared.redis_client.redis_client.get_connection')
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_list_documents_with_status_filter(self, mock_get_json, mock_get_connection, mock_get_user, client, mock_user_session, sample_documents):
        """Test document listing with status filter."""
        mock_get_user.return_value = mock_user_session
        
        mock_client = Mock()
        mock_client.keys.return_value = [
            b"document:test-user:test-group:doc-1",
            b"document:test-user:common_rules:doc-2"
        ]
        mock_get_connection.return_value.__enter__.return_value = mock_client
        
        def mock_get_json_side_effect(key):
            if "doc-1" in key:
                return sample_documents[0]  # completed status
            elif "doc-2" in key:
                return sample_documents[1]  # processing status
            return None
        
        mock_get_json.side_effect = mock_get_json_side_effect
        
        response = client.get("/api/documents?status=completed")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert data["documents"][0]["processing_status"] == "completed"
    
    @patch('app.shared.middleware.get_current_user')
    @patch('app.shared.redis_client.redis_client.get_connection')
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_list_documents_pagination(self, mock_get_json, mock_get_connection, mock_get_user, client, mock_user_session, sample_documents):
        """Test document listing with pagination."""
        mock_get_user.return_value = mock_user_session
        
        mock_client = Mock()
        mock_client.keys.return_value = [
            b"document:test-user:test-group:doc-1",
            b"document:test-user:common_rules:doc-2"
        ]
        mock_get_connection.return_value.__enter__.return_value = mock_client
        
        def mock_get_json_side_effect(key):
            if "doc-1" in key:
                return sample_documents[0]
            elif "doc-2" in key:
                return sample_documents[1]
            return None
        
        mock_get_json.side_effect = mock_get_json_side_effect
        
        response = client.get("/api/documents?limit=1&offset=0")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 2
        assert len(data["documents"]) == 1
        assert data["has_more"] is True
        assert data["limit"] == 1
        assert data["offset"] == 0


class TestDocumentRetrieval:
    """Test document retrieval endpoint."""
    
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
    def sample_document(self):
        """Create sample document for testing."""
        return {
            "document_id": "doc-123",
            "user_id": "test-user",
            "group_id": "test-group",
            "filename": "test.pdf",
            "file_size": 1024,
            "upload_date": "2025-01-21T10:00:00Z",
            "processing_status": "completed",
            "page_count": 5,
            "chunk_count": 20
        }
    
    @patch('app.shared.middleware.get_current_user')
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_get_document_success(self, mock_get_json, mock_get_user, client, mock_user_session, sample_document):
        """Test successful document retrieval."""
        mock_get_user.return_value = mock_user_session
        mock_get_json.return_value = sample_document
        
        response = client.get("/api/documents/doc-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "doc-123"
        assert data["filename"] == "test.pdf"
        assert data["processing_status"] == "completed"
    
    @patch('app.shared.middleware.get_current_user')
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_get_document_not_found(self, mock_get_json, mock_get_user, client, mock_user_session):
        """Test document retrieval when document doesn't exist."""
        mock_get_user.return_value = mock_user_session
        mock_get_json.return_value = None
        
        response = client.get("/api/documents/nonexistent-doc")
        
        assert response.status_code == 404
        assert "Document not found" in response.json()["error"]["message"]


class TestDocumentDeletion:
    """Test document deletion endpoint."""
    
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
    def sample_document(self):
        """Create sample document for testing."""
        return {
            "document_id": "doc-123",
            "user_id": "test-user",
            "group_id": "test-group",
            "filename": "test.pdf",
            "file_size": 1024,
            "upload_date": "2025-01-21T10:00:00Z",
            "processing_status": "completed",
            "page_count": 5,
            "chunk_count": 20
        }
    
    @patch('app.shared.middleware.get_current_user')
    @patch('app.shared.redis_client.redis_client.get_json')
    @patch('app.shared.redis_client.redis_client.delete')
    def test_delete_document_success(self, mock_delete, mock_get_json, mock_get_user, client, mock_user_session, sample_document):
        """Test successful document deletion."""
        mock_get_user.return_value = mock_user_session
        mock_get_json.return_value = sample_document
        mock_delete.return_value = True
        
        response = client.delete("/api/documents/doc-123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Document deleted successfully"
        assert data["document_id"] == "doc-123"
        assert data["filename"] == "test.pdf"
        
        # Verify delete was called
        mock_delete.assert_called_once()
    
    @patch('app.shared.middleware.get_current_user')
    def test_delete_document_no_permission(self, mock_get_user, client):
        """Test document deletion without delete permission."""
        user_session = UserSession(
            session_id="test-session-123",
            user_id="test-user",
            groups=["test-group"],
            permissions=["upload", "query"]  # No delete permission
        )
        mock_get_user.return_value = user_session
        
        response = client.delete("/api/documents/doc-123")
        
        assert response.status_code == 403
        assert "delete permission" in response.json()["error"]["message"]
    
    @patch('app.shared.middleware.get_current_user')
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_delete_document_not_found(self, mock_get_json, mock_get_user, client, mock_user_session):
        """Test document deletion when document doesn't exist."""
        mock_get_user.return_value = mock_user_session
        mock_get_json.return_value = None
        
        response = client.delete("/api/documents/nonexistent-doc")
        
        assert response.status_code == 404
        assert "Document not found" in response.json()["error"]["message"]


class TestDocumentStatus:
    """Test document status endpoint."""
    
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
    def sample_document(self):
        """Create sample document for testing."""
        return {
            "document_id": "doc-123",
            "user_id": "test-user",
            "group_id": "test-group",
            "filename": "test.pdf",
            "file_size": 1024,
            "upload_date": "2025-01-21T10:00:00Z",
            "processing_status": "completed",
            "page_count": 5,
            "chunk_count": 20
        }
    
    @pytest.fixture
    def sample_job(self):
        """Create sample job for testing."""
        return Job(
            job_id="job-123",
            user_id="test-user",
            job_type=JobType.EMBEDDING,
            status=JobStatus.COMPLETED,
            progress=1.0,
            metadata={
                "group_id": "test-group",
                "filenames": ["test.pdf"]
            }
        )
    
    @patch('app.shared.middleware.get_current_user')
    @patch('app.shared.redis_client.redis_client.get_json')
    @patch('app.shared.job_manager.job_manager.get_user_jobs')
    def test_get_document_status_with_job(self, mock_get_user_jobs, mock_get_json, mock_get_user, client, mock_user_session, sample_document, sample_job):
        """Test document status retrieval with associated job."""
        mock_get_user.return_value = mock_user_session
        mock_get_json.return_value = sample_document
        mock_get_user_jobs.return_value = [sample_job]
        
        response = client.get("/api/documents/doc-123/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "doc-123"
        assert data["filename"] == "test.pdf"
        assert data["processing_status"] == "completed"
        assert data["job_info"] is not None
        assert data["job_info"]["job_id"] == "job-123"
        assert data["job_info"]["status"] == "completed"
    
    @patch('app.shared.middleware.get_current_user')
    @patch('app.shared.redis_client.redis_client.get_json')
    @patch('app.shared.job_manager.job_manager.get_user_jobs')
    def test_get_document_status_without_job(self, mock_get_user_jobs, mock_get_json, mock_get_user, client, mock_user_session, sample_document):
        """Test document status retrieval without associated job."""
        mock_get_user.return_value = mock_user_session
        mock_get_json.return_value = sample_document
        mock_get_user_jobs.return_value = []
        
        response = client.get("/api/documents/doc-123/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "doc-123"
        assert data["job_info"] is None
    
    @patch('app.shared.middleware.get_current_user')
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_get_document_status_not_found(self, mock_get_json, mock_get_user, client, mock_user_session):
        """Test document status retrieval when document doesn't exist."""
        mock_get_user.return_value = mock_user_session
        mock_get_json.return_value = None
        
        response = client.get("/api/documents/nonexistent-doc/status")
        
        assert response.status_code == 404
        assert "Document not found" in response.json()["error"]["message"]


class TestDocumentUserIsolation:
    """Test user isolation in document operations."""
    
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
    
    @pytest.fixture
    def user1_document(self):
        """Create document owned by user1."""
        return {
            "document_id": "user1-doc",
            "user_id": "user1",
            "group_id": "group1",
            "filename": "user1.pdf",
            "file_size": 1024,
            "upload_date": "2025-01-21T10:00:00Z",
            "processing_status": "completed"
        }
    
    @patch('app.shared.middleware.get_current_user')
    @patch('app.shared.redis_client.redis_client.get_json')
    def test_user_cannot_access_other_user_document(self, mock_get_json, mock_get_user, client, user2_session, user1_document):
        """Test that user2 cannot access user1's document."""
        mock_get_user.return_value = user2_session
        
        # Mock that no document is found for user2's groups
        mock_get_json.return_value = None
        
        response = client.get("/api/documents/user1-doc")
        
        assert response.status_code == 404
        assert "Document not found" in response.json()["error"]["message"]
    
    @patch('app.shared.middleware.get_current_user')
    @patch('app.shared.redis_client.redis_client.get_connection')
    def test_user_only_sees_own_documents_in_listing(self, mock_get_connection, mock_get_user, client, user1_session):
        """Test that user only sees their own documents in listing."""
        mock_get_user.return_value = user1_session
        
        # Mock Redis connection to only return keys for user1's documents
        mock_client = Mock()
        mock_client.keys.return_value = [b"document:user1:group1:user1-doc"]
        mock_get_connection.return_value.__enter__.return_value = mock_client
        
        response = client.get("/api/documents")
        
        assert response.status_code == 200
        # The keys method should only be called with patterns for user1's groups
        expected_pattern = "document:user1:group1:*"
        mock_client.keys.assert_called_with(expected_pattern)


if __name__ == "__main__":
    pytest.main([__file__])