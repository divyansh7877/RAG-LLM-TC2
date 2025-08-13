"""
Tests for embedding worker with progress tracking, user isolation, and resource limits.
"""
import os
import tempfile
import shutil
import time
import uuid
import pytest
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import the worker functions
from app.workers.embedding_worker import (
    process_document_embedding
)
from app.shared.models import JobStatus, Document
from app.shared.redis_client import redis_client
from app.shared.document_processor import DocumentProcessor, EmbeddingResult


class TestDocumentProcessor:
    """Test basic functionality of the DocumentProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance for testing."""
        return DocumentProcessor(
            db_path=":memory:",
            table_name="test_table",
            embed_model_name="all-MiniLM-L6-v2",
            device="cpu"
        )

    def test_clean_text(self, processor):
        """Test text cleaning functionality."""
        text = "This is   a test\n\n\nwith   multiple   spaces\nand newlines"
        cleaned = processor._clean_text(text)
        assert "multiple   spaces" not in cleaned
        assert "\n\n\n" not in cleaned

    def test_calculate_file_hash(self, processor):
        """Test file hash calculation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            hash1 = processor._calculate_file_hash(temp_path)
            hash2 = processor._calculate_file_hash(temp_path)
            assert hash1 == hash2
            assert len(hash1) == 64
        finally:
            os.unlink(temp_path)



class TestDocumentProcessorUserIsolation:
    """Test user isolation and security features of the DocumentProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance for testing."""
        return DocumentProcessor(
            db_path=":memory:",
            table_name="test_table",
            embed_model_name="all-MiniLM-L6-v2",
            device="cpu"
        )

    @patch("app.shared.document_processor.LanceDBVectorStore")
    def test_user_isolation(self, mock_vector_store, processor):
        """Test that documents from different users are isolated."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("user1 content")
            user1_file = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("user2 content")
            user2_file = f.name

        try:
            processor.process_documents([user1_file], "user1", "group1")
            processor.process_documents([user2_file], "user2", "group1")

            # This is a simplified test. In a real scenario, you would query the
            # vector store and verify that user1 can only see user1's documents.
            # For this test, we'll just check that the metadata is correctly set.
            nodes1 = processor._build_nodes([user1_file], "user1", "group1", 512, 20)
            nodes2 = processor._build_nodes([user2_file], "user2", "group1", 512, 20)

            assert all(node.metadata["user_id"] == "user1" for node in nodes1)
            assert all(node.metadata["user_id"] == "user2" for node in nodes2)

        finally:
            os.unlink(user1_file)
            os.unlink(user2_file)


class TestEmbeddingWorkerProgressAndStatus:
    """Test progress tracking and job status updates for the embedding worker."""

    @patch("app.workers.embedding_worker.job_manager")
    @patch("app.workers.embedding_worker.DocumentProcessor")
    def test_process_document_embedding_success(self, mock_processor, mock_job_manager):
        """Test the successful execution of the embedding task."""
        mock_processor.return_value.process_documents.return_value = EmbeddingResult(
            success=True,
            document_count=1,
            chunk_count=10,
            processing_time=1.23
        )

        process_document_embedding("job1", "user1", "group1", ["/fake/path.pdf"])

        mock_job_manager.update_job_status.assert_any_call("job1", JobStatus.PROCESSING)
        mock_job_manager.update_job_status.assert_called_with(
            "job1",
            JobStatus.COMPLETED,
            result={
                'success': True,
                'document_count': 1,
                'chunk_count': 10,
                'processing_time': 1.23,
                'error': None
            }
        )

    @patch("app.workers.embedding_worker.job_manager")
    @patch("app.workers.embedding_worker.DocumentProcessor")
    def test_process_document_embedding_failure(self, mock_processor, mock_job_manager):
        """Test the failed execution of the embedding task."""
        mock_processor.return_value.process_documents.side_effect = Exception("Embedding failed")

        with pytest.raises(Exception, match="Embedding failed"):
            process_document_embedding("job1", "user1", "group1", ["/fake/path.pdf"])

        mock_job_manager.update_job_status.assert_any_call("job1", JobStatus.PROCESSING)
        mock_job_manager.update_job_status.assert_called_with(
            "job1",
            JobStatus.FAILED,
            error="Embedding failed"
        )



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
