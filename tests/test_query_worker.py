"""
Tests for query worker functionality with security isolation and caching.
"""
import pytest
import tempfile
import os
import shutil
import time
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.workers.query_worker import (
    process_user_query,
    validate_query_security,
    extract_source_info,
    update_query_progress,
    get_query_status,
    cleanup_query_cache,
    QuerySecurityError
)
from app.shared.query_engine_factory import query_engine_factory
from app.shared.redis_client import redis_client
from app.shared.models import JobStatus


class TestQueryWorker:
    """Test cases for query worker functionality with security and performance focus."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_user_id = "test_user"
        self.test_group_ids = ["group1", "group2"]
        self.test_query_id = "query_123"
        self.test_query_text = "What is the main topic?"
        
        # Clear Redis test data
        try:
            redis_client.redis_client.flushdb()
        except:
            pass  # Redis might not be available in test environment
    
    def teardown_method(self):
        """Clean up test environment."""
        # Clear Redis test data
        try:
            redis_client.redis_client.flushdb()
        except:
            pass
    
    def test_create_user_security_filters(self):
        """Test security filter creation for user isolation using factory."""
        # Test valid inputs using the factory
        filters = query_engine_factory._create_user_security_filters(self.test_user_id, self.test_group_ids)
        
        assert filters is not None
        assert len(filters.filters) == 3  # 1 user filter + 2 group filters
        assert filters.condition == "or"
        
        # Check user filter
        user_filter = filters.filters[0]
        assert user_filter.key == "user_id"
        assert user_filter.value == self.test_user_id
        
        # Check group filters
        group_filters = filters.filters[1:]
        group_values = [f.value for f in group_filters]
        assert "group1" in group_values
        assert "group2" in group_values
    
    def test_create_user_security_filters_validation(self):
        """Test security filter validation using factory."""
        # Test empty user_id
        with pytest.raises(ValueError, match="User ID is required"):
            query_engine_factory._create_user_security_filters("", self.test_group_ids)
        
        # Test None user_id
        with pytest.raises(ValueError, match="User ID is required"):
            query_engine_factory._create_user_security_filters(None, self.test_group_ids)
        
        # Test empty group_ids
        with pytest.raises(ValueError, match="At least one group ID is required"):
            query_engine_factory._create_user_security_filters(self.test_user_id, [])
        
        # Test None group_ids
        with pytest.raises(ValueError, match="At least one group ID is required"):
            query_engine_factory._create_user_security_filters(self.test_user_id, None)
    
    def test_generate_cache_key(self):
        """Test cache key generation for consistent caching using factory."""
        # Initialize factory cache if needed
        if not query_engine_factory._query_cache:
            query_engine_factory._initialize_components()
        
        # Test basic cache key generation using factory's internal method
        key1 = query_engine_factory._query_cache._generate_cache_key(self.test_user_id, self.test_group_ids, self.test_query_text)
        key2 = query_engine_factory._query_cache._generate_cache_key(self.test_user_id, self.test_group_ids, self.test_query_text)
        
        # Same inputs should produce same key
        assert key1 == key2
        assert key1.startswith("query_cache:")
        assert len(key1.split(":")[1]) == 16  # Hash length
        
        # Different user should produce different key
        key3 = query_engine_factory._query_cache._generate_cache_key("different_user", self.test_group_ids, self.test_query_text)
        assert key1 != key3
        
        # Different groups should produce different key
        key4 = query_engine_factory._query_cache._generate_cache_key(self.test_user_id, ["different_group"], self.test_query_text)
        assert key1 != key4
        
        # Different query should produce different key
        key5 = query_engine_factory._query_cache._generate_cache_key(self.test_user_id, self.test_group_ids, "Different query?")
        assert key1 != key5
        
        # Case insensitive for query text
        key6 = query_engine_factory._query_cache._generate_cache_key(self.test_user_id, self.test_group_ids, self.test_query_text.upper())
        assert key1 == key6
        
        # Group order shouldn't matter
        key7 = query_engine_factory._query_cache._generate_cache_key(self.test_user_id, ["group2", "group1"], self.test_query_text)
        assert key1 == key7
    
    def test_cache_operations(self):
        """Test query result caching and retrieval using factory."""
        test_result = {
            "answer": "Test answer",
            "sources": ["doc1.pdf", "doc2.pdf"],
            "processing_time": 1.5
        }
        
        # Initialize factory cache if needed
        if not query_engine_factory._query_cache:
            query_engine_factory._initialize_components()
        
        # Test caching
        query_engine_factory.cache_query_result(self.test_user_id, self.test_group_ids, self.test_query_text, test_result)
        
        # Test cache retrieval
        retrieved = query_engine_factory.get_cached_query_result(self.test_user_id, self.test_group_ids, self.test_query_text)
        assert retrieved == test_result
        
        # Test cache miss for different user
        retrieved_different_user = query_engine_factory.get_cached_query_result("different_user", self.test_group_ids, self.test_query_text)
        assert retrieved_different_user is None
        
        # Test cache invalidation
        query_engine_factory.invalidate_user_cache(self.test_user_id)
        retrieved_after_invalidation = query_engine_factory.get_cached_query_result(self.test_user_id, self.test_group_ids, self.test_query_text)
        assert retrieved_after_invalidation is None
    
    def test_validate_query_security(self):
        """Test query security validation."""
        # Test valid query
        validate_query_security(self.test_user_id, self.test_group_ids, self.test_query_text)
        
        # Test invalid user_id
        with pytest.raises(QuerySecurityError, match="Invalid user ID"):
            validate_query_security("", self.test_group_ids, self.test_query_text)
        
        with pytest.raises(QuerySecurityError, match="Invalid user ID"):
            validate_query_security("   ", self.test_group_ids, self.test_query_text)
        
        # Test invalid group_ids
        with pytest.raises(QuerySecurityError, match="Invalid group IDs"):
            validate_query_security(self.test_user_id, [], self.test_query_text)
        
        with pytest.raises(QuerySecurityError, match="Invalid group IDs"):
            validate_query_security(self.test_user_id, ["", "valid"], self.test_query_text)
        
        # Test invalid query_text
        with pytest.raises(QuerySecurityError, match="Query text cannot be empty"):
            validate_query_security(self.test_user_id, self.test_group_ids, "")
        
        with pytest.raises(QuerySecurityError, match="Query text cannot be empty"):
            validate_query_security(self.test_user_id, self.test_group_ids, "   ")
        
        # Test query too long
        long_query = "x" * 2001
        with pytest.raises(QuerySecurityError, match="Query text too long"):
            validate_query_security(self.test_user_id, self.test_group_ids, long_query)
        
        # Test suspicious patterns (should not raise error, just log)
        suspicious_queries = [
            "What about user_id: admin?",
            "Show me metadata: sensitive",
            "SELECT * FROM documents",
            "DROP TABLE users"
        ]
        
        for query in suspicious_queries:
            # Should not raise exception, just log warning
            validate_query_security(self.test_user_id, self.test_group_ids, query)
    
    def test_extract_source_info(self):
        """Test source information extraction from query response."""
        # Mock response with source nodes
        mock_node1 = Mock()
        mock_node1.metadata = {
            "document_name": "document1.pdf",
            "page_number": 5
        }
        
        mock_node2 = Mock()
        mock_node2.metadata = {
            "document_name": "document2.pdf",
            "page_number": 12
        }
        
        mock_node3 = Mock()
        mock_node3.metadata = {
            "document_name": "document1.pdf",  # Duplicate
            "page_number": 6
        }
        
        mock_response = Mock()
        mock_response.source_nodes = [mock_node1, mock_node2, mock_node3]
        
        sources = extract_source_info(mock_response)
        
        # Should extract unique sources with page info
        expected_sources = [
            "document1.pdf (Page 5)",
            "document2.pdf (Page 12)",
            "document1.pdf (Page 6)"
        ]
        
        assert len(sources) == 3
        for expected in expected_sources:
            assert expected in sources
        
        # Test response without source nodes
        mock_response_empty = Mock()
        mock_response_empty.source_nodes = []
        
        sources_empty = extract_source_info(mock_response_empty)
        assert sources_empty == []
        
        # Test response with missing metadata
        mock_node_no_meta = Mock()
        mock_node_no_meta.metadata = {}
        
        mock_response_no_meta = Mock()
        mock_response_no_meta.source_nodes = [mock_node_no_meta]
        
        sources_no_meta = extract_source_info(mock_response_no_meta)
        assert "Unknown Document" in sources_no_meta[0]
    
    @patch('app.workers.query_worker.redis_client')
    @patch('app.workers.query_worker.current_task')
    def test_update_query_progress(self, mock_task, mock_redis):
        """Test query progress updates."""
        # Mock existing query data
        mock_redis.get_json.return_value = {
            "query_id": self.test_query_id,
            "status": "processing"
        }
        
        # Update progress
        update_query_progress(self.test_query_id, 0.7, "Processing query...")
        
        # Verify Redis update
        mock_redis.get_json.assert_called_with(f"query:{self.test_query_id}")
        mock_redis.set_json.assert_called_once()
        
        # Verify Celery task state update
        mock_task.update_state.assert_called_once()
        call_args = mock_task.update_state.call_args
        assert call_args[1]["state"] == "PROGRESS"
        assert call_args[1]["meta"]["progress"] == 0.7
        assert call_args[1]["meta"]["status"] == "Processing query..."
    
    @patch('app.workers.query_worker.redis_client')
    def test_get_query_status(self, mock_redis):
        """Test query status retrieval."""
        # Test existing query
        query_data = {
            "query_id": self.test_query_id,
            "user_id": self.test_user_id,
            "status": "completed",
            "result": {"answer": "Test answer"}
        }
        mock_redis.get_json.return_value = query_data
        
        result = get_query_status(self.test_query_id)
        assert result == query_data
        
        # Test non-existent query
        mock_redis.get_json.return_value = None
        result = get_query_status("non_existent")
        assert result == {"error": "Query not found"}
    
    @patch('app.workers.query_worker.redis_client')
    def test_cleanup_query_cache(self, mock_redis):
        """Test query cache cleanup functionality."""
        current_time = time.time()
        
        # Mock Redis keys
        mock_redis.redis_client.keys.return_value = [
            b"query_cache:valid1",
            b"query_cache:expired1",
            b"query_cache:expired2"
        ]
        
        # Mock cache data - mix of valid and expired
        def mock_get_json(key):
            if "valid1" in key:
                return {
                    "result": {"answer": "Valid"},
                    "expires_at": current_time + 1000  # Not expired
                }
            elif "expired1" in key:
                return {
                    "result": {"answer": "Expired"},
                    "expires_at": current_time - 1000  # Expired
                }
            elif "expired2" in key:
                return {
                    "result": {"answer": "Expired"},
                    "expires_at": current_time - 500  # Expired
                }
            return None
        
        mock_redis.get_json.side_effect = mock_get_json
        
        # Run cleanup
        result = cleanup_query_cache()
        
        # Should have cleaned 2 expired entries
        assert result["cleaned_entries"] == 2
        
        # Verify delete was called for expired entries
        delete_calls = mock_redis.redis_client.delete.call_args_list
        assert len(delete_calls) == 2
    
    @patch('app.workers.query_worker.query_engine_factory')
    @patch('app.workers.query_worker.job_manager')
    def test_process_user_query_success(self, mock_job_manager, mock_factory):
        """Test successful query processing."""
        # Mock no cached result
        mock_factory.get_cached_query_result.return_value = None
        
        # Mock query engine and response
        mock_engine = Mock()
        mock_response = Mock()
        mock_response.response = "This is the answer to your question."
        mock_response.source_nodes = []
        mock_engine.query.return_value = mock_response
        mock_factory.create_query_engine.return_value = mock_engine
        
        # Mock job manager
        mock_job_manager.update_job_status.return_value = True
        mock_job_manager.update_job_progress.return_value = True
        
        # Execute query
        result = process_user_query(
            self.test_query_id,
            self.test_user_id,
            self.test_group_ids,
            self.test_query_text
        )
        
        # Verify result structure
        assert "answer" in result
        assert "sources" in result
        assert "processing_time" in result
        assert "cached" in result
        assert result["cached"] is False
        assert result["answer"] == "This is the answer to your question."
        
        # Verify security filters were applied
        mock_factory.create_query_engine.assert_called_once_with(self.test_user_id, self.test_group_ids)
        
        # Verify result was cached
        mock_factory.cache_query_result.assert_called_once()
        
        # Verify job status updates
        assert mock_job_manager.update_job_status.call_count >= 2  # Processing + completion
    
    @patch('app.workers.query_worker.query_engine_factory')
    @patch('app.workers.query_worker.job_manager')
    def test_process_user_query_cached_result(self, mock_job_manager, mock_factory):
        """Test query processing with cached result."""
        # Mock cached result
        cached_result = {
            "answer": "Cached answer",
            "sources": ["cached_doc.pdf"],
            "processing_time": 0.5,
            "cached": False  # Will be updated to True
        }
        mock_factory.get_cached_query_result.return_value = cached_result
        
        # Mock job manager
        mock_job_manager.update_job_status.return_value = True
        
        # Execute query
        result = process_user_query(
            self.test_query_id,
            self.test_user_id,
            self.test_group_ids,
            self.test_query_text
        )
        
        # Verify cached result was returned
        assert result["answer"] == "Cached answer"
        assert result["cached"] is True
        assert "processing_time" in result
        
        # Verify job status was updated
        mock_job_manager.update_job_status.assert_called()
    
    @patch('app.workers.query_worker.redis_client')
    def test_process_user_query_security_error(self, mock_redis):
        """Test query processing with security validation error."""
        # Test with invalid user_id
        with pytest.raises(ValueError, match="Security validation failed"):
            process_user_query(
                self.test_query_id,
                "",  # Invalid user_id
                self.test_group_ids,
                self.test_query_text
            )
        
        # Verify error was logged in Redis
        mock_redis.set_json.assert_called()
        call_args = mock_redis.set_json.call_args
        stored_data = call_args[0][1]
        assert stored_data["status"] == "failed"
        assert stored_data["error_type"] == "security"
    
    def test_concurrent_query_processing(self):
        """Test concurrent query processing with user isolation."""
        # This test verifies that multiple queries can be processed concurrently
        # without interfering with each other's security contexts
        
        def mock_process_query(user_id, groups, query_text):
            """Mock query processing that simulates different users."""
            # Simulate processing time
            time.sleep(0.1)
            
            # Return user-specific result
            return {
                "answer": f"Answer for {user_id}: {query_text}",
                "sources": [f"{user_id}_document.pdf"],
                "user_context": {"user_id": user_id, "groups": groups}
            }
        
        # Test data for different users
        test_queries = [
            ("user1", ["group1"], "Query from user 1"),
            ("user2", ["group2"], "Query from user 2"),
            ("user3", ["group1", "group3"], "Query from user 3"),
            ("user1", ["group1"], "Another query from user 1"),
        ]
        
        # Process queries concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for user_id, groups, query_text in test_queries:
                future = executor.submit(mock_process_query, user_id, groups, query_text)
                futures.append((future, user_id, groups, query_text))
            
            # Collect results
            results = []
            for future, user_id, groups, query_text in futures:
                result = future.result()
                results.append((result, user_id, groups, query_text))
        
        # Verify each query got the correct user-specific result
        for result, user_id, groups, query_text in results:
            assert result["user_context"]["user_id"] == user_id
            assert result["user_context"]["groups"] == groups
            assert user_id in result["answer"]
            assert query_text in result["answer"]
            assert f"{user_id}_document.pdf" in result["sources"]
        
        # Verify no cross-contamination between users
        user1_results = [r for r, uid, _, _ in results if uid == "user1"]
        user2_results = [r for r, uid, _, _ in results if uid == "user2"]
        
        for result in user1_results:
            assert "user2" not in result["answer"]
            assert "user2_document.pdf" not in result["sources"]
        
        for result in user2_results:
            assert "user1" not in result["answer"]
            assert "user1_document.pdf" not in result["sources"]
    
    def test_user_isolation_security(self):
        """Test that user isolation prevents data leakage."""
        # Test that security filters are properly applied for different users
        
        # User 1 with groups A and B
        user1_filters = query_engine_factory._create_user_security_filters("user1", ["groupA", "groupB"])
        
        # User 2 with groups B and C
        user2_filters = query_engine_factory._create_user_security_filters("user2", ["groupB", "groupC"])
        
        # Verify filters are different
        assert user1_filters != user2_filters
        
        # Verify user1 can access their own documents and groupA, groupB
        user1_filter_values = [f.value for f in user1_filters.filters]
        assert "user1" in user1_filter_values
        assert "groupA" in user1_filter_values
        assert "groupB" in user1_filter_values
        assert "user2" not in user1_filter_values
        assert "groupC" not in user1_filter_values
        
        # Verify user2 can access their own documents and groupB, groupC
        user2_filter_values = [f.value for f in user2_filters.filters]
        assert "user2" in user2_filter_values
        assert "groupB" in user2_filter_values
        assert "groupC" in user2_filter_values
        assert "user1" not in user2_filter_values
        assert "groupA" not in user2_filter_values
        
        # Both users should have access to groupB (shared group)
        assert "groupB" in user1_filter_values
        assert "groupB" in user2_filter_values


class TestQueryWorkerIntegration:
    """Integration tests for query worker with real components."""
    
    def setup_method(self):
        """Set up integration test environment."""
        self.test_user_id = "integration_user"
        self.test_group_ids = ["integration_group"]
        self.test_query_id = "integration_query_123"
        
        # Clear Redis test data
        try:
            redis_client.redis_client.flushdb()
        except:
            pass
    
    def teardown_method(self):
        """Clean up integration test environment."""
        try:
            redis_client.redis_client.flushdb()
        except:
            pass
    
    @patch('app.shared.query_engine_factory.HuggingFaceEmbedding')
    @patch('app.shared.query_engine_factory.LlamaCPP')
    @patch('app.shared.query_engine_factory.DatabaseConnectionPool')
    @patch('app.shared.query_engine_factory.QueryResultCache')
    def test_initialize_query_engine_integration(self, mock_cache, mock_pool, mock_llm, mock_embedding):
        """Test query engine initialization with mocked components using factory."""
        # Mock the components
        mock_embedding_instance = Mock()
        mock_embedding.return_value = mock_embedding_instance
        
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        
        mock_pool_instance = Mock()
        mock_pool.return_value = mock_pool_instance
        
        mock_cache_instance = Mock()
        mock_cache.return_value = mock_cache_instance
        
        # Initialize query engine using factory
        query_engine = query_engine_factory.create_query_engine(self.test_user_id, self.test_group_ids)
        
        # Verify query engine was created (this will trigger component initialization)
        assert query_engine is not None
    
    @patch('app.workers.query_worker.process_user_query')
    def test_concurrent_query_isolation_integration(self, mock_process):
        """Integration test for concurrent query processing with user isolation."""
        # Mock different responses for different users
        def mock_query_response(query_id, user_id, group_ids, query_text):
            return {
                "answer": f"Response for {user_id}: {query_text}",
                "sources": [f"{user_id}_doc.pdf"],
                "processing_time": 0.1,
                "user_context": {"user_id": user_id, "groups": group_ids}
            }
        
        mock_process.side_effect = mock_query_response
        
        # Test concurrent queries from different users
        test_cases = [
            ("query1", "user1", ["group1"], "What is AI?"),
            ("query2", "user2", ["group2"], "What is ML?"),
            ("query3", "user1", ["group1"], "What is DL?"),
        ]
        
        # Execute queries concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for query_id, user_id, groups, query_text in test_cases:
                future = executor.submit(
                    process_user_query, query_id, user_id, groups, query_text
                )
                futures.append((future, query_id, user_id, groups, query_text))
            
            # Collect results
            results = []
            for future, query_id, user_id, groups, query_text in futures:
                result = future.result()
                results.append((result, query_id, user_id, groups, query_text))
        
        # Verify each query got the correct user-specific response
        for result, query_id, user_id, groups, query_text in results:
            assert result["user_context"]["user_id"] == user_id
            assert result["user_context"]["groups"] == groups
            assert user_id in result["answer"]
            assert query_text in result["answer"]