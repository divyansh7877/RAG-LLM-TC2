"""
Tests for the thread-safe query engine factory.
"""
import pytest
import threading
import time
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from app.shared.query_engine_factory import (
    ThreadSafeQueryEngineFactory,
    DatabaseConnectionPool,
    QueryResultCache,
    ConnectionInfo,
    QueryCacheEntry
)


class TestDatabaseConnectionPool:
    """Test database connection pool functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_lancedb(self):
        """Mock LanceDB for testing."""
        with patch('app.shared.query_engine_factory.lancedb') as mock_lancedb:
            mock_db = Mock()
            mock_table = Mock()
            mock_db.open_table.return_value = mock_table
            mock_lancedb.connect.return_value = mock_db
            
            with patch('app.shared.query_engine_factory.LanceDBVectorStore') as mock_vector_store:
                mock_vector_store.return_value = Mock()
                yield mock_lancedb, mock_vector_store
    
    def test_connection_pool_initialization(self, temp_db_path, mock_lancedb):
        """Test connection pool initializes correctly."""
        mock_lancedb_lib, mock_vector_store = mock_lancedb
        
        pool = DatabaseConnectionPool(
            db_path=temp_db_path,
            table_name="test_table",
            max_connections=3
        )
        
        assert pool.max_connections == 3
        assert pool._connections.qsize() == 3
        assert len(pool._connection_info) == 3
        
        # Verify connections were created
        assert mock_lancedb_lib.connect.call_count == 3
    
    def test_connection_pool_get_connection(self, temp_db_path, mock_lancedb):
        """Test getting connections from pool."""
        mock_lancedb_lib, mock_vector_store = mock_lancedb
        
        pool = DatabaseConnectionPool(
            db_path=temp_db_path,
            table_name="test_table",
            max_connections=2
        )
        
        # Get connection
        with pool.get_connection() as conn_data:
            assert 'db' in conn_data
            assert 'table' in conn_data
            assert 'vector_store' in conn_data
            assert 'connection_id' in conn_data
            
            # Check that connection is marked as in use
            conn_id = conn_data['connection_id']
            assert pool._connection_info[conn_id].in_use is True
        
        # After context manager, connection should be returned to pool
        assert pool._connection_info[conn_id].in_use is False
        assert pool._connections.qsize() == 2
    
    def test_connection_pool_concurrent_access(self, temp_db_path, mock_lancedb):
        """Test concurrent access to connection pool."""
        mock_lancedb_lib, mock_vector_store = mock_lancedb
        
        pool = DatabaseConnectionPool(
            db_path=temp_db_path,
            table_name="test_table",
            max_connections=3
        )
        
        results = []
        errors = []
        
        def get_connection_worker(worker_id: int):
            """Worker function to test concurrent connection access."""
            try:
                with pool.get_connection() as conn_data:
                    conn_id = conn_data['connection_id']
                    results.append(f"Worker {worker_id} got connection {conn_id}")
                    time.sleep(0.1)  # Simulate work
                    return conn_id
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
                return None
        
        # Run multiple workers concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_connection_worker, i) for i in range(5)]
            connection_ids = [future.result() for future in as_completed(futures)]
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        assert len([cid for cid in connection_ids if cid is not None]) == 5
        
        # All connections should be returned to pool
        assert pool._connections.qsize() == 3
        for info in pool._connection_info.values():
            assert info.in_use is False
    
    def test_connection_pool_stats(self, temp_db_path, mock_lancedb):
        """Test connection pool statistics."""
        mock_lancedb_lib, mock_vector_store = mock_lancedb
        
        pool = DatabaseConnectionPool(
            db_path=temp_db_path,
            table_name="test_table",
            max_connections=2
        )
        
        stats = pool.get_pool_stats()
        
        assert stats['max_connections'] == 2
        assert stats['available_connections'] == 2
        assert stats['total_connections'] == 2
        assert stats['connections_in_use'] == 0
        assert len(stats['connection_details']) == 2
        
        # Test with connection in use
        with pool.get_connection():
            stats = pool.get_pool_stats()
            assert stats['available_connections'] == 1
            assert stats['connections_in_use'] == 1


class TestQueryResultCache:
    """Test query result cache functionality."""
    
    @pytest.fixture
    def cache(self):
        """Create query result cache for testing."""
        return QueryResultCache(max_size=5)
    
    def test_cache_basic_operations(self, cache):
        """Test basic cache operations."""
        user_id = "test_user"
        group_ids = ["group1", "group2"]
        query_text = "test query"
        result = {"answer": "test answer", "sources": ["doc1"]}
        
        # Test cache miss
        cached_result = cache.get(user_id, group_ids, query_text)
        assert cached_result is None
        
        # Test cache set and hit
        cache.set(user_id, group_ids, query_text, result)
        cached_result = cache.get(user_id, group_ids, query_text)
        assert cached_result == result
    
    def test_cache_user_isolation(self, cache):
        """Test that cache properly isolates users."""
        user1_id = "user1"
        user2_id = "user2"
        group_ids = ["group1"]
        query_text = "same query"
        result1 = {"answer": "answer for user1"}
        result2 = {"answer": "answer for user2"}
        
        # Cache results for different users
        cache.set(user1_id, group_ids, query_text, result1)
        cache.set(user2_id, group_ids, query_text, result2)
        
        # Verify isolation
        cached_result1 = cache.get(user1_id, group_ids, query_text)
        cached_result2 = cache.get(user2_id, group_ids, query_text)
        
        assert cached_result1 == result1
        assert cached_result2 == result2
        assert cached_result1 != cached_result2
    
    def test_cache_group_sensitivity(self, cache):
        """Test that cache is sensitive to group changes."""
        user_id = "test_user"
        groups1 = ["group1"]
        groups2 = ["group2"]
        query_text = "test query"
        result1 = {"answer": "answer for group1"}
        result2 = {"answer": "answer for group2"}
        
        # Cache results for different groups
        cache.set(user_id, groups1, query_text, result1)
        cache.set(user_id, groups2, query_text, result2)
        
        # Verify group sensitivity
        cached_result1 = cache.get(user_id, groups1, query_text)
        cached_result2 = cache.get(user_id, groups2, query_text)
        
        assert cached_result1 == result1
        assert cached_result2 == result2
    
    def test_cache_expiration(self, cache):
        """Test cache entry expiration."""
        user_id = "test_user"
        group_ids = ["group1"]
        query_text = "test query"
        result = {"answer": "test answer"}
        
        # Mock time to test expiration
        with patch('time.time') as mock_time:
            # Set initial time
            mock_time.return_value = 1000.0
            cache.set(user_id, group_ids, query_text, result)
            
            # Verify cache hit
            cached_result = cache.get(user_id, group_ids, query_text)
            assert cached_result == result
            
            # Advance time past expiration
            mock_time.return_value = 1000.0 + 3601  # 1 hour + 1 second
            
            # Verify cache miss due to expiration
            cached_result = cache.get(user_id, group_ids, query_text)
            assert cached_result is None
    
    def test_cache_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        user_id = "test_user"
        group_ids = ["group1"]
        
        # Fill cache to capacity
        for i in range(5):
            cache.set(user_id, group_ids, f"query_{i}", {"answer": f"answer_{i}"})
        
        # Verify all entries are cached
        for i in range(5):
            result = cache.get(user_id, group_ids, f"query_{i}")
            assert result == {"answer": f"answer_{i}"}
        
        # Add one more entry to trigger eviction
        cache.set(user_id, group_ids, "query_new", {"answer": "new_answer"})
        
        # First entry should be evicted (LRU)
        result = cache.get(user_id, group_ids, "query_0")
        assert result is None
        
        # New entry should be cached
        result = cache.get(user_id, group_ids, "query_new")
        assert result == {"answer": "new_answer"}
    
    def test_cache_invalidate_user(self, cache):
        """Test invalidating all cache entries for a user."""
        user1_id = "user1"
        user2_id = "user2"
        group_ids = ["group1"]
        
        # Cache entries for both users
        cache.set(user1_id, group_ids, "query1", {"answer": "answer1"})
        cache.set(user1_id, group_ids, "query2", {"answer": "answer2"})
        cache.set(user2_id, group_ids, "query1", {"answer": "answer3"})
        
        # Verify entries are cached
        assert cache.get(user1_id, group_ids, "query1") is not None
        assert cache.get(user1_id, group_ids, "query2") is not None
        assert cache.get(user2_id, group_ids, "query1") is not None
        
        # Invalidate user1's cache
        cache.invalidate_user_cache(user1_id)
        
        # User1's entries should be gone
        assert cache.get(user1_id, group_ids, "query1") is None
        assert cache.get(user1_id, group_ids, "query2") is None
        
        # User2's entries should remain
        assert cache.get(user2_id, group_ids, "query1") is not None
    
    def test_cache_concurrent_access(self, cache):
        """Test concurrent access to cache."""
        user_id = "test_user"
        group_ids = ["group1"]
        results = []
        errors = []
        
        def cache_worker(worker_id: int):
            """Worker function to test concurrent cache access."""
            try:
                query_text = f"query_{worker_id}"
                result = {"answer": f"answer_{worker_id}"}
                
                # Set cache entry
                cache.set(user_id, group_ids, query_text, result)
                
                # Get cache entry
                cached_result = cache.get(user_id, group_ids, query_text)
                
                if cached_result == result:
                    results.append(f"Worker {worker_id} success")
                else:
                    errors.append(f"Worker {worker_id} cache mismatch")
                    
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # Run multiple workers concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(cache_worker, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()  # Wait for completion
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10


class TestThreadSafeQueryEngineFactory:
    """Test thread-safe query engine factory."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies."""
        with patch('app.shared.query_engine_factory.HuggingFaceEmbedding') as mock_embed, \
             patch('app.shared.query_engine_factory.LlamaCPP') as mock_llm, \
             patch('app.shared.query_engine_factory.load_index_from_storage') as mock_load_index, \
             patch('app.shared.query_engine_factory.DatabaseConnectionPool') as mock_pool, \
             patch('app.shared.query_engine_factory.QueryResultCache') as mock_cache:
            
            # Configure mocks
            mock_embed.return_value = Mock()
            mock_llm.return_value = Mock()
            mock_load_index.return_value = Mock()
            mock_pool.return_value = Mock()
            mock_cache.return_value = Mock()
            
            yield {
                'embed': mock_embed,
                'llm': mock_llm,
                'load_index': mock_load_index,
                'pool': mock_pool,
                'cache': mock_cache
            }
    
    def test_factory_initialization(self, mock_dependencies):
        """Test factory initialization."""
        factory = ThreadSafeQueryEngineFactory()
        
        assert factory._initialized is False
        assert factory._embed_model is None
        assert factory._llm is None
        assert factory._index is None
        assert factory._connection_pool is None
        assert factory._query_cache is None
    
    def test_factory_component_initialization(self, mock_dependencies):
        """Test factory component initialization."""
        factory = ThreadSafeQueryEngineFactory()
        
        # Mock Settings to avoid validation issues
        with patch('app.shared.query_engine_factory.Settings') as mock_settings:
            # Initialize components
            factory._initialize_components()
            
            assert factory._initialized is True
            assert factory._embed_model is not None
            assert factory._llm is not None
            assert factory._connection_pool is not None
            assert factory._query_cache is not None
            
            # Verify mocks were called
            mock_dependencies['embed'].assert_called_once()
            mock_dependencies['llm'].assert_called_once()
            mock_dependencies['pool'].assert_called_once()
            mock_dependencies['cache'].assert_called_once()
            
            # Verify Settings were set
            assert mock_settings.embed_model is not None
            assert mock_settings.llm is not None
    
    def test_factory_create_query_engine(self, mock_dependencies):
        """Test creating query engines."""
        factory = ThreadSafeQueryEngineFactory()
        
        # Mock the index loading
        mock_index = Mock()
        factory._index = mock_index
        factory._initialized = True
        factory._embed_model = Mock()
        factory._llm = Mock()
        
        with patch('app.shared.query_engine_factory.VectorIndexRetriever') as mock_retriever, \
             patch('app.shared.query_engine_factory.SentenceTransformerRerank') as mock_reranker, \
             patch('app.shared.query_engine_factory.get_response_synthesizer') as mock_synthesizer, \
             patch('app.shared.query_engine_factory.RetrieverQueryEngine') as mock_query_engine:
            
            mock_retriever.return_value = Mock()
            mock_reranker.return_value = Mock()
            mock_synthesizer.return_value = Mock()
            mock_query_engine.return_value = Mock()
            
            # Create query engine
            user_id = "test_user"
            group_ids = ["group1", "group2"]
            
            query_engine = factory.create_query_engine(user_id, group_ids)
            
            # Verify query engine was created
            assert query_engine is not None
            mock_query_engine.assert_called_once()
            
            # Verify security filters were applied
            mock_retriever.assert_called_once()
            call_args = mock_retriever.call_args
            assert 'filters' in call_args.kwargs
    
    def test_factory_concurrent_query_engine_creation(self, mock_dependencies):
        """Test concurrent query engine creation."""
        factory = ThreadSafeQueryEngineFactory()
        
        # Pre-initialize to avoid initialization race conditions in test
        factory._initialized = True
        factory._embed_model = Mock()
        factory._llm = Mock()
        factory._index = Mock()
        factory._connection_pool = Mock()
        factory._query_cache = Mock()
        
        results = []
        errors = []
        
        def create_engine_worker(worker_id: int):
            """Worker function to test concurrent engine creation."""
            try:
                with patch('app.shared.query_engine_factory.VectorIndexRetriever'), \
                     patch('app.shared.query_engine_factory.SentenceTransformerRerank'), \
                     patch('app.shared.query_engine_factory.get_response_synthesizer'), \
                     patch('app.shared.query_engine_factory.RetrieverQueryEngine') as mock_engine:
                    
                    mock_engine.return_value = Mock()
                    
                    user_id = f"user_{worker_id}"
                    group_ids = [f"group_{worker_id}"]
                    
                    query_engine = factory.create_query_engine(user_id, group_ids)
                    
                    if query_engine is not None:
                        results.append(f"Worker {worker_id} success")
                    else:
                        errors.append(f"Worker {worker_id} got None engine")
                        
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # Run multiple workers concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_engine_worker, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()  # Wait for completion
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
    
    def test_factory_cache_operations(self, mock_dependencies):
        """Test factory cache operations."""
        factory = ThreadSafeQueryEngineFactory()
        
        # Mock cache
        mock_cache = Mock()
        factory._query_cache = mock_cache
        
        user_id = "test_user"
        group_ids = ["group1"]
        query_text = "test query"
        result = {"answer": "test answer"}
        
        # Test get cached result
        mock_cache.get.return_value = result
        cached_result = factory.get_cached_query_result(user_id, group_ids, query_text)
        assert cached_result == result
        mock_cache.get.assert_called_once_with(user_id, group_ids, query_text)
        
        # Test cache result
        factory.cache_query_result(user_id, group_ids, query_text, result)
        mock_cache.set.assert_called_once_with(user_id, group_ids, query_text, result)
        
        # Test invalidate user cache
        factory.invalidate_user_cache(user_id)
        mock_cache.invalidate_user_cache.assert_called_once_with(user_id)
    
    def test_factory_stats_and_health_check(self, mock_dependencies):
        """Test factory statistics and health check."""
        factory = ThreadSafeQueryEngineFactory()
        
        # Mock components
        mock_pool = Mock()
        mock_cache = Mock()
        mock_pool.get_pool_stats.return_value = {"connections": 5}
        mock_cache.get_cache_stats.return_value = {"entries": 10}
        
        factory._connection_pool = mock_pool
        factory._query_cache = mock_cache
        factory._initialized = True
        factory._embed_model = Mock()
        factory._llm = Mock()
        factory._index = Mock()
        
        # Test stats
        stats = factory.get_factory_stats()
        assert stats["initialized"] is True
        assert stats["connection_pool"] == {"connections": 5}
        assert stats["query_cache"] == {"entries": 10}
        
        # Test health check
        health = factory.health_check()
        assert health["factory_initialized"] is True
        assert health["embed_model_loaded"] is True
        assert health["llm_loaded"] is True
        assert health["index_loaded"] is True
    
    def test_factory_cleanup(self, mock_dependencies):
        """Test factory cleanup operations."""
        factory = ThreadSafeQueryEngineFactory()
        
        # Mock components
        mock_pool = Mock()
        mock_cache = Mock()
        factory._connection_pool = mock_pool
        factory._query_cache = mock_cache
        
        # Test cleanup
        factory.cleanup()
        
        mock_cache.cleanup.assert_called_once()
        mock_pool.cleanup_stale_connections.assert_called_once()


class TestSecurityFilters:
    """Test security filter creation and validation."""
    
    def test_security_filter_creation(self):
        """Test creation of user security filters."""
        factory = ThreadSafeQueryEngineFactory()
        
        user_id = "test_user"
        group_ids = ["group1", "group2"]
        
        filters = factory._create_user_security_filters(user_id, group_ids)
        
        assert filters is not None
        assert len(filters.filters) == 3  # 1 user filter + 2 group filters
        assert filters.condition == "or"
    
    def test_security_filter_validation(self):
        """Test validation of security filter inputs."""
        factory = ThreadSafeQueryEngineFactory()
        
        # Test empty user ID
        with pytest.raises(ValueError, match="User ID is required"):
            factory._create_user_security_filters("", ["group1"])
        
        # Test empty group IDs
        with pytest.raises(ValueError, match="At least one group ID is required"):
            factory._create_user_security_filters("user1", [])
        
        # Test None inputs
        with pytest.raises(ValueError):
            factory._create_user_security_filters(None, ["group1"])


@pytest.mark.integration
class TestQueryEngineIntegration:
    """Integration tests for query engine factory."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_end_to_end_query_processing(self, temp_db_path):
        """Test end-to-end query processing with mocked components."""
        # This would be a more comprehensive integration test
        # For now, we'll skip it as it requires actual LanceDB setup
        pytest.skip("Integration test requires full database setup")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])