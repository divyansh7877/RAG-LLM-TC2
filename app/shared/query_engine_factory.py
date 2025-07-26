"""
Thread-safe query engine factory with connection pooling and caching.
"""
import os
import time
import hashlib
import threading
import logging
from typing import Dict, Any, Optional, List, Tuple
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue, Empty
import weakref

import lancedb
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate,
    VectorStoreIndex
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

from .redis_client import redis_client
from .config import config

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DB_PATH = config.LANCEDB_PATH
TABLE_NAME = "document_embeddings"
EMBED_MODEL_NAME = config.EMBEDDING_MODEL_PATH
GGUF_MODEL_PATH = "./models/Llama-3.2-3B-Instruct-IQ3_M.gguf"
CACHE_EXPIRE_SECONDS = 3600  # 1 hour cache expiration
MAX_CONNECTIONS = 10  # Maximum database connections in pool
CONNECTION_TIMEOUT = 30  # Connection timeout in seconds

# Query template
QA_TEMPLATE = (
    "You are an AI assistant specialised in answering questions from provided expert‑call transcripts.\n"
    "Use the context but do not just provide the context, use the CONTEXT and the QUESTION to generate a meaningful answer. "
    "If the context lacks the answer, reply: \"The provided context does not contain information to answer this question.\"\n"
    "When citing, follow this format: (Source: {document_name}, Page: {page_number}).\n"
    "------------------------\n"
    "CONTEXT:\n{context_str}\n"
    "------------------------\n"
    "QUESTION: {query_str}\n"
    "------------------------\n"
    "ANSWER:\n"
)


@dataclass
class ConnectionInfo:
    """Information about a database connection."""
    connection_id: str
    created_at: float
    last_used: float
    in_use: bool
    thread_id: int


@dataclass
class QueryCacheEntry:
    """Query cache entry with metadata."""
    result: Dict[str, Any]
    created_at: float
    expires_at: float
    user_id: str
    group_ids: List[str]
    query_hash: str


class DatabaseConnectionPool:
    """Thread-safe database connection pool for LanceDB."""
    
    def __init__(self, db_path: str, table_name: str, max_connections: int = MAX_CONNECTIONS):
        """Initialize connection pool."""
        self.db_path = db_path
        self.table_name = table_name
        self.max_connections = max_connections
        self._connections: Queue = Queue(maxsize=max_connections)
        self._connection_info: Dict[str, ConnectionInfo] = {}
        self._lock = threading.RLock()
        self._connection_counter = 0
        
        # Pre-create connections
        self._initialize_connections()
        
        logger.info(f"Database connection pool initialized with {max_connections} connections")
    
    def _initialize_connections(self):
        """Pre-create database connections."""
        for _ in range(self.max_connections):
            try:
                conn_id = self._create_connection()
                if conn_id:
                    logger.debug(f"Pre-created database connection: {conn_id}")
            except Exception as e:
                logger.warning(f"Failed to pre-create database connection: {e}")
    
    def _create_connection(self) -> Optional[str]:
        """Create a new database connection."""
        try:
            with self._lock:
                self._connection_counter += 1
                conn_id = f"conn_{self._connection_counter}_{threading.current_thread().ident}"
            
            # Create LanceDB connection
            db = lancedb.connect(self.db_path)
            table = db.open_table(self.table_name)
            
            # Create vector store
            vector_store = LanceDBVectorStore(
                uri=self.db_path,
                table_name=self.table_name,
                mode="r"  # Read-only for thread safety
            )
            
            connection_data = {
                'db': db,
                'table': table,
                'vector_store': vector_store,
                'connection_id': conn_id
            }
            
            # Store connection info
            with self._lock:
                self._connection_info[conn_id] = ConnectionInfo(
                    connection_id=conn_id,
                    created_at=time.time(),
                    last_used=time.time(),
                    in_use=False,
                    thread_id=threading.current_thread().ident
                )
            
            # Add to pool
            self._connections.put(connection_data, block=False)
            
            logger.debug(f"Created database connection: {conn_id}")
            return conn_id
            
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            return None
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool."""
        connection_data = None
        conn_id = None
        
        try:
            # Get connection from pool with timeout
            connection_data = self._connections.get(timeout=CONNECTION_TIMEOUT)
            conn_id = connection_data['connection_id']
            
            # Update connection info
            with self._lock:
                if conn_id in self._connection_info:
                    self._connection_info[conn_id].last_used = time.time()
                    self._connection_info[conn_id].in_use = True
            
            logger.debug(f"Retrieved database connection: {conn_id}")
            yield connection_data
            
        except Empty:
            logger.error("Database connection pool exhausted - timeout waiting for connection")
            raise RuntimeError("Database connection pool exhausted")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            # Return connection to pool
            if connection_data and conn_id:
                try:
                    with self._lock:
                        if conn_id in self._connection_info:
                            self._connection_info[conn_id].in_use = False
                    
                    self._connections.put(connection_data, block=False)
                    logger.debug(f"Returned database connection to pool: {conn_id}")
                except Exception as e:
                    logger.warning(f"Failed to return connection to pool: {e}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                "max_connections": self.max_connections,
                "available_connections": self._connections.qsize(),
                "total_connections": len(self._connection_info),
                "connections_in_use": sum(1 for info in self._connection_info.values() if info.in_use),
                "connection_details": [
                    {
                        "connection_id": info.connection_id,
                        "created_at": info.created_at,
                        "last_used": info.last_used,
                        "in_use": info.in_use,
                        "thread_id": info.thread_id
                    }
                    for info in self._connection_info.values()
                ]
            }
    
    def cleanup_stale_connections(self, max_idle_time: float = 300):
        """Clean up connections that have been idle too long."""
        current_time = time.time()
        cleaned = 0
        
        with self._lock:
            stale_connections = [
                conn_id for conn_id, info in self._connection_info.items()
                if not info.in_use and (current_time - info.last_used) > max_idle_time
            ]
            
            for conn_id in stale_connections:
                try:
                    del self._connection_info[conn_id]
                    cleaned += 1
                    logger.debug(f"Cleaned stale connection: {conn_id}")
                except Exception as e:
                    logger.warning(f"Failed to clean stale connection {conn_id}: {e}")
        
        if cleaned > 0:
            logger.info(f"Cleaned {cleaned} stale database connections")
        
        return cleaned


class QueryResultCache:
    """Thread-safe query result cache with user isolation."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize query cache."""
        self.max_size = max_size
        self._cache: Dict[str, QueryCacheEntry] = {}
        self._lock = threading.RLock()
        self._access_order: List[str] = []  # For LRU eviction
        
        logger.info(f"Query result cache initialized with max size: {max_size}")
    
    def _generate_cache_key(self, user_id: str, group_ids: List[str], query_text: str) -> str:
        """Generate a cache key for query results."""
        # Create a deterministic cache key that includes user context
        context_str = f"{user_id}:{':'.join(sorted(group_ids))}:{query_text.strip().lower()}"
        cache_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]
        return f"query_cache:{cache_hash}"
    
    def _evict_expired_entries(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at <= current_time
        ]
        
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
        
        if expired_keys:
            logger.debug(f"Evicted {len(expired_keys)} expired cache entries")
    
    def _evict_lru_entries(self):
        """Evict least recently used entries if cache is full."""
        while len(self._cache) >= self.max_size and self._access_order:
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
                logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    def get(self, user_id: str, group_ids: List[str], query_text: str) -> Optional[Dict[str, Any]]:
        """Get cached query result if available and not expired."""
        cache_key = self._generate_cache_key(user_id, group_ids, query_text)
        
        with self._lock:
            # Clean up expired entries first
            self._evict_expired_entries()
            
            entry = self._cache.get(cache_key)
            if entry and entry.expires_at > time.time():
                # Update access order for LRU
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                
                logger.debug(f"Cache hit for key: {cache_key}")
                return entry.result
            elif entry:
                # Expired entry
                del self._cache[cache_key]
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                logger.debug(f"Expired cache entry removed: {cache_key}")
        
        return None
    
    def set(self, user_id: str, group_ids: List[str], query_text: str, result: Dict[str, Any]):
        """Cache query result with expiration and user isolation."""
        cache_key = self._generate_cache_key(user_id, group_ids, query_text)
        current_time = time.time()
        
        with self._lock:
            # Clean up expired entries
            self._evict_expired_entries()
            
            # Evict LRU entries if needed
            self._evict_lru_entries()
            
            # Create cache entry
            entry = QueryCacheEntry(
                result=result,
                created_at=current_time,
                expires_at=current_time + CACHE_EXPIRE_SECONDS,
                user_id=user_id,
                group_ids=group_ids.copy(),
                query_hash=cache_key
            )
            
            # Store in cache
            self._cache[cache_key] = entry
            
            # Update access order
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            
            logger.debug(f"Cached query result with key: {cache_key}")
    
    def invalidate_user_cache(self, user_id: str):
        """Invalidate all cache entries for a specific user."""
        with self._lock:
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if entry.user_id == user_id
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
            
            if keys_to_remove:
                logger.info(f"Invalidated {len(keys_to_remove)} cache entries for user: {user_id}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            current_time = time.time()
            expired_count = sum(
                1 for entry in self._cache.values()
                if entry.expires_at <= current_time
            )
            
            return {
                "total_entries": len(self._cache),
                "expired_entries": expired_count,
                "active_entries": len(self._cache) - expired_count,
                "max_size": self.max_size,
                "hit_rate": getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1)
            }
    
    def cleanup(self):
        """Clean up expired cache entries."""
        with self._lock:
            self._evict_expired_entries()


class ThreadSafeQueryEngineFactory:
    """Thread-safe factory for creating query engines with connection pooling and caching."""
    
    def __init__(self):
        """Initialize the query engine factory."""
        self._lock = threading.RLock()
        self._embed_model = None
        self._llm = None
        self._index = None
        self._connection_pool = None
        self._query_cache = None
        self._initialized = False
        
        # Thread-local storage for query engines
        self._local = threading.local()
        
        # Weak reference tracking for cleanup
        self._active_engines = weakref.WeakSet()
        
        logger.info("ThreadSafeQueryEngineFactory initialized")
    
    def _initialize_components(self):
        """Initialize shared components (thread-safe singleton pattern)."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            try:
                # Initialize connection pool
                self._connection_pool = DatabaseConnectionPool(
                    db_path=DB_PATH,
                    table_name=TABLE_NAME,
                    max_connections=MAX_CONNECTIONS
                )
                
                # Initialize query cache
                self._query_cache = QueryResultCache(max_size=1000)
                
                # Initialize embedding model
                self._embed_model = HuggingFaceEmbedding(
                    model_name=EMBED_MODEL_NAME,
                    device="cpu",  # Use CPU for thread safety
                    trust_remote_code=True,
                    model_kwargs={"quantize": "static-int8"},
                )
                
                # Initialize LLM
                self._llm = LlamaCPP(
                    model_path=GGUF_MODEL_PATH,
                    temperature=0.3,
                    max_new_tokens=512,
                    context_window=2048,
                    model_kwargs={
                        "n_batch": 64,  # Conservative batch size for stability
                        "n_gpu_layers": 0,  # CPU only for thread safety
                        "n_threads": 1,  # Single thread per instance
                    },
                    verbose=False,
                )
                
                # Set global settings
                Settings.embed_model = self._embed_model
                Settings.llm = self._llm
                
                self._initialized = True
                logger.info("Query engine factory components initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize query engine factory: {e}")
                raise RuntimeError(f"Query engine factory initialization failed: {e}")
    
    def _get_or_create_index(self):
        """Get or create the vector index (thread-safe)."""
        if self._index is not None:
            return self._index
        
        with self._lock:
            if self._index is not None:
                return self._index
            
            try:
                # Use connection pool to get database connection
                with self._connection_pool.get_connection() as conn_data:
                    vector_store = conn_data['vector_store']
                    
                    # Create storage context
                    storage_context = StorageContext.from_defaults(
                        persist_dir=os.path.join(DB_PATH, "li_storage"),
                        vector_store=vector_store,
                    )
                    
                    # Load index from storage
                    self._index = load_index_from_storage(
                        storage_context,
                        embed_model=self._embed_model
                    )
                
                logger.info("Vector index loaded successfully")
                return self._index
                
            except Exception as e:
                logger.error(f"Failed to load vector index: {e}")
                raise RuntimeError(f"Vector index loading failed: {e}")
    
    def create_query_engine(self, user_id: str, group_ids: List[str]) -> RetrieverQueryEngine:
        """
        Create a new query engine instance with user-specific security filters.
        
        Args:
            user_id: User identifier for security filtering
            group_ids: List of group IDs user has access to
            
        Returns:
            Configured RetrieverQueryEngine with security filters
        """
        # Ensure components are initialized
        self._initialize_components()
        
        try:
            # Create user-specific security filters
            user_filters = self._create_user_security_filters(user_id, group_ids)
            
            # Get the shared index
            index = self._get_or_create_index()
            
            # Create a new retriever instance (not shared between threads)
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=30,  # Initial beam for MMR
                filters=user_filters
            )
            
            # Create reranker for better results
            reranker = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_n=8,  # Final number of chunks
                device="cpu",
            )
            
            # Create response synthesizer
            response_synthesizer = get_response_synthesizer(
                llm=self._llm,
                text_qa_template=PromptTemplate(QA_TEMPLATE),
            )
            
            # Create query engine
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                node_postprocessors=[reranker],
                response_synthesizer=response_synthesizer,
            )
            
            # Track active engines for cleanup
            self._active_engines.add(query_engine)
            
            logger.debug(f"Created query engine for user {user_id} with groups {group_ids}")
            return query_engine
            
        except Exception as e:
            logger.error(f"Failed to create query engine for user {user_id}: {e}")
            raise RuntimeError(f"Query engine creation failed: {e}")
    
    def _create_user_security_filters(self, user_id: str, group_ids: List[str]) -> MetadataFilters:
        """Create security filters that ensure user can only access authorized documents."""
        if not user_id:
            raise ValueError("User ID is required for security filtering")
        
        if not group_ids:
            raise ValueError("At least one group ID is required for security filtering")
        
        # Create filters for user's own documents and group documents
        filters = []
        
        # User can access their own documents
        user_filter = ExactMatchFilter(key="user_id", value=user_id)
        filters.append(user_filter)
        
        # User can access documents from their groups
        for group_id in group_ids:
            group_filter = ExactMatchFilter(key="group_id", value=group_id)
            filters.append(group_filter)
        
        # Combine filters with OR condition
        return MetadataFilters(filters=filters, condition="or")
    
    def get_cached_query_result(self, user_id: str, group_ids: List[str], query_text: str) -> Optional[Dict[str, Any]]:
        """Get cached query result if available."""
        if not self._query_cache:
            return None
        
        return self._query_cache.get(user_id, group_ids, query_text)
    
    def cache_query_result(self, user_id: str, group_ids: List[str], query_text: str, result: Dict[str, Any]):
        """Cache query result with user isolation."""
        if self._query_cache:
            self._query_cache.set(user_id, group_ids, query_text, result)
    
    def invalidate_user_cache(self, user_id: str):
        """Invalidate all cached results for a specific user."""
        if self._query_cache:
            self._query_cache.invalidate_user_cache(user_id)
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics including connection pool and cache stats."""
        stats = {
            "initialized": self._initialized,
            "active_engines": len(self._active_engines),
            "connection_pool": None,
            "query_cache": None
        }
        
        if self._connection_pool:
            stats["connection_pool"] = self._connection_pool.get_pool_stats()
        
        if self._query_cache:
            stats["query_cache"] = self._query_cache.get_cache_stats()
        
        return stats
    
    def cleanup(self):
        """Clean up resources and expired cache entries."""
        try:
            if self._query_cache:
                self._query_cache.cleanup()
            
            if self._connection_pool:
                self._connection_pool.cleanup_stale_connections()
            
            logger.debug("Query engine factory cleanup completed")
        except Exception as e:
            logger.warning(f"Error during factory cleanup: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on factory components."""
        health = {
            "factory_initialized": self._initialized,
            "embed_model_loaded": self._embed_model is not None,
            "llm_loaded": self._llm is not None,
            "index_loaded": self._index is not None,
            "connection_pool_healthy": False,
            "cache_healthy": False,
            "errors": []
        }
        
        try:
            if self._connection_pool:
                pool_stats = self._connection_pool.get_pool_stats()
                health["connection_pool_healthy"] = pool_stats["available_connections"] > 0
            
            if self._query_cache:
                cache_stats = self._query_cache.get_cache_stats()
                health["cache_healthy"] = True
            
        except Exception as e:
            health["errors"].append(f"Health check error: {e}")
        
        return health


# Global factory instance
query_engine_factory = ThreadSafeQueryEngineFactory()