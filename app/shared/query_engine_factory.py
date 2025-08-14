#!/usr/bin/env python3
"""
Thread-safe query engine factory for the concurrent RAG system.

This module provides a thread-safe query engine factory that can be used
by Celery workers to process user queries with proper isolation and security.
"""
import os
import multiprocessing as mp
import threading
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
import torch

from .config import config
from .error_handling import StructuredLogger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_PATH = "./multi_user_db.lance"
TABLE_NAME = "document_embeddings"
EMBED_MODEL_NAME = "./models/gte-large-en-v1.5"  # INT8‐quantised, CPU
GGUF_MODEL_PATH = "./models/Llama-3.2-3B-Instruct-IQ3_M.gguf"  # llama.cpp model

has_cuda = torch.cuda.is_available()

# Runtime knobs
device = "cuda" if has_cuda else "cpu"  
BEAM_K = 30      # initial ANN beam for MMR
FINAL_K = 8     # chunks passed to the LLM
MMR_LAMBDA = 0.1
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
N_THREADS = mp.cpu_count()
N_GPU_LAYERS = -1 if has_cuda else 0 # set 0 if no GPU / VRAM < 12 GB
N_BATCH = 1024 if has_cuda else 64   # llama.cpp prompt batch size

# ---------------------------------------------------------------------------
# Prompt enforcing source citations
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Thread-safe Query Engine Factory
# ---------------------------------------------------------------------------

class QueryEngineFactory:
    """Thread-safe factory for creating query engines with proper isolation."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._embed_model = None
        self._llm = None
        self._vector_store = None
        self._index = None
        # Simple in-process cache for query results
        self._query_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._cache_ttl_seconds: int = 3600
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self.logger = StructuredLogger(__name__)
    
    def _get_embed_model(self):
        """Get or create the embedding model (singleton per process)."""
        if self._embed_model is None:
            with self._lock:
                if self._embed_model is None:
                    self._embed_model = HuggingFaceEmbedding(
                        model_name=EMBED_MODEL_NAME,
                        device=device,
                        trust_remote_code=True,
                        model_kwargs={"quantize": "static-int8"},
                    )
        return self._embed_model
    
    def _get_llm(self):
        """Get or create the LLM (singleton per process)."""
        if self._llm is None:
            with self._lock:
                if self._llm is None:
                    self._llm = LlamaCPP(
                        model_path=GGUF_MODEL_PATH,
                        temperature=0.3,
                        max_new_tokens=512,
                        context_window=2048,
                        model_kwargs={
                            "n_batch": N_BATCH,
                            "n_gpu_layers": N_GPU_LAYERS,
                        },
                        verbose=True,
                    )
        return self._llm
    
    def _get_vector_store(self):
        """Get or create the vector store (singleton per process)."""
        if self._vector_store is None:
            with self._lock:
                if self._vector_store is None:
                    self._vector_store = LanceDBVectorStore(
                        uri=DB_PATH, 
                        table_name=TABLE_NAME, 
                        mode="r"
                    )
        return self._vector_store
    
    def _get_index(self):
        """Get or create the index (singleton per process)."""
        if self._index is None:
            with self._lock:
                if self._index is None:
                    vector_store = self._get_vector_store()
                    storage_context = StorageContext.from_defaults(
                        persist_dir=os.path.join(DB_PATH, "li_storage"),
                        vector_store=vector_store,
                    )
                    self._index = load_index_from_storage(
                        storage_context, 
                        embed_model=self._get_embed_model()
                    )
        return self._index
    
    def create_query_engine(
        self,
        user_filters: Optional[MetadataFilters] = None,
        user_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
    ):
        """Create a new query engine instance with optional user filters.

        Accepts either a pre-built `user_filters` or a `user_id` with `group_ids`.
        """
        # Get the shared components
        embed_model = self._get_embed_model()
        llm = self._get_llm()
        index = self._get_index()
        
        # Create a new retriever instance (not shared)
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=BEAM_K,
            search_type="similarity",
        )
        
        # Build filters if user_id/group_ids provided
        if user_filters is None and user_id is not None:
            try:
                user_filter = ExactMatchFilter(key="user_id", value=user_id)
                group_filters = [ExactMatchFilter(key="group_id", value=gid) for gid in (group_ids or [])]
                all_filters = [user_filter] + group_filters
                user_filters = MetadataFilters(filters=all_filters, condition="or")
            except Exception as e:
                self.logger.error(f"Failed to build user filters: {e}")

        # Apply filters if available
        if user_filters is not None:
            retriever.vector_store_kwargs = {"filters": user_filters}
        
        # Create reranker
        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
            top_n=FINAL_K,
            device=device, 
        )
        
        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            text_qa_template=PromptTemplate(QA_TEMPLATE),
        )
        
        return RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[reranker],
            response_synthesizer=response_synthesizer,
        )

    # -----------------------------------------------------------------------
    # Simple query result cache helpers
    # -----------------------------------------------------------------------
    def _make_cache_key(self, user_id: str, group_ids: List[str], query_text: str) -> str:
        import hashlib
        normalized_groups = ",".join(sorted(group_ids or []))
        payload = f"u:{user_id}|g:{normalized_groups}|q:{query_text.strip()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_cached_query_result(self, user_id: str, group_ids: List[str], query_text: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        cache_key = self._make_cache_key(user_id, group_ids, query_text)
        with self._lock:
            entry = self._query_cache.get(cache_key)
            if not entry:
                self._cache_misses += 1
                return None
            ts, value = entry
            if now - ts > self._cache_ttl_seconds:
                # expired
                self._query_cache.pop(cache_key, None)
                self._cache_misses += 1
                return None
            self._cache_hits += 1
            return value

    def cache_query_result(self, user_id: str, group_ids: List[str], query_text: str, result: Dict[str, Any]) -> None:
        cache_key = self._make_cache_key(user_id, group_ids, query_text)
        with self._lock:
            self._query_cache[cache_key] = (time.time(), result)
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the query engine factory."""
        try:
            # Test basic functionality
            test_filters = MetadataFilters(
                filters=[ExactMatchFilter(key="user_id", value="health_check")],
                condition="or"
            )
            
            # Try to create a query engine (this will initialize models if needed)
            query_engine = self.create_query_engine(user_filters=test_filters)
            
            return {
                "status": "healthy",
                "models_loaded": True,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "status": "unhealthy",
                "error": str(e),
                "models_loaded": False,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """Get statistics about the query engine factory."""
        with self._lock:
            cache_size = len(self._query_cache)
            stats = {
                "models_initialized": {
                    "embedding_model": self._embed_model is not None,
                    "llm": self._llm is not None,
                    "vector_store": self._vector_store is not None,
                    "index": self._index is not None,
                },
                "query_cache": {
                    "size": cache_size,
                    "ttl_seconds": self._cache_ttl_seconds,
                    "hits": self._cache_hits,
                    "misses": self._cache_misses,
                },
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            return stats

    def cleanup(self):
        """Clean up resources to free up memory."""
        with self._lock:
            self._embed_model = None
            self._llm = None
            self._vector_store = None
            self._index = None
            # Remove expired cache entries and trim cache
            now = time.time()
            keys_to_delete = [k for k, (ts, _) in self._query_cache.items() if now - ts > self._cache_ttl_seconds]
            for k in keys_to_delete:
                self._query_cache.pop(k, None)
            # Optionally, clear entire cache if it grows too much
            if len(self._query_cache) > 1000:
                self._query_cache.clear()
            self.logger.info("Query engine factory resources have been cleaned up.")

# ---------------------------------------------------------------------------
# Query Processing Functions
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Result of a query operation."""
    response: str
    sources: List[Dict[str, Any]]
    processing_time: float
    query_id: Optional[str] = None
    error: Optional[str] = None

@dataclass
class QuerySource:
    """Information about a query result source."""
    document_name: str
    page_number: int
    score: float
    content_snippet: str

class QueryEngineService:
    """Service class for processing queries with proper error handling and monitoring."""
    
    def __init__(self):
        self.factory = QueryEngineFactory()
        self.logger = StructuredLogger(__name__)
    
    def process_query(self, query_text: str, user_id: str, group_ids: List[str], query_id: Optional[str] = None) -> QueryResult:
        """
        Process a user query with proper isolation and error handling.
        
        Args:
            query_text: The query text to process
            user_id: ID of the user making the query
            group_ids: List of group IDs the user has access to
            query_id: Optional query ID for tracking
            
        Returns:
            QueryResult: The query result with response and sources
        """
        start_time = time.time()
        
        try:
            if not query_text or not query_text.strip():
                return QueryResult(
                    response="Please enter a question.",
                    sources=[],
                    processing_time=0.0,
                    query_id=query_id,
                    error="Empty query text"
                )
            
            self.logger.info(f"Processing query for user {user_id}", extra={
                'query_id': query_id,
                'user_id': user_id,
                'group_ids': group_ids,
                'query_length': len(query_text)
            })
            
            # Create user-specific filters
            user_filter = ExactMatchFilter(key="user_id", value=user_id)
            group_filters = [ExactMatchFilter(key="group_id", value=group_id) for group_id in group_ids]
            
            # Combine filters: user can access their personal docs OR docs from their groups
            all_filters = [user_filter] + group_filters
            filters = MetadataFilters(filters=all_filters, condition="or")
            
            # Create a new query engine instance with user-specific filters
            query_engine = self.factory.create_query_engine(user_filters=filters)
            
            # Execute the query
            response = query_engine.query(query_text)
            
            # Process sources
            sources = []
            for sn in response.source_nodes:
                meta = sn.node.metadata
                sources.append({
                    "document_name": meta.get("document_name", "N/A"),
                    "page_number": meta.get("page_number", "?"),
                    "score": float(sn.score) if sn.score else 0.0,
                    "content_snippet": sn.node.text[:200] + "..." if len(sn.node.text) > 200 else sn.node.text
                })
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"Query processed successfully", extra={
                'query_id': query_id,
                'user_id': user_id,
                'processing_time': processing_time,
                'source_count': len(sources)
            })
            
            return QueryResult(
                response=str(response.response),
                sources=sources,
                processing_time=processing_time,
                query_id=query_id
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Query processing failed: {str(e)}"
            
            self.logger.error(error_msg, extra={
                'query_id': query_id,
                'user_id': user_id,
                'processing_time': processing_time,
                'error': str(e)
            }, exc_info=True)
            
            return QueryResult(
                response="I apologize, but I encountered an error while processing your query. Please try again later.",
                sources=[],
                processing_time=processing_time,
                query_id=query_id,
                error=error_msg
            )
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the query engine service."""
        return self.factory.health_check()

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def create_user_filters(user_id: str, group_ids: List[str]) -> MetadataFilters:
    """
    Create metadata filters for user isolation.
    
    Args:
        user_id: ID of the user
        group_ids: List of group IDs the user has access to
        
    Returns:
        MetadataFilters: Filters that ensure user can only access authorized documents
    """
    user_filter = ExactMatchFilter(key="user_id", value=user_id)
    group_filters = [ExactMatchFilter(key="group_id", value=group_id) for group_id in group_ids]
    
    all_filters = [user_filter] + group_filters
    return MetadataFilters(filters=all_filters, condition="or")

# Global instances
query_engine_factory = QueryEngineFactory()
query_service = QueryEngineService()