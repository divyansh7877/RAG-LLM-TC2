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
    Settings,
    PromptTemplate,
    VectorStoreIndex,
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
# Prefer centralized config and env var overrides to avoid mismatches
DB_PATH = getattr(config, "LANCEDB_PATH", "./multi_user_db.lance")
TABLE_NAME = os.getenv("LANCEDB_TABLE_NAME", "document_embeddings_v2")
EMBED_MODEL_NAME = getattr(config, "EMBEDDING_MODEL_PATH", "./models/gte-large-en-v1.5")
GGUF_MODEL_PATH = "./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"  # llama.cpp model

has_cuda = getattr(config, "HAS_CUDA")

# Runtime knobs
device = "cuda" if has_cuda else "cpu"  
EMBEDDING_DEVICE = device
BEAM_K = 30      # initial ANN beam for MMR
FINAL_K = 8     # chunks passed to the LLM
MMR_LAMBDA = 0.1
RERANK_MODEL_REPO = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH", "./models/cross-encoder/ms-marco-MiniLM-L6-v2")
RERANK_ENABLED = os.getenv("ENABLE_RERANKER", "1").lower() in ("1", "true", "yes")
N_THREADS = mp.cpu_count()
# Optimize for limited GPU memory (3.6 GB)
# Use partial GPU layers to fit within memory constraints
N_GPU_LAYERS = -1 if has_cuda else 0  # Use 20 layers on GPU, rest on CPU
N_BATCH = 64 if has_cuda else 16     # Smaller batch size for limited VRAM

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
        # Use re-entrant lock to avoid deadlocks when nested getters call each other
        self._lock = threading.RLock()
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
        self._config_logged = False

    def _log_configuration(self) -> None:
        """Log important runtime configuration once per process."""
        if self._config_logged:
            return
        self._config_logged = True
        try:
            self.logger.info(
                "[QE] Configuration: "
                f"DB_PATH={os.path.abspath(DB_PATH)}, TABLE_NAME={TABLE_NAME}, "
                f"EMBED_MODEL_NAME={EMBED_MODEL_NAME}, GGUF_MODEL_PATH={GGUF_MODEL_PATH}, "
                f"CUDA={has_cuda}, device={device}, N_GPU_LAYERS={N_GPU_LAYERS}, N_BATCH={N_BATCH}, "
                f"RERANK_ENABLED={RERANK_ENABLED}, RERANK_MODEL_PATH={RERANK_MODEL_PATH}"
            )
        except Exception:
            pass
    
    def _get_embed_model(self):
        """Get or create the embedding model (singleton per process)."""
        if self._embed_model is None:
            with self._lock:
                if self._embed_model is None:
                    # Use configured embedding device (default: GPU if available)
                    embed_device = EMBEDDING_DEVICE
                    
                    t0 = time.perf_counter()
                    self.logger.info("[QE] Step 1: Initializing embedding model ...")
                    self._embed_model = HuggingFaceEmbedding(
                        model_name=EMBED_MODEL_NAME,
                        device=embed_device,
                        trust_remote_code=True,
                    )
                    self.logger.info(f"[QE] Step 1: Embedding model ready on device={embed_device} (took {(time.perf_counter()-t0):.2f}s)")
        return self._embed_model
    
    def _get_llm(self):
        """Get or create the LLM (singleton per process)."""
        if self._llm is None:
            with self._lock:
                if self._llm is None:
                    self.logger.info(f"[QE] Step 2: Initializing LLM (CUDA={has_cuda}, n_gpu_layers={N_GPU_LAYERS}) ...")
                    t0 = time.perf_counter()
                    if not has_cuda:
                        self.logger.warning("LLM Factory: CUDA not available. LLM will run on CPU. Check PyTorch/CUDA installation and NVIDIA drivers.")
                    self._llm = LlamaCPP(
                        model_path=GGUF_MODEL_PATH,
                        temperature=0.3,
                        max_new_tokens=512,
                        context_window=1024,
                        model_kwargs={
                            "n_batch": N_BATCH,
                            "n_gpu_layers": N_GPU_LAYERS,
                        },
                        verbose=True,
                    )
                    self.logger.info(f"[QE] Step 2: LLM ready (took {(time.perf_counter()-t0):.2f}s)")
        return self._llm
    
    def _get_vector_store(self):
        """Get or create the vector store (singleton per process)."""
        if self._vector_store is None:
            with self._lock:
                if self._vector_store is None:
                    preferred_table = TABLE_NAME
                    try:
                        # Prefer an existing shared LanceDB connection if available to avoid file locks
                        self.logger.info(f"[QE] Step 3: Opening LanceDB vector store via shared connection (table={preferred_table}) ...")
                        t0 = time.perf_counter()
                        try:
                            from .lancedb_client import get_db_connection
                            db = get_db_connection()
                            self._vector_store = LanceDBVectorStore(db=db, table_name=preferred_table)  # type: ignore[arg-type]
                            self.logger.info(f"[QE] Step 3: Vector store opened via shared connection (took {(time.perf_counter()-t0):.2f}s)")
                        except TypeError:
                            # Older versions may not support db= parameter
                            self.logger.info(f"[QE] Step 3: Fallback to URI open at {os.path.abspath(DB_PATH)} ...")
                            self._vector_store = LanceDBVectorStore(
                                uri=DB_PATH,
                                table_name=preferred_table,
                                mode="r",
                            )
                            self.logger.info(f"[QE] Step 3: Vector store opened via URI (took {(time.perf_counter()-t0):.2f}s)")
                    except Exception as primary_err:
                        # Backward-compat fallback for older table name
                        fallback_table = "document_embeddings"
                        if preferred_table != fallback_table:
                            self.logger.warning(
                                f"Failed to open table '{preferred_table}': {primary_err}. Trying fallback table '{fallback_table}'."
                            )
                            t1 = time.perf_counter()
                            try:
                                from .lancedb_client import get_db_connection
                                db = get_db_connection()
                                self._vector_store = LanceDBVectorStore(db=db, table_name=fallback_table)  # type: ignore[arg-type]
                                self.logger.info(f"[QE] Step 3: Fallback vector store opened via shared connection (took {(time.perf_counter()-t1):.2f}s)")
                            except TypeError:
                                self._vector_store = LanceDBVectorStore(
                                    uri=DB_PATH, table_name=fallback_table, mode="r"
                                )
                                self.logger.info(f"[QE] Step 3: Fallback vector store opened via URI (took {(time.perf_counter()-t1):.2f}s)")
                        else:
                            raise
        return self._vector_store
    
    def _get_index(self):
        """Get or create the index (singleton per process).

        We build the index directly from the existing LanceDB vector store to
        avoid dependency on a separate persisted `li_storage` directory. This
        matches how embeddings are written by the DocumentProcessor.
        """
        if self._index is None:
            with self._lock:
                if self._index is None:
                    self.logger.info("[QE] Step 4: Building VectorStoreIndex from LanceDB vector store ...")
                    t0 = time.perf_counter()
                    vector_store = self._get_vector_store()
                    self.logger.info("[QE] Step 4.1: Creating StorageContext ...")
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    self.logger.info("[QE] Step 4.2: Constructing VectorStoreIndex.from_vector_store ...")
                    try:
                        self._index = VectorStoreIndex.from_vector_store(
                            vector_store=vector_store,
                            embed_model=self._get_embed_model(),
                            storage_context=storage_context,
                        )
                    except Exception as e:
                        self.logger.error(f"[QE] Step 4 ERROR: Failed to construct VectorStoreIndex: {e}", exc_info=True)
                        raise
                    self.logger.info(f"[QE] Step 4: Index ready (took {(time.perf_counter()-t0):.2f}s)")
        return self._index
    
    def _create_user_security_filters(self, user_id: str, group_ids: List[str]):
        """
        Create security filters for user isolation.
        
        Args:
            user_id: User identifier
            group_ids: List of group IDs the user has access to
            
        Returns:
            MetadataFilters: Filters that ensure user can only access authorized documents
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not user_id or not user_id.strip():
            raise ValueError("User ID is required")
        
        if not group_ids or len(group_ids) == 0:
            raise ValueError("At least one group ID is required")
        
        try:
            # Create user filter
            user_filter = ExactMatchFilter(key="user_id", value=user_id)
            
            # Create group filters
            group_filters = [ExactMatchFilter(key="group_id", value=group_id) for group_id in group_ids]
            
            # Combine filters: user can access their personal docs OR docs from their groups
            all_filters = [user_filter] + group_filters
            
            return MetadataFilters(filters=all_filters, condition="or")
            
        except Exception as e:
            self.logger.error(f"Failed to create user security filters: {e}")
            raise ValueError(f"Failed to create security filters: {e}")

    def create_query_engine(
        self,
        user_filters: Optional[MetadataFilters] = None,
        user_id: Optional[str] = None,
        group_ids: Optional[List[str]] = None,
    ):
        """Create a new query engine instance with optional user filters.

        Accepts either a pre-built `user_filters` or a `user_id` with `group_ids`.
        """
        self._log_configuration()
        self.logger.info("[QE] Creating query engine (building components)...")
        # Get the shared components
        embed_model = self._get_embed_model()
        llm = self._get_llm()
        index = self._get_index()
        
        # Create a new retriever instance (not shared)
        self.logger.info("[QE] Step 5: Creating retriever ...")
        t0 = time.perf_counter()
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=BEAM_K,
            search_type="similarity",
        )
        self.logger.info(f"[QE] Step 5: Retriever ready (took {(time.perf_counter()-t0):.2f}s)")
        
        # Build filters if user_id/group_ids provided
        if user_filters is None and user_id is not None:
            try:
                self.logger.info("[QE] Step 6: Building user security filters ...")
                user_filters = self._create_user_security_filters(user_id, group_ids or [])
            except Exception as e:
                self.logger.error(f"Failed to build user filters: {e}")

        # Apply filters if available
        if user_filters is not None:
            self.logger.info("[QE] Step 6: Applying user filters to retriever")
            retriever.vector_store_kwargs = {"filters": user_filters}
        
        # Create reranker optionally - default disabled to avoid cold-download stalls
        reranker = None
        if RERANK_ENABLED:
            try:
                # Prefer a local path if present; otherwise fall back to repo name
                model_to_load = RERANK_MODEL_PATH if os.path.exists(RERANK_MODEL_PATH) else RERANK_MODEL_REPO
                self.logger.info(f"[QE] Step 7: Initializing sentence transformer reranker (device=cuda if available, model={model_to_load}) ...")
                t0 = time.perf_counter()
                reranker = SentenceTransformerRerank(
                    model=model_to_load,
                    top_n=FINAL_K,
                    device="cuda" if has_cuda else "cpu",
                )
                self.logger.info(f"[QE] Step 7: Reranker ready (took {(time.perf_counter()-t0):.2f}s)")
            except Exception as rerank_err:
                self.logger.warning(f"[QE] Step 7: Failed to initialize reranker: {rerank_err}. Proceeding without reranker.")
        else:
            self.logger.info("[QE] Step 7: Reranker disabled (ENABLE_RERANKER not set). Proceeding without reranker.")
        
        # Create response synthesizer
        self.logger.info("[QE] Step 8: Creating response synthesizer ...")
        t0 = time.perf_counter()
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            text_qa_template=PromptTemplate(QA_TEMPLATE),
        )
        self.logger.info(f"[QE] Step 8: Response synthesizer ready (took {(time.perf_counter()-t0):.2f}s)")
        
        self.logger.info("[QE] Step 9: Finalizing query engine ...")
        t0 = time.perf_counter()
        engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[reranker] if reranker is not None else [],
            response_synthesizer=response_synthesizer,
        )
        self.logger.info(f"[QE] Step 9: Query engine initialized successfully (took {(time.perf_counter()-t0):.2f}s)")
        return engine

    def query(self, query_text: str, user_id: str, group_ids: List[str], user_filters: Optional[MetadataFilters] = None):
        """Convenience method: create an engine and execute a single query."""
        self.logger.info("[QE] Executing single query via factory.create_query_engine -> engine.query()")
        engine = self.create_query_engine(user_filters=user_filters, user_id=user_id, group_ids=group_ids)
        return engine.query(query_text)

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
#query_service = QueryEngineService()