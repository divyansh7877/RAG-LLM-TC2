#!/usr/bin/env python3
"""
Thread-safe query engine factory for the concurrent RAG system.

- Single-shot synthesis: response_mode="compact" (reduces LLM call count).
- Lower similarity_top_k by default (env SIM_TOP_K, default 4).
- OpenAI LLM: timeout + max_retries=0 (no blind retries on 429).
- Quota-aware error handling: clear fail-fast path on insufficient_quota.
- Token usage logging via LlamaIndex callbacks.
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
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

import torch

from .config import config
from .error_handling import StructuredLogger
from .openai_config import get_openai_params, get_retrieval_config

# --------------------------------------------------------------------------- 
# Configuration
# --------------------------------------------------------------------------- 
DB_PATH = getattr(config, "LANCEDB_PATH", "./multi_user_db.lance")
TABLE_NAME = os.getenv("LANCEDB_TABLE_NAME", "document_embeddings_v2")
EMBED_MODEL_NAME = getattr(config, "EMBEDDING_MODEL_PATH", "./models/gte-large-en-v1.5")
has_cuda = getattr(config, "HAS_CUDA")

# Runtime knobs / env overrides for optimized usage
device = "cuda" if has_cuda else "cpu"
EMBEDDING_DEVICE = device

# Get OpenAI and retrieval config from centralized module
try:
    _retrieval_config = get_retrieval_config()
    SIM_TOP_K = int(os.getenv("SIM_TOP_K", str(_retrieval_config["similarity_top_k"])))
    FINAL_K = int(os.getenv("FINAL_K", str(_retrieval_config["final_k"])))
except Exception:
    # Fallback values if config module fails
    SIM_TOP_K = int(os.getenv("SIM_TOP_K", "8"))
    FINAL_K = int(os.getenv("FINAL_K", "5"))

RERANK_MODEL_REPO = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH", "./models/cross-encoder/ms-marco-MiniLM-L6-v2")
RERANK_ENABLED = os.getenv("ENABLE_RERANKER", "1").lower() in ("1", "true", "yes")
N_THREADS = mp.cpu_count()

# --------------------------------------------------------------------------- 
# Prompt enforcing source citations
# --------------------------------------------------------------------------- 
# Optimized prompt template for better OpenAI responses
QA_TEMPLATE = """You are an expert AI assistant that provides accurate, well-structured answers based on provided document context.

INSTRUCTIONS:
1. Analyze the provided context carefully and provide a comprehensive answer to the question
2. Structure your response clearly with key points and explanations
3. ALWAYS cite your sources using the format: (Source: {document_name}, Page: {page_number})
4. If the context doesn't contain sufficient information, state this clearly and suggest what additional information might be needed
5. Provide actionable insights when relevant
6. Keep your response focused and avoid unnecessary repetition

------------------------
CONTEXT INFORMATION:
{context_str}

------------------------
USER QUESTION:
{query_str}

------------------------
EXPERT RESPONSE:
"""

# --------------------------------------------------------------------------- 
# Thread-safe Query Engine Factory
# --------------------------------------------------------------------------- 

class QueryEngineFactory:
    """Thread-safe factory for creating query engines with proper isolation."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._embed_model = None
        self._llm = None
        self._vector_store = None
        self._index = None
        self._query_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._cache_ttl_seconds: int = 3600
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self.logger = StructuredLogger(__name__)
        self._config_logged = False

        # Token usage handler (visible in logs after each query)
        self._token_handler = TokenCountingHandler()
        Settings.callback_manager = CallbackManager([self._token_handler])

    def _log_configuration(self) -> None:
        if self._config_logged:
            return
        self._config_logged = True
        try:
            from .openai_config import get_openai_config
            openai_config = get_openai_config()
            self.logger.info(
                "[QE] Configuration: "
                f"DB_PATH={os.path.abspath(DB_PATH)}, TABLE_NAME={TABLE_NAME}, "
                f"EMBED_MODEL_NAME={EMBED_MODEL_NAME}, LLM_PROVIDER=OpenAI, OPENAI_MODEL_NAME={openai_config.model_name}, "
                f"CUDA={has_cuda}, device={device}, "
                f"SIM_TOP_K={SIM_TOP_K}, FINAL_K={FINAL_K}, "
                f"RERANK_ENABLED={RERANK_ENABLED}, RERANK_MODEL_PATH={RERANK_MODEL_PATH}, "
                f"OPENAI_TIMEOUT_SEC={openai_config.timeout_sec}, OPENAI_MAX_RETRIES={openai_config.max_retries}"
            )
        except Exception:
            pass
    
    def _get_embed_model(self):
        if self._embed_model is None:
            with self._lock:
                if self._embed_model is None:
                    from .embedding_optimizer import get_embedding_model
                    from .gpu_memory_manager import gpu_memory_manager
                    
                    t0 = time.perf_counter()
                    self.logger.info("[QE] Step 1: Initializing optimized embedding model ...")
                    
                    # Use the optimized embedding model with GPU memory management
                    embed_device = EMBEDDING_DEVICE
                    
                    # Check GPU availability for embedding model
                    if embed_device == "cuda" and not gpu_memory_manager.can_allocate_for_embedding(16, 384):
                        self.logger.warning("[QE] Insufficient GPU memory for embedding model, using CPU")
                        embed_device = "cpu"
                    
                    self._embed_model = get_embedding_model(EMBED_MODEL_NAME, embed_device)
                    self.logger.info(f"[QE] Step 1: Optimized embedding model ready on device={embed_device} (took {(time.perf_counter()-t0):.2f}s)")
        return self._embed_model
    
    def _get_llm(self):
        if self._llm is None:
            with self._lock:
                if self._llm is None:
                    # Get centralized OpenAI configuration
                    openai_params = get_openai_params()
                    self.logger.info(f"[QE] Step 2: Initializing OpenAI LLM ({openai_params['model']}) ...")
                    
                    t0 = time.perf_counter()
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        self.logger.error("OPENAI_API_KEY environment variable not set.")
                        raise ValueError("OPENAI_API_KEY must be set to use the OpenAI LLM.")

                    # Use centralized OpenAI configuration
                    self._llm = OpenAI(**openai_params)
                    self.logger.info(f"[QE] Step 2: LLM ready (took {(time.perf_counter()-t0):.2f}s)")
        return self._llm
    
    def _get_vector_store(self):
        if self._vector_store is None:
            with self._lock:
                if self._vector_store is None:
                    preferred_table = TABLE_NAME
                    try:
                        self.logger.info(f"[QE] Step 3: Opening LanceDB vector store via shared connection (table={preferred_table}) ...")
                        t0 = time.perf_counter()
                        try:
                            from .lancedb_client import get_db_connection
                            db = get_db_connection()
                            self._vector_store = LanceDBVectorStore(db=db, table_name=preferred_table)  # type: ignore[arg-type]
                            self.logger.info(f"[QE] Step 3: Vector store opened via shared connection (took {(time.perf_counter()-t0):.2f}s)")
                        except TypeError:
                            self.logger.info(f"[QE] Step 3: Fallback to URI open at {os.path.abspath(DB_PATH)} ...")
                            self._vector_store = LanceDBVectorStore(
                                uri=DB_PATH,
                                table_name=preferred_table,
                                mode="r",
                            )
                            self.logger.info(f"[QE] Step 3: Vector store opened via URI (took {(time.perf_counter()-t0):.2f}s)")
                    except Exception as primary_err:
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
        if not user_id or not user_id.strip():
            raise ValueError("User ID is required")
        if not group_ids or len(group_ids) == 0:
            raise ValueError("At least one group ID is required")
        try:
            user_filter = ExactMatchFilter(key="user_id", value=user_id)
            group_filters = [ExactMatchFilter(key="group_id", value=group_id) for group_id in group_ids]
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
        """Create a new query engine instance with optional user filters."""
        self._log_configuration()
        self.logger.info("[QE] Creating query engine (building components)...")

        # Shared components
        _ = self._get_embed_model()
        llm = self._get_llm()
        index = self._get_index()
        
        # Retriever
        self.logger.info("[QE] Step 5: Creating retriever ...")
        t0 = time.perf_counter()
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=SIM_TOP_K,
            search_type="similarity",
        )
        self.logger.info(f"[QE] Step 5: Retriever ready (took {(time.perf_counter()-t0):.2f}s)")
        
        # User filters
        if user_filters is None and user_id is not None:
            try:
                self.logger.info("[QE] Step 6: Building user security filters ...")
                user_filters = self._create_user_security_filters(user_id, group_ids or [])
            except Exception as e:
                self.logger.error(f"Failed to build user filters: {e}")

        if user_filters is not None:
            self.logger.info("[QE] Step 6: Applying user filters to retriever")
            retriever.vector_store_kwargs = {"filters": user_filters}
        
        # Optional reranker
        reranker = None
        if RERANK_ENABLED:
            try:
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
        
        # Optimized response synthesizer for OpenAI
        self.logger.info("[QE] Step 8: Creating optimized response synthesizer ...")
        t0 = time.perf_counter()
        response_synthesizer = get_response_synthesizer(
            llm=llm,
            response_mode="compact",                # Single-shot for efficiency
            text_qa_template=PromptTemplate(QA_TEMPLATE),
            streaming=False,                        # Disable streaming for stability
            use_async=False,                       # Sync mode for better error handling
        )
        self.logger.info(f"[QE] Step 8: Optimized response synthesizer ready (took {(time.perf_counter()-t0):.2f}s)")
        
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
            test_filters = MetadataFilters(
                filters=[ExactMatchFilter(key="user_id", value="health_check")],
                condition="or"
            )
            _ = self.create_query_engine(user_filters=test_filters)
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
        with self._lock:
            self._embed_model = None
            self._llm = None
            self._vector_store = None
            self._index = None
            now = time.time()
            keys_to_delete = [k for k, (ts, _) in self._query_cache.items() if now - ts > self._cache_ttl_seconds]
            for k in keys_to_delete:
                self._query_cache.pop(k, None)
            if len(self._query_cache) > 1000:
                self._query_cache.clear()
            self.logger.info("Query engine factory resources have been cleaned up.")

# --------------------------------------------------------------------------- 
# Query Processing
# --------------------------------------------------------------------------- 

@dataclass
class QueryResult:
    response: str
    sources: List[Dict[str, Any]]
    processing_time: float
    query_id: Optional[str] = None
    error: Optional[str] = None

@dataclass
class QuerySource:
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
            
            # Build user filters
            user_filter = ExactMatchFilter(key="user_id", value=user_id)
            group_filters = [ExactMatchFilter(key="group_id", value=group_id) for group_id in group_ids]
            filters = MetadataFilters(filters=[user_filter] + group_filters, condition="or")
            
            # Create engine & run
            query_engine = self.factory.create_query_engine(user_filters=filters)
            response = query_engine.query(query_text)

            # Collect sources
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

            # Token usage log
            h = self.factory._token_handler
            try:
                self.logger.info(
                    f"[Tokens] prompt={h.prompt_llm_token_count} completion={h.completion_llm_token_count} total={h.total_llm_token_count}",
                    extra={'query_id': query_id}
                )
            except Exception:
                pass

            return QueryResult(
                response=str(response.response),
                sources=sources,
                processing_time=processing_time,
                query_id=query_id
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            emsg = str(e)
            lowered = emsg.lower()

            # Quota-aware fast fail
            quota_hit = ("insufficient_quota" in lowered) or ("error code: 429" in lowered and "quota" in lowered)
            if quota_hit:
                friendly = "Model provider quota exceeded. Check billing or switch provider, then retry."
                self.logger.error(friendly + f" Raw error: {emsg}", extra={'query_id': query_id}, exc_info=False)
                return QueryResult(
                    response=friendly,
                    sources=[],
                    processing_time=processing_time,
                    query_id=query_id,
                    error="insufficient_quota"
                )

            self.logger.error(f"Query processing failed: {emsg}", extra={
                'query_id': query_id,
                'user_id': user_id,
                'processing_time': processing_time,
                'error': emsg
            }, exc_info=True)
            
            return QueryResult(
                response="I encountered an error while processing your query. Please try again.",
                sources=[],
                processing_time=processing_time,
                query_id=query_id,
                error=emsg
            )
    
    def health_check(self) -> Dict[str, Any]:
        return self.factory.health_check()

# --------------------------------------------------------------------------- 
# Utility
# --------------------------------------------------------------------------- 

def create_user_filters(user_id: str, group_ids: List[str]) -> MetadataFilters:
    user_filter = ExactMatchFilter(key="user_id", value=user_id)
    group_filters = [ExactMatchFilter(key="group_id", value=group_id) for group_id in group_ids]
    return MetadataFilters(filters=[user_filter] + group_filters, condition="or")

# Global instances
query_engine_factory = QueryEngineFactory()
query_service = QueryEngineService()
