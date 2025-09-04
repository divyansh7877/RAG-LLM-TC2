#!/usr/bin/env python3
"""
Main service for processing and embedding documents.

This module orchestrates the document processing workflow, including text extraction,
chunking, embedding, and storage, using optimized, singleton components.
"""
import os
import time
import hashlib
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from .lancedb_client import get_db_connection
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from .error_handling import StructuredLogger
from .pdf_utils import extract_text_from_document, clean_text, is_supported_format, get_document_info as _get_doc_info
from .embedding_optimizer import get_embedding_model, optimize_for_batch_processing, clear_embedding_model
from .gpu_memory_manager import gpu_memory_manager

# --- Data Classes ---
@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    success: bool
    document_count: int
    chunk_count: int
    processing_time: float
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)

# --- Main Service Class ---
class DocumentProcessor:
    """A service to process and embed documents into a vector store."""
    
    def __init__(self, db_path: str = "./multi_user_db.lance", table_name: str = "document_embeddings_v2", embed_model_name: str = "./models/gte-large-en-v1.5", device: Optional[str] = None):
        self.logger = StructuredLogger(__name__)
        self.db_path = db_path
        self.table_name = table_name
        self.embed_model_name = embed_model_name
        # Resolve device from config when not explicitly provided
        try:
            from .config import config as _config
            self.device = device or ("cuda" if _config.HAS_CUDA else "cpu")
        except Exception:
            self.device = device or "cpu"
        self.db = None
        self.vector_store = None

    def _initialize_vector_store(self):
        """Initializes the LanceDB connection and vector store if not already done."""
        if self.vector_store is None:
            import os as _os
            resolved_db_path = _os.path.abspath(self.db_path)
            self.logger.info(
                f"Initializing LanceDB vector store... path={resolved_db_path}, table={self.table_name}"
            )
            try:
                self.logger.info("Connecting to LanceDB (singleton)...")
                self.db = get_db_connection()
                self.logger.info("Connected to LanceDB.")
            except Exception as e:
                self.logger.error(f"Failed connecting to LanceDB at {resolved_db_path}: {e}", exc_info=True)
                raise

            try:
                self.logger.info("Creating LanceDBVectorStore instance (using existing DB connection if supported)...")
                try:
                    # Prefer passing the already-open DB connection to avoid duplicate locks
                    self.vector_store = LanceDBVectorStore(db=self.db, table_name=self.table_name)  # type: ignore[arg-type]
                    self.logger.info("LanceDB vector store initialized via db connection.")
                except TypeError:
                    # Fallback for older versions that don't accept a 'db' parameter
                    self.vector_store = LanceDBVectorStore(uri=resolved_db_path, table_name=self.table_name)
                    self.logger.info("LanceDB vector store initialized via URI.")
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize LanceDBVectorStore at {resolved_db_path} table {self.table_name}: {e}",
                    exc_info=True,
                )
                raise

    def process_documents(
        self, 
        file_paths: List[str], 
        user_id: str, 
        group_id: str,
        chunk_size: int = 512,
        chunk_overlap: int = 20
    ) -> EmbeddingResult:
        """
        Orchestrates the end-to-end document processing and embedding workflow with
        intelligent GPU memory management to prevent CUDA OOM errors.
        
        Uses sequential resource allocation:
        1. Text extraction (Docling) - uses GPU if sufficient memory
        2. Clear GPU cache and free memory
        3. Embedding generation - uses GPU if sufficient memory
        """
        self._initialize_vector_store()
        start_time = time.time()
        self.logger.info(f"Starting document processing for user '{user_id}'. Files: {len(file_paths)}")
        
        # Log initial GPU memory state
        initial_memory = gpu_memory_manager.get_memory_info()
        if initial_memory:
            self.logger.info(f"Initial GPU memory usage: {initial_memory.utilization_percent:.1f}%")

        # 1) Normalize and validate input paths
        normalized_files = [os.path.abspath(fp) for fp in file_paths]
        valid_files = []
        for fp in normalized_files:
            exists = os.path.exists(fp)
            if not exists:
                self.logger.warning(f"Input file not found, skipping: {fp}")
                continue
            valid_files.append(fp)
        if not valid_files:
            self.logger.error("No valid document files after existence check.")
            return EmbeddingResult(False, 0, 0, 0.0, "No valid document files provided.")

        # 2) Build nodes from documents with intelligent GPU/CPU allocation
        self.logger.info("Phase 1: Text extraction with Docling")
        try:
            with gpu_memory_manager.managed_gpu_allocation(
                "Document text extraction", 
                clear_cache_before=True, 
                clear_cache_after=True
            ):
                nodes = self._build_nodes(valid_files, user_id, group_id, chunk_size, chunk_overlap)
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}", exc_info=True)
            return EmbeddingResult(False, len(valid_files), 0, 0.0, f"Text extraction failed: {e}")
        
        if not nodes:
            return EmbeddingResult(False, len(valid_files), 0, 0.0, "No content could be extracted from documents.")
        
        self.logger.info(f"Phase 1 complete: Extracted {len(nodes)} text chunks")
        
        # 3) Clear any existing embedding model to free GPU memory
        self.logger.info("Phase 2: Preparing for embedding generation")
        clear_embedding_model()
        gpu_memory_manager.clear_cache()
        
        # Check GPU memory before loading embedding model
        memory_after_extraction = gpu_memory_manager.get_memory_info()
        if memory_after_extraction:
            self.logger.info(f"GPU memory after text extraction: {memory_after_extraction.utilization_percent:.1f}%")
        
        # 4) Load embedding model with intelligent device selection
        try:
            embed_model = get_embedding_model(self.embed_model_name, self.device)
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
            return EmbeddingResult(False, len(valid_files), 0, 0.0, f"Embedder init failed: {e}")

        # 5) Embed nodes and store with memory management
        self.logger.info("Phase 3: Generating embeddings")
        try:
            with gpu_memory_manager.managed_gpu_allocation(
                "Embedding generation",
                clear_cache_before=False,  # Already cleared above
                clear_cache_after=True
            ):
                self._embed_and_store(nodes, embed_model)
                
            processing_time = time.time() - start_time
            self.logger.info(f"Successfully processed {len(valid_files)} documents in {processing_time:.2f}s.")
            return EmbeddingResult(True, len(valid_files), len(nodes), processing_time)
            
        except Exception as e:
            self.logger.error(f"Embedding failed: {e}", exc_info=True)
            
            # Try CPU fallback if GPU embedding failed
            if self.device != "cpu":
                self.logger.info("Attempting CPU fallback for embedding generation")
                try:
                    clear_embedding_model()
                    gpu_memory_manager.clear_cache()
                    
                    cpu_embed_model = get_embedding_model(self.embed_model_name, "cpu")
                    self._embed_and_store(nodes, cpu_embed_model)
                    
                    processing_time = time.time() - start_time
                    self.logger.info(f"Successfully processed {len(valid_files)} documents using CPU fallback in {processing_time:.2f}s.")
                    return EmbeddingResult(True, len(valid_files), len(nodes), processing_time)
                    
                except Exception as cpu_error:
                    self.logger.error(f"CPU fallback also failed: {cpu_error}", exc_info=True)
                    return EmbeddingResult(False, len(valid_files), len(nodes), 0.0, f"Both GPU and CPU embedding failed: {e}, {cpu_error}")
            
            return EmbeddingResult(False, len(valid_files), len(nodes), 0.0, str(e))

    # Backwards-compatible helper expected by some tests
    def _clean_text(self, raw_text: str) -> str:
        from .pdf_utils import clean_text as _clean
        return _clean(raw_text)

    def _build_nodes(self, file_paths: List[str], user_id: str, group_id: str, 
                     chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Extracts text with intelligent GPU allocation, cleans it, and builds structured Document nodes."""
        all_docs = []
        
        for file_path in file_paths:
            try:
                # Calculate total file sizes to decide on processing strategy
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                
                # Determine if we should force CPU for very large files to preserve GPU memory
                force_cpu = file_size_mb > 100  # Force CPU for files larger than 100MB
                
                if force_cpu:
                    self.logger.info(f"Large file detected ({file_size_mb:.1f}MB), forcing CPU for text extraction")
                
                pages = extract_text_from_document(file_path, force_cpu=force_cpu)
                
                for page_text, page_num in pages:
                    cleaned_text = clean_text(page_text)
                    if not cleaned_text:
                        continue
                    
                    doc = Document(
                        text=cleaned_text,
                        metadata={
                            "document_name": os.path.basename(file_path),
                            "page_number": page_num,
                            "user_id": user_id,
                            "group_id": group_id,
                            "file_hash": self._calculate_file_hash(file_path),
                            "file_size_mb": round(file_size_mb, 2)
                        }
                    )
                    all_docs.append(doc)
                    
                # Clear GPU cache between large files
                if file_size_mb > 50:
                    gpu_memory_manager.clear_cache()
                    
            except Exception as e:
                self.logger.error(f"Failed to build nodes for {file_path}: {e}", exc_info=True)
                # Clear GPU cache on error to prevent memory leaks
                gpu_memory_manager.clear_cache()
                continue # Skip to the next file

        if not all_docs:
            return []

        # Use a sentence splitter for chunking
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.get_nodes_from_documents(all_docs)

    def _embed_and_store(self, nodes: List[Document], embed_model):
        """
        Embeds the given nodes and stores them in LanceDB with GPU memory management.
        Uses adaptive batching based on available GPU memory to prevent OOM errors.
        """
        total_nodes = len(nodes)
        self.logger.info(f"Embedding {total_nodes} nodes.")
        
        # Check current GPU memory before embedding
        memory_info = gpu_memory_manager.get_memory_info()
        if memory_info:
            self.logger.info(f"GPU memory before embedding: {memory_info.utilization_percent:.1f}%")
            # Use conservative batching if memory is already high
            force_conservative = memory_info.utilization_percent > 70
        else:
            force_conservative = False

        # Determine batching strategy with GPU memory awareness
        batch_cfg = optimize_for_batch_processing(total_nodes, force_conservative=force_conservative)
        node_batch_size = batch_cfg.get("node_batch_size", 64)
        embed_bs = batch_cfg.get("embed_batch_size")
        
        self.logger.info(f"Using batch sizes: embed_batch_size={embed_bs}, node_batch_size={node_batch_size}")
        
        try:
            if embed_bs and hasattr(embed_model, "embed_batch_size"):
                setattr(embed_model, "embed_batch_size", embed_bs)
                self.logger.info(f"Set embed_batch_size={embed_bs} for this run.")
        except Exception:
            pass

        # Storage context with existing vector store
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        if total_nodes <= node_batch_size:
            try:
                VectorStoreIndex(nodes, embed_model=embed_model, storage_context=storage_context)
                self.logger.info(f"Successfully stored {total_nodes} new vectors.")
            except Exception as e:
                # Handle potential CUDA OOM during embedding
                if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                    self.logger.warning(f"GPU OOM during embedding, clearing cache and retrying: {e}")
                    self._force_free_cuda()
                    # Retry with more conservative settings
                    conservative_batch = optimize_for_batch_processing(total_nodes, force_conservative=True)
                    if hasattr(embed_model, "embed_batch_size"):
                        setattr(embed_model, "embed_batch_size", conservative_batch.get("embed_batch_size", 4))
                    VectorStoreIndex(nodes, embed_model=embed_model, storage_context=storage_context)
                    self.logger.info(f"Successfully stored {total_nodes} new vectors after GPU memory recovery.")
                else:
                    raise
            finally:
                self._maybe_free_cuda()
            return

        # Batched insertion for large corpora with memory monitoring
        index = None
        num_batches = (total_nodes + node_batch_size - 1) // node_batch_size
        
        for batch_idx in range(0, total_nodes, node_batch_size):
            batch_nodes = nodes[batch_idx: batch_idx + node_batch_size]
            human_batch = (batch_idx // node_batch_size) + 1
            self.logger.info(
                f"Processing batch {human_batch}/{num_batches}: nodes {batch_idx+1}-{min(batch_idx+len(batch_nodes), total_nodes)}"
            )
            
            # Check GPU memory before each batch
            if gpu_memory_manager.is_gpu_available():
                memory_info = gpu_memory_manager.get_memory_info()
                if memory_info and memory_info.utilization_percent > 85:
                    self.logger.info(f"High GPU memory usage ({memory_info.utilization_percent:.1f}%), clearing cache")
                    self._force_free_cuda()

            try:
                if index is None:
                    index = VectorStoreIndex(batch_nodes, embed_model=embed_model, storage_context=storage_context)
                else:
                    index.insert_nodes(batch_nodes)
            except Exception as e:
                if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                    self.logger.warning(f"GPU OOM in batch {human_batch}, clearing cache and retrying: {e}")
                    self._force_free_cuda()
                    # Reduce batch size for this batch
                    if hasattr(embed_model, "embed_batch_size"):
                        current_batch_size = getattr(embed_model, "embed_batch_size", 16)
                        new_batch_size = max(2, current_batch_size // 2)
                        setattr(embed_model, "embed_batch_size", new_batch_size)
                        self.logger.info(f"Reduced embed_batch_size to {new_batch_size} for memory recovery")
                    
                    # Retry the batch
                    if index is None:
                        index = VectorStoreIndex(batch_nodes, embed_model=embed_model, storage_context=storage_context)
                    else:
                        index.insert_nodes(batch_nodes)
                else:
                    raise
            
            # Clear cache between batches to prevent memory buildup
            if human_batch % 3 == 0:  # Clear every 3 batches
                self._maybe_free_cuda()

        self.logger.info(f"Successfully stored {total_nodes} new vectors (batched).")
        self._maybe_free_cuda()

    def _maybe_free_cuda(self):
        """Best-effort CUDA cache release (no-op on CPU)."""
        try:
            gpu_memory_manager.clear_cache()
        except Exception:
            pass
    
    def _force_free_cuda(self):
        """Aggressive GPU memory cleanup for OOM recovery."""
        try:
            import torch
            import gc
            
            if torch.cuda.is_available():
                # Clear all caches
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
                # Force garbage collection multiple times
                for _ in range(3):
                    gc.collect()
                
                # Synchronize to ensure cleanup completes
                torch.cuda.synchronize()
                
                self.logger.info("Aggressive CUDA cache cleanup completed")
        except Exception as e:
            self.logger.warning(f"Failed to force CUDA cleanup: {e}")

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file's content for deduplication."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


# --- Helper functions for tests/backwards-compat ---
def get_document_info(file_path: str) -> Dict[str, Any]:
    """Expose pdf_utils.get_document_info for external callers/tests."""
    return _get_doc_info(file_path)

def estimate_processing_time(file_paths: List[str]) -> Dict[str, Any]:
    """
    Provide a lightweight estimate of processing time based on file size and page count.
    This is used by tests; it's heuristic and safe for dry-run.
    """
    if not file_paths:
        return {"error": "No files provided"}

    total_size = 0
    total_pages = 0
    valid_files: List[str] = []

    for path in file_paths:
        if os.path.exists(path) and is_supported_format(path):
            valid_files.append(path)
            try:
                info = _get_doc_info(path)
                total_size += int(info.get("file_size", 0))
                total_pages += int(info.get("page_count", 0))
            except Exception:
                # Ignore failures for estimation purposes
                continue

    if not valid_files:
        return {"error": "No valid files provided"}

    # Heuristic: 0.2s per page + 0.5s per MB, minimum 1s
    total_size_mb = round(total_size / (1024 * 1024), 2)
    estimated_time = max(1.0, total_pages * 0.2 + total_size_mb * 0.5)

    # Format distribution by extension
    format_distribution: Dict[str, int] = {}
    for path in valid_files:
        ext = os.path.splitext(path)[1].lower()
        format_distribution[ext] = format_distribution.get(ext, 0) + 1

    return {
        "estimated_time_seconds": round(estimated_time, 2),
        "total_pages": total_pages,
        "total_size_mb": total_size_mb,
        "valid_files": len(valid_files),
        "format_distribution": format_distribution,
    }

def health_check() -> Dict[str, Any]:
    """
    Simple health check report for the document processing subsystem.
    """
    try:
        # We don't load the model here, only reference configuration
        embed_model_name = "./models/gte-large-en-v1.5"
        return {
            "status": "healthy",
            "embedding_model_loaded": False,
            "embedding_dimension": None,
            "model_name": embed_model_name,
            "timestamp": time.time(),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
