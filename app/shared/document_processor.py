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
from .embedding_optimizer import get_embedding_model, optimize_for_batch_processing

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
    
    def __init__(self, db_path: str = "./multi_user_db.lance", table_name: str = "document_embeddings", embed_model_name: str = "./models/gte-large-en-v1.5", device: str = "cpu"):
        self.logger = StructuredLogger(__name__)
        self.db_path = db_path
        self.table_name = table_name
        self.embed_model_name = embed_model_name
        self.device = device
        self.db = None
        self.vector_store = None

    def _initialize_vector_store(self):
        """Initializes the LanceDB connection and vector store if not already done."""
        if self.vector_store is None:
            self.logger.info("Initializing LanceDB vector store...")
            self.db = get_db_connection()
            self.vector_store = LanceDBVectorStore(uri=self.db_path, table_name=self.table_name)
            self.logger.info("LanceDB vector store initialized.")

    def process_documents(
        self, 
        file_paths: List[str], 
        user_id: str, 
        group_id: str,
        chunk_size: int = 512,
        chunk_overlap: int = 20
    ) -> EmbeddingResult:
        """
        Orchestrates the end-to-end document processing and embedding workflow.
        """
        self._initialize_vector_store()
        start_time = time.time()
        self.logger.info(f"Starting document processing for user '{user_id}'. Files: {len(file_paths)}")

        # 1. Load the embedding model for this operation
        embed_model = get_embedding_model(self.embed_model_name, self.device)

        # 2. Filter out unsupported or non-existent files
        valid_files = [fp for fp in file_paths if os.path.exists(fp) and is_supported_format(fp)]
        if not valid_files:
            return EmbeddingResult(False, 0, 0, 0.0, "No valid document files provided.")

        # 3. Build nodes from documents
        nodes = self._build_nodes(valid_files, user_id, group_id, chunk_size, chunk_overlap)
        if not nodes:
            return EmbeddingResult(False, len(valid_files), 0, 0.0, "No content could be extracted from documents.")

        # 4. Embed nodes and store them in the vector database
        try:
            self._embed_and_store(nodes, embed_model)
            processing_time = time.time() - start_time
            self.logger.info(f"Successfully processed {len(valid_files)} documents in {processing_time:.2f}s.")
            return EmbeddingResult(True, len(valid_files), len(nodes), processing_time)
        except Exception as e:
            self.logger.error(f"Embedding failed: {e}", exc_info=True)
            return EmbeddingResult(False, len(valid_files), len(nodes), 0.0, str(e))

    # Backwards-compatible helper expected by some tests
    def _clean_text(self, raw_text: str) -> str:
        from .pdf_utils import clean_text as _clean
        return _clean(raw_text)

    def _build_nodes(self, file_paths: List[str], user_id: str, group_id: str, 
                     chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Extracts text, cleans it, and builds structured Document nodes."""
        all_docs = []
        for file_path in file_paths:
            try:
                pages = extract_text_from_document(file_path)
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
                            "file_hash": self._calculate_file_hash(file_path)
                        }
                    )
                    all_docs.append(doc)
            except Exception as e:
                self.logger.error(f"Failed to build nodes for {file_path}: {e}", exc_info=True)
                continue # Skip to the next file

        if not all_docs:
            return []

        # Use a sentence splitter for chunking
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.get_nodes_from_documents(all_docs)

    def _embed_and_store(self, nodes: List[Document], embed_model):
        """
        Embeds the given nodes and stores them in LanceDB, using an explicit embed model.
        """
        total_nodes = len(nodes)
        self.logger.info(f"Embedding {total_nodes} nodes.")

        # Use the provided embedding model
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        index = VectorStoreIndex(nodes, embed_model=embed_model, storage_context=storage_context)

        self.logger.info(f"Successfully stored {total_nodes} new vectors.")

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
