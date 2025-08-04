#!/usr/bin/env python3
"""
Document processing service for the concurrent RAG system.

This module provides document processing and embedding functionality that can be used
by Celery workers to process user documents with proper isolation and progress tracking.
"""
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import lancedb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore

from .config import config
from .error_handling import StructuredLogger
from .pdf_utils import extract_text_from_document, extract_text_from_pdf, clean_text, is_supported_format, get_supported_extensions

# ---------------------------------------------------------------------------
# Service Functions
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    success: bool
    document_count: int
    chunk_count: int
    processing_time: float
    error: Optional[str] = None
    job_id: Optional[str] = None

class DocumentProcessor:
    """Service class for processing document embeddings with proper error handling and monitoring."""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
    
    def process_documents(
        self, 
        file_paths: List[str], 
        user_id: str, 
        group_id: str,
        job_id: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 20,
        db_path: str = "./multi_user_db.lance",
        table_name: str = "document_embeddings",
        embed_model_name: str = "./models/gte-large-en-v1.5",
        device: str = "cpu"
    ) -> EmbeddingResult:
        """
        Process documents for embedding with proper error handling and monitoring.
        
        Args:
            file_paths: List of document file paths to process (PDF, DOCX, PPTX, XLSX, etc.)
            user_id: ID of the user uploading the documents
            group_id: Group ID for the documents
            job_id: Optional job ID for tracking
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between text chunks
            db_path: Path to the LanceDB database
            table_name: Name of the table to store embeddings
            embed_model_name: Path to the embedding model
            device: Device to use for embedding
            
        Returns:
            EmbeddingResult: Result of the embedding operation
        """
        start_time = time.time()
        
        try:
            if not file_paths:
                return EmbeddingResult(
                    success=False,
                    document_count=0,
                    chunk_count=0,
                    processing_time=0.0,
                    error="No files provided",
                    job_id=job_id
                )
            
            self.logger.info(f"Starting document embedding for user {user_id}", extra={
                'job_id': job_id,
                'user_id': user_id,
                'group_id': group_id,
                'file_count': len(file_paths),
                'files': [os.path.basename(f) for f in file_paths]
            })
            
            # Validate files exist and are supported formats
            valid_files = []
            supported_exts = get_supported_extensions()
            
            for file_path in file_paths:
                if os.path.exists(file_path) and is_supported_format(file_path):
                    valid_files.append(file_path)
                else:
                    ext = os.path.splitext(file_path)[1].lower()
                    self.logger.warning(f"Skipping invalid file: {file_path} (extension: {ext}, supported: {supported_exts})")
            
            if not valid_files:
                return EmbeddingResult(
                    success=False,
                    document_count=0,
                    chunk_count=0,
                    processing_time=time.time() - start_time,
                    error=f"No valid document files found. Supported formats: {', '.join(supported_exts)}",
                    job_id=job_id
                )
            
            # Build nodes from documents
            nodes = self._build_nodes_from_documents(
                valid_files,
                user_id=user_id,
                group_id=group_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            if not nodes:
                return EmbeddingResult(
                    success=False,
                    document_count=len(valid_files),
                    chunk_count=0,
                    processing_time=time.time() - start_time,
                    error="No content extracted from documents",
                    job_id=job_id
                )
            
            # Embed and store
            self._embed_and_store(
                nodes,
                db_path=db_path,
                table_name=table_name,
                embed_model_name=embed_model_name,
                device=device
            )
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"Document embedding completed successfully", extra={
                'job_id': job_id,
                'user_id': user_id,
                'group_id': group_id,
                'document_count': len(valid_files),
                'chunk_count': len(nodes),
                'processing_time': processing_time
            })
            
            return EmbeddingResult(
                success=True,
                document_count=len(valid_files),
                chunk_count=len(nodes),
                processing_time=processing_time,
                job_id=job_id
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Document embedding failed: {str(e)}"
            
            self.logger.error(error_msg, extra={
                'job_id': job_id,
                'user_id': user_id,
                'group_id': group_id,
                'processing_time': processing_time,
                'error': str(e)
            }, exc_info=True)
            
            return EmbeddingResult(
                success=False,
                document_count=len(file_paths) if file_paths else 0,
                chunk_count=0,
                processing_time=processing_time,
                error=error_msg,
                job_id=job_id
            )
    
    def _build_nodes_from_documents(
        self,
        file_paths: List[str],
        user_id: str,
        group_id: str,
        chunk_size: int = 512,
        chunk_overlap: int = 20,
    ):
        """Build nodes from a list of documents with user/group metadata."""
        nodes = []
        for file_path in file_paths:
            document_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            try:
                # Extract text from document using Docling
                pages = extract_text_from_document(file_path)
                
                # Create documents for each page
                documents = []
                for page_text, page_number in pages:
                    cleaned_text = clean_text(page_text)
                    metadata = {
                        "document_name": document_name,
                        "page_number": page_number,
                        "user_id": user_id,
                        "group_id": group_id,
                        "file_format": file_ext,
                        "file_path": file_path,
                    }
                    documents.append(
                        Document(
                            text=cleaned_text,
                            metadata=metadata,
                            id_=f"{document_name}_p{page_number}",
                        )
                    )
                
                # Split documents into chunks
                splitter = SentenceSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    include_metadata=True,
                )
                nodes.extend(splitter.get_nodes_from_documents(documents))
                
                self.logger.info(f"Processed {document_name}: {len(pages)} pages, {len(documents)} documents")
                
            except Exception as e:
                self.logger.error(f"Failed to process {document_name}: {str(e)}", exc_info=True)
                # Continue processing other files even if one fails
                continue
        
        return nodes

    def _build_nodes_from_pdfs(
        self,
        pdf_paths: List[str],
        user_id: str,
        group_id: str,
        chunk_size: int = 512,
        chunk_overlap: int = 20,
    ):
        """Build nodes from a list of PDFs with user/group metadata (backward compatibility)."""
        return self._build_nodes_from_documents(pdf_paths, user_id, group_id, chunk_size, chunk_overlap)
    
    def _embed_and_store(
        self,
        nodes,
        db_path: str,
        table_name: str = "document_embeddings",
        embed_model_name: str = "./models/gte-large-en-v1.5",
        device: str = "cuda",
    ):
        """Embed nodes and store in LanceDB."""
        # Initialize embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name,
            device=device,
            trust_remote_code=True,
        )

        # Connect to LanceDB
        ldb = lancedb.connect(db_path)
        vector_store = LanceDBVectorStore(uri=db_path, table_name=table_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Build index and store embeddings
        index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)

        # Persist metadata
        persist_dir = os.path.join(db_path, "li_storage")
        index.storage_context.persist(persist_dir)

        # Log results
        tbl = ldb.open_table(table_name)
        self.logger.info(f"Stored {tbl.count_rows()} vectors in '{table_name}'")
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the embedding service."""
        try:
            # Test basic functionality
            test_model = HuggingFaceEmbedding(
                model_name="./models/gte-large-en-v1.5",
                device="cpu",
                trust_remote_code=True,
            )
            
            # Test embedding a small text
            test_embedding = test_model.get_text_embedding("test")
            
            return {
                "status": "healthy",
                "embedding_model_loaded": True,
                "embedding_dimension": len(test_embedding) if test_embedding else 0,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
        except Exception as e:
            self.logger.error(f"Embedding service health check failed: {e}", exc_info=True)
            return {
                "status": "unhealthy",
                "error": str(e),
                "embedding_model_loaded": False,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def get_document_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a document file.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        dict: Document information including page count, file size, format, etc.
    """
    try:
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        
        if not is_supported_format(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            supported_exts = get_supported_extensions()
            return {
                "error": f"Unsupported file format: {ext}. Supported formats: {', '.join(supported_exts)}",
                "filename": os.path.basename(file_path)
            }
        
        # Get file stats
        file_stats = os.stat(file_path)
        file_size = file_stats.st_size
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Get document info using utility function
        pages = extract_text_from_document(file_path)
        page_count = len(pages)
        
        return {
            "filename": os.path.basename(file_path),
            "file_format": file_ext,
            "file_size": file_size,
            "page_count": page_count,
            "file_path": file_path,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
    except Exception as e:
        return {
            "error": f"Failed to get document info: {str(e)}",
            "filename": os.path.basename(file_path) if file_path else "unknown"
        }

def estimate_processing_time(file_paths: List[str]) -> Dict[str, Any]:
    """
    Estimate processing time for a list of document files.
    
    Args:
        file_paths: List of document file paths (PDF, DOCX, PPTX, XLSX, etc.)
        
    Returns:
        dict: Estimated processing time and other metrics
    """
    try:
        total_pages = 0
        total_size = 0
        valid_files = 0
        format_counts = {}
        
        for file_path in file_paths:
            if os.path.exists(file_path) and is_supported_format(file_path):
                doc_info = get_document_info(file_path)
                if "error" not in doc_info:
                    total_pages += doc_info.get("page_count", 0)
                    total_size += doc_info.get("file_size", 0)
                    valid_files += 1
                    
                    # Track format distribution
                    file_format = doc_info.get("file_format", "unknown")
                    format_counts[file_format] = format_counts.get(file_format, 0) + 1
        
        # Rough estimates based on typical processing times
        # These should be calibrated based on actual system performance
        # Different formats may have different processing speeds
        base_time_per_page = 2  # seconds per page
        base_time_per_mb = 10   # seconds per MB
        
        # Adjust processing time based on file formats
        format_multipliers = {
            '.pdf': 1.0,     # baseline
            '.docx': 0.8,    # usually faster than PDF
            '.pptx': 1.2,    # may be slower due to complex layouts
            '.xlsx': 0.6,    # usually simpler structure
            '.xls': 0.6,
            '.html': 0.5,    # simple text processing
            '.md': 0.3,      # very simple
            '.txt': 0.2,     # fastest
        }
        
        # Calculate weighted processing time
        weighted_multiplier = 1.0
        if format_counts:
            total_files = sum(format_counts.values())
            weighted_multiplier = sum(
                format_multipliers.get(fmt, 1.0) * count / total_files
                for fmt, count in format_counts.items()
            )
        
        estimated_seconds = (
            (total_pages * base_time_per_page) + 
            (total_size / (1024 * 1024) * base_time_per_mb)
        ) * weighted_multiplier
        
        return {
            "estimated_time_seconds": int(estimated_seconds),
            "estimated_time_minutes": round(estimated_seconds / 60, 1),
            "total_pages": total_pages,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "valid_files": valid_files,
            "invalid_files": len(file_paths) - valid_files,
            "format_distribution": format_counts,
            "supported_formats": get_supported_extensions()
        }
        
    except Exception as e:
        return {
            "error": f"Failed to estimate processing time: {str(e)}",
            "estimated_time_seconds": 0
        }

# Global service instance
document_processor = DocumentProcessor()