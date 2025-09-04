#!/usr/bin/env python3
"""
Document processing utilities for the concurrent RAG system.

This module provides utilities for extracting text and metadata from various 
document formats using optimized GPU/CPU allocation to prevent CUDA OOM errors.
"""
import re
import os
import contextlib
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from .gpu_memory_manager import gpu_memory_manager

logger = logging.getLogger(__name__)

# --- Module-level Singleton for Docling Converter ---
# Initialize with default settings - GPU/CPU selection handled per-operation
DOCUMENT_CONVERTER = DocumentConverter()

# --- Supported Formats ---
SUPPORTED_FORMATS = {
    '.pdf': InputFormat.PDF,
    '.docx': InputFormat.DOCX,
    '.pptx': InputFormat.PPTX,
    '.xlsx': InputFormat.XLSX,
    '.xls': InputFormat.XLSX,
    '.html': InputFormat.HTML,
    '.md': InputFormat.MD,
    '.csv': InputFormat.CSV,
}

def get_supported_extensions() -> List[str]:
    """Get a list of supported file extensions."""
    return list(SUPPORTED_FORMATS.keys())

def is_supported_format(file_path: str) -> bool:
    """Check if a file format is supported based on its extension."""
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_FORMATS

def _should_use_gpu_for_docling(file_size_mb: int) -> bool:
    """
    Determine if GPU should be used for Docling processing based on available memory.
    
    Args:
        file_size_mb: Size of the file being processed in MB
        
    Returns:
        True if GPU should be used, False if CPU fallback should be used
    """
    if not gpu_memory_manager.is_gpu_available():
        logger.info("GPU not available, using CPU for Docling")
        return False
    
    # Check if GPU has sufficient memory for this file size
    if gpu_memory_manager.can_allocate_for_docling(file_size_mb):
        logger.info(f"Using GPU for Docling processing (file size: {file_size_mb}MB)")
        return True
    else:
        logger.info(f"Insufficient GPU memory for file size {file_size_mb}MB, using CPU for Docling")
        return False

def extract_text_from_document(file_path: str, force_cpu: bool = False) -> List[Tuple[str, int]]:
    """
    Extract text from a document file using intelligent GPU/CPU allocation.
    Uses GPU when sufficient memory is available, falls back to CPU otherwise.

    Args:
        file_path: Path to the document file.
        force_cpu: If True, force CPU usage regardless of GPU availability.

    Returns:
        A list of (page_text, page_number) tuples.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not is_supported_format(file_path):
        raise ValueError(f"Unsupported file format for {file_path}")

    # Calculate file size for memory estimation
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logger.info(f"Processing file: {os.path.basename(file_path)} ({file_size_mb:.2f}MB)")
    
    # Determine processing strategy
    use_gpu = not force_cpu and _should_use_gpu_for_docling(file_size_mb)
    
    try:
        if use_gpu:
            # Use GPU with managed allocation
            with gpu_memory_manager.managed_gpu_allocation(
                f"Docling processing: {os.path.basename(file_path)}",
                clear_cache_before=True,
                clear_cache_after=True
            ):
                result = DOCUMENT_CONVERTER.convert(file_path)
        else:
            # Force CPU usage
            with gpu_memory_manager.force_cpu_context():
                result = DOCUMENT_CONVERTER.convert(file_path)
        
        doc = result.document

        # Prefer markdown export when available (gives a single-page coherent text)
        if hasattr(doc, 'export_to_markdown'):
            try:
                md = doc.export_to_markdown()
                if md:
                    logger.info(f"Successfully extracted text using {'GPU' if use_gpu else 'CPU'}")
                    return [(md, 1)]
            except Exception as e:
                logger.warning(f"Markdown export failed: {e}, falling back to page-wise extraction")

        # Otherwise, fall back to page-wise text if present
        if hasattr(doc, 'pages') and getattr(doc, 'pages', None):
            pages = [(getattr(page, 'text', '') or '', i + 1) for i, page in enumerate(doc.pages)]
            logger.info(f"Successfully extracted {len(pages)} pages using {'GPU' if use_gpu else 'CPU'}")
            return pages

        # Or top-level text as a single page
        if hasattr(doc, 'text') and getattr(doc, 'text', None):
            logger.info(f"Successfully extracted single-page text using {'GPU' if use_gpu else 'CPU'}")
            return [(doc.text, 1)]

        logger.warning("No text content found in document")
        return []

    except Exception as e:
        if use_gpu:
            # If GPU processing failed, try CPU fallback
            logger.warning(f"GPU processing failed: {e}, attempting CPU fallback")
            try:
                with gpu_memory_manager.force_cpu_context():
                    result = DOCUMENT_CONVERTER.convert(file_path)
                    doc = result.document
                    
                    if hasattr(doc, 'export_to_markdown'):
                        try:
                            md = doc.export_to_markdown()
                            if md:
                                logger.info("Successfully extracted text using CPU fallback")
                                return [(md, 1)]
                        except Exception:
                            pass
                    
                    if hasattr(doc, 'pages') and getattr(doc, 'pages', None):
                        pages = [(getattr(page, 'text', '') or '', i + 1) for i, page in enumerate(doc.pages)]
                        logger.info(f"Successfully extracted {len(pages)} pages using CPU fallback")
                        return pages
                    
                    if hasattr(doc, 'text') and getattr(doc, 'text', None):
                        logger.info("Successfully extracted single-page text using CPU fallback")
                        return [(doc.text, 1)]
                        
                    return []
            except Exception as cpu_error:
                raise RuntimeError(f"Both GPU and CPU processing failed. GPU error: {e}, CPU error: {cpu_error}") from cpu_error
        else:
            raise RuntimeError(f"Docling failed to extract text from {file_path}: {e}") from e

def clean_text(raw_text: str) -> str:
    """
    Clean extracted text by removing excess whitespace and normalizing formatting.
    """
    if not raw_text:
        return ""
    text = re.sub(r'\n{3,}', '\n\n', raw_text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def get_document_info(file_path: str) -> Dict[str, Any]:
    """
    Get key information and metadata about a document file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        pages = extract_text_from_document(file_path)
        page_count = len(pages)
        
        return {
            "filename": os.path.basename(file_path),
            "file_format": Path(file_path).suffix.lower(),
            "file_size": os.path.getsize(file_path),
            "page_count": page_count,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        raise RuntimeError(f"Failed to get document info for {file_path}: {e}") from e

def get_document_metadata(file_path: str) -> Dict[str, Any]:
    """Backwards-compatible alias for get_document_info expected by some tests."""
    return get_document_info(file_path)
