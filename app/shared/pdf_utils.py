#!/usr/bin/env python3
"""
Document processing utilities for the concurrent RAG system.

This module provides utilities for extracting text and metadata from various 
document formats using a singleton instance of the Docling converter for efficiency.
"""
import re
import os
from typing import List, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# --- Module-level Singleton for Docling Converter ---
# This is more efficient than initializing on every call.
# It's configured to use the CPU for OCR to avoid multiprocessing issues with CUDA.
PDF_OPTIONS = PdfPipelineOptions(use_gpu=False)
DOCUMENT_CONVERTER = DocumentConverter(format_options={InputFormat.PDF: PDF_OPTIONS})

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

def extract_text_from_document(file_path: str) -> List[Tuple[str, int]]:
    """
    Extract text from a document file using the singleton Docling converter.
    
    Args:
        file_path: Path to the document file.
        
    Returns:
        A list of (page_text, page_number) tuples.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not is_supported_format(file_path):
        raise ValueError(f"Unsupported file format for {file_path}")

    try:
        result = DOCUMENT_CONVERTER.convert(file_path)
        doc = result.document
        
        # Use markdown export for consistent structure, with fallback to page-by-page text.
        if hasattr(doc, 'export_to_markdown') and doc.export_to_markdown():
            return [(doc.export_to_markdown(), 1)]
        
        if hasattr(doc, 'pages') and doc.pages:
            return [(page.text, i + 1) for i, page in enumerate(doc.pages) if hasattr(page, 'text')]
            
        if hasattr(doc, 'text') and doc.text:
            return [(doc.text, 1)]

        return [] # Return empty list if no text could be extracted
        
    except Exception as e:
        raise RuntimeError(f"Docling failed to extract text from {file_path}: {e}") from e

def clean_text(raw_text: str) -> str:
    """
    Clean extracted text by removing excess whitespace and normalizing formatting.
    
    Args:
        raw_text: The raw text extracted from a document.
        
    Returns:
        The cleaned text.
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
    
    Args:
        file_path: Path to the document file.
        
    Returns:
        A dictionary with document information (page count, size, etc.).
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
    """
    Backwards-compatible alias for get_document_info expected by some tests.
    """
    return get_document_info(file_path)
