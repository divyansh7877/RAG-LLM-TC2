#!/usr/bin/env python3
"""
Document processing utilities for the concurrent RAG system.

This module provides utilities for extracting text from various document formats
using Docling for improved document parsing and multi-format support.
"""
import re
import os
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Supported file extensions and their corresponding InputFormat
SUPPORTED_FORMATS = {
    '.pdf': InputFormat.PDF,
    '.docx': InputFormat.DOCX,
    '.pptx': InputFormat.PPTX,
    '.xlsx': InputFormat.XLSX,
    '.xls': InputFormat.XLSX,  # Treat .xls as .xlsx
    '.html': InputFormat.HTML,
    '.md': InputFormat.MD,
    '.csv': InputFormat.CSV,
}

def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions."""
    return list(SUPPORTED_FORMATS.keys())

def is_supported_format(file_path: str) -> bool:
    """Check if file format is supported."""
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_FORMATS

def extract_text_from_document(file_path: str) -> List[Tuple[str, int]]:
    """
    Extract text from a document file using Docling.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of (page_text, page_number) tuples (1-indexed)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Initialize converter with default options
    converter = DocumentConverter()
    
    try:
        # Convert document
        result = converter.convert(file_path)
        
        # Extract text by pages
        pages = []
        doc = result.document
        
        # Use the export_to_markdown method for consistent text extraction
        try:
            # Try to get markdown export which handles structure well
            doc_text = result.document.export_to_markdown()
            pages.append((doc_text, 1))
        except AttributeError:
            # Fallback to manual text extraction
            if hasattr(doc, 'pages') and doc.pages:
                # Document has page structure
                for i, page in enumerate(doc.pages):
                    page_text = ""
                    # Try different ways to extract text from page
                    if hasattr(page, 'export_to_markdown'):
                        page_text = page.export_to_markdown()
                    elif hasattr(page, 'text') and page.text:
                        page_text = page.text
                    elif hasattr(page, 'body') and hasattr(page.body, 'text'):
                        page_text = page.body.text
                    pages.append((page_text, i + 1))
            else:
                # Document doesn't have page structure, treat as single page
                doc_text = ""
                if hasattr(doc, 'export_to_markdown'):
                    doc_text = doc.export_to_markdown()
                elif hasattr(doc, 'text') and doc.text:
                    doc_text = doc.text
                elif hasattr(doc, 'body') and hasattr(doc.body, 'text'):
                    doc_text = doc.body.text
                
                pages.append((doc_text, 1))
        
        return pages
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from {file_path}: {str(e)}")

# Maintain backward compatibility
def extract_text_from_pdf(pdf_path: str) -> List[Tuple[str, int]]:
    """
    Extract text from a PDF file (backward compatibility function).
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of (page_text, page_number) tuples (1-indexed)
    """
    return extract_text_from_document(pdf_path)

def clean_text(raw_text: str) -> str:
    """
    Clean extracted text by removing excess whitespace and normalizing formatting.
    
    Args:
        raw_text: Raw text extracted from PDF
        
    Returns:
        Cleaned text
    """
    if not raw_text:
        return ""
    
    # Collapse multiple newlines into double newlines
    text = re.sub(r"\n{3,}", "\n\n", raw_text)
    
    # Strip whitespace from each line
    text = "\n".join(line.strip() for line in text.split("\n"))
    
    # Collapse multiple spaces into single spaces
    text = re.sub(r" {2,}", " ", text)
    
    return text.strip()

def get_document_metadata(file_path: str) -> dict:
    """
    Extract metadata from a document file using Docling.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Dictionary containing document metadata
    """
    try:
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        
        ext = Path(file_path).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            return {"error": f"Unsupported file format: {ext}"}
        
        # Initialize converter with default options
        converter = DocumentConverter()
        
        # Convert document
        result = converter.convert(file_path)
        doc = result.document
        
        # Extract basic metadata
        metadata = {
            "filename": os.path.basename(file_path),
            "file_format": ext,
            "file_size": os.path.getsize(file_path),
        }
        
        # Extract document-specific metadata
        if hasattr(doc, 'pages') and doc.pages:
            metadata["page_count"] = len(doc.pages)
        else:
            metadata["page_count"] = 1
        
        # Extract document properties if available
        if hasattr(doc, 'meta') and doc.meta:
            doc_meta = doc.meta
            if hasattr(doc_meta, 'title') and doc_meta.title:
                metadata["title"] = doc_meta.title
            if hasattr(doc_meta, 'author') and doc_meta.author:
                metadata["author"] = doc_meta.author
            if hasattr(doc_meta, 'subject') and doc_meta.subject:
                metadata["subject"] = doc_meta.subject
            if hasattr(doc_meta, 'creator') and doc_meta.creator:
                metadata["creator"] = doc_meta.creator
            if hasattr(doc_meta, 'producer') and doc_meta.producer:
                metadata["producer"] = doc_meta.producer
            if hasattr(doc_meta, 'creation_date') and doc_meta.creation_date:
                metadata["creation_date"] = str(doc_meta.creation_date)
            if hasattr(doc_meta, 'modification_date') and doc_meta.modification_date:
                metadata["modification_date"] = str(doc_meta.modification_date)
        
        # Add processing timestamp
        from datetime import datetime
        metadata["processed_at"] = datetime.utcnow().isoformat() + "Z"
        
        return metadata
        
    except Exception as e:
        return {"error": f"Failed to extract metadata: {str(e)}"}

# Maintain backward compatibility
def get_pdf_metadata(pdf_path: str) -> dict:
    """
    Extract metadata from a PDF file (backward compatibility function).
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing PDF metadata
    """
    return get_document_metadata(pdf_path)