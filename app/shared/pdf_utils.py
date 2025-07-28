#!/usr/bin/env python3
"""
PDF processing utilities for the concurrent RAG system.

This module provides utilities for extracting text from PDF files
and cleaning the extracted text.
"""
import re
from typing import List, Tuple

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> List[Tuple[str, int]]:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of (page_text, page_number) tuples (1-indexed)
    """
    doc = fitz.open(pdf_path)
    pages = []
    
    try:
        for i in range(len(doc)):
            page_text = doc.load_page(i).get_text()
            pages.append((page_text, i + 1))
    finally:
        doc.close()
    
    return pages

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

def get_pdf_metadata(pdf_path: str) -> dict:
    """
    Extract metadata from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing PDF metadata
    """
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        page_count = len(doc)
        doc.close()
        
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "creator": metadata.get("creator", ""),
            "producer": metadata.get("producer", ""),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
            "page_count": page_count
        }
    except Exception as e:
        return {"error": f"Failed to extract metadata: {str(e)}"}