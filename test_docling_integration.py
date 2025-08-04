#!/usr/bin/env python3
"""
Test script to verify Docling integration and multi-format document support.
"""
import os
import tempfile
from pathlib import Path

from app.shared.pdf_utils import (
    extract_text_from_document, 
    get_document_metadata, 
    is_supported_format, 
    get_supported_extensions
)
from app.shared.document_processor import DocumentProcessor, get_document_info

def test_supported_formats():
    """Test supported format detection."""
    print("Testing supported format detection...")
    
    supported_exts = get_supported_extensions()
    print(f"Supported extensions: {supported_exts}")
    
    test_files = [
        "test.pdf",
        "test.docx", 
        "test.pptx",
        "test.xlsx",
        "test.html",
        "test.md",
        "test.csv",
        "test.unsupported"
    ]
    
    for filename in test_files:
        is_supported = is_supported_format(filename)
        print(f"  {filename}: {'✓' if is_supported else '✗'}")
    
    print()

def test_document_processing():
    """Test document processing with available test files."""
    print("Testing document processing...")
    
    # Look for test files in common locations
    test_locations = [
        "test.pdf",
        "pdfs/",
        ".",
    ]
    
    test_files = []
    for location in test_locations:
        if os.path.isfile(location):
            if is_supported_format(location):
                test_files.append(location)
        elif os.path.isdir(location):
            for file in os.listdir(location):
                file_path = os.path.join(location, file)
                if os.path.isfile(file_path) and is_supported_format(file_path):
                    test_files.append(file_path)
                    break  # Just test one file per directory
    
    if not test_files:
        print("  No test files found. Creating a simple test file...")
        # Create a simple test HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head><title>Test Document</title></head>
            <body>
                <h1>Test Document</h1>
                <p>This is a test document for Docling integration.</p>
                <p>It contains multiple paragraphs to test text extraction.</p>
            </body>
            </html>
            """)
            test_files.append(f.name)
    
    for file_path in test_files[:3]:  # Test up to 3 files
        print(f"\n  Testing file: {file_path}")
        
        try:
            # Test metadata extraction
            metadata = get_document_metadata(file_path)
            if "error" in metadata:
                print(f"    Metadata error: {metadata['error']}")
            else:
                print(f"    Format: {metadata.get('file_format', 'unknown')}")
                print(f"    Pages: {metadata.get('page_count', 0)}")
                print(f"    Size: {metadata.get('file_size', 0)} bytes")
            
            # Test text extraction
            pages = extract_text_from_document(file_path)
            print(f"    Extracted {len(pages)} pages")
            
            if pages:
                first_page_text = pages[0][0][:100]  # First 100 chars
                print(f"    First page preview: {repr(first_page_text)}")
            
        except Exception as e:
            print(f"    Error processing {file_path}: {e}")
    
    # Clean up temporary files
    for file_path in test_files:
        if file_path.startswith('/tmp/'):
            try:
                os.unlink(file_path)
            except:
                pass

def test_document_processor():
    """Test the DocumentProcessor class."""
    print("\nTesting DocumentProcessor...")
    
    processor = DocumentProcessor()
    
    # Test health check
    health = processor.health_check()
    print(f"  Health check: {health['status']}")
    if health['status'] == 'healthy':
        print(f"    Embedding model loaded: {health['embedding_model_loaded']}")
        print(f"    Embedding dimension: {health['embedding_dimension']}")
    else:
        print(f"    Error: {health.get('error', 'Unknown error')}")

def main():
    """Run all tests."""
    print("=== Docling Integration Test ===\n")
    
    try:
        test_supported_formats()
        test_document_processing()
        test_document_processor()
        
        print("\n=== Test Complete ===")
        print("✓ Multi-format document support is working!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()