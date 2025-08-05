#!/usr/bin/env python3
"""
Comprehensive test script to verify embedding functionality is working correctly.
"""
import os
import tempfile
import time
import uuid
from pathlib import Path

from app.shared.document_processor import DocumentProcessor, get_document_info, estimate_processing_time
from app.workers.embedding_worker import process_document_embedding
from app.shared.redis_client import redis_client

def create_test_documents():
    """Create test documents in various formats."""
    temp_dir = tempfile.mkdtemp()
    test_files = []
    
    # Create HTML test document
    html_file = os.path.join(temp_dir, "test_document.html")
    with open(html_file, 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head><title>Test Document for Embedding</title></head>
        <body>
            <h1>Test Document for Embedding</h1>
            <p>This is a test document to verify that the embedding functionality is working correctly.</p>
            <p>The document contains multiple paragraphs with different content to test text chunking and embedding generation.</p>
            <p>We should be able to extract meaningful embeddings from this content that can be used for similarity search and retrieval.</p>
            <h2>Technical Details</h2>
            <p>The embedding system uses the gte-large-en-v1.5 model to generate 1024-dimensional vectors.</p>
            <p>These vectors are stored in LanceDB for efficient similarity search and retrieval.</p>
        </body>
        </html>
        """)
    test_files.append(html_file)
    
    # Create Markdown test document
    md_file = os.path.join(temp_dir, "test_document.md")
    with open(md_file, 'w') as f:
        f.write("""
# Test Document for Embedding

This is a **markdown** test document to verify embedding functionality.

## Features to Test

- Text extraction from markdown
- Proper chunking of content
- Embedding generation
- Storage in vector database

## Technical Requirements

The system should:
1. Extract clean text from markdown
2. Split into appropriate chunks
3. Generate embeddings using gte-large-en-v1.5
4. Store with proper user/group isolation

### Code Example

```python
def test_embedding():
    return "This code should be embedded too"
```

This document tests various markdown features including headers, lists, code blocks, and formatting.
        """)
    test_files.append(md_file)
    
    return temp_dir, test_files

def test_document_info():
    """Test document information extraction."""
    print("=== Testing Document Information Extraction ===")
    
    temp_dir, test_files = create_test_documents()
    
    try:
        for file_path in test_files:
            print(f"\nTesting file: {os.path.basename(file_path)}")
            
            # Test document info
            doc_info = get_document_info(file_path)
            
            if "error" in doc_info:
                print(f"  ❌ Error: {doc_info['error']}")
            else:
                print(f"  ✓ Format: {doc_info['file_format']}")
                print(f"  ✓ Size: {doc_info['file_size']} bytes")
                print(f"  ✓ Pages: {doc_info['page_count']}")
                print(f"  ✓ Filename: {doc_info['filename']}")
        
        # Test processing time estimation
        print(f"\n--- Processing Time Estimation ---")
        estimate = estimate_processing_time(test_files)
        
        if "error" in estimate:
            print(f"  ❌ Error: {estimate['error']}")
        else:
            print(f"  ✓ Estimated time: {estimate['estimated_time_seconds']} seconds")
            print(f"  ✓ Total pages: {estimate['total_pages']}")
            print(f"  ✓ Total size: {estimate['total_size_mb']} MB")
            print(f"  ✓ Valid files: {estimate['valid_files']}")
            print(f"  ✓ Format distribution: {estimate['format_distribution']}")
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

def test_document_processor_health():
    """Test document processor health check."""
    print("\n=== Testing Document Processor Health ===")
    
    processor = DocumentProcessor()
    health = processor.health_check()
    
    print(f"Status: {health['status']}")
    
    if health['status'] == 'healthy':
        print(f"  ✓ Embedding model loaded: {health['embedding_model_loaded']}")
        print(f"  ✓ Embedding dimension: {health['embedding_dimension']}")
        print(f"  ✓ Timestamp: {health['timestamp']}")
        return True
    else:
        print(f"  ❌ Error: {health.get('error', 'Unknown error')}")
        return False

def test_document_processing():
    """Test actual document processing and embedding generation."""
    print("\n=== Testing Document Processing and Embedding ===")
    
    temp_dir, test_files = create_test_documents()
    
    try:
        processor = DocumentProcessor()
        
        # Test processing
        print(f"Processing {len(test_files)} test documents...")
        
        result = processor.process_documents(
            file_paths=test_files,
            user_id="test_user_embedding",
            group_id="test_group_embedding",
            job_id=f"test_job_{uuid.uuid4()}",
            chunk_size=256,  # Smaller chunks for testing
            chunk_overlap=20,
            db_path="./test_embedding_db.lance",
            table_name="test_embeddings",
            embed_model_name="./models/gte-large-en-v1.5",
            device="cpu"
        )
        
        print(f"\nProcessing Result:")
        print(f"  Success: {result.success}")
        print(f"  Document count: {result.document_count}")
        print(f"  Chunk count: {result.chunk_count}")
        print(f"  Processing time: {result.processing_time:.2f} seconds")
        
        if result.error:
            print(f"  ❌ Error: {result.error}")
            return False
        else:
            print(f"  ✓ Successfully processed {result.document_count} documents into {result.chunk_count} chunks")
            return True
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        # Cleanup test database
        if os.path.exists("./test_embedding_db.lance"):
            shutil.rmtree("./test_embedding_db.lance")

def test_redis_connectivity():
    """Test Redis connectivity for job tracking."""
    print("\n=== Testing Redis Connectivity ===")
    
    try:
        # Test basic Redis operations
        test_key = f"test_embedding_{uuid.uuid4()}"
        test_data = {"test": "embedding_functionality", "timestamp": time.time()}
        
        # Set data
        redis_client.set_json(test_key, test_data, expire_seconds=60)
        print("  ✓ Successfully wrote to Redis")
        
        # Get data
        retrieved_data = redis_client.get_json(test_key)
        if retrieved_data and retrieved_data.get("test") == "embedding_functionality":
            print("  ✓ Successfully read from Redis")
        else:
            print("  ❌ Failed to read correct data from Redis")
            return False
        
        # Cleanup
        redis_client.client.delete(test_key)
        print("  ✓ Successfully cleaned up Redis test data")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Redis connectivity error: {e}")
        return False

def test_embedding_worker_integration():
    """Test the embedding worker integration (without actually running Celery)."""
    print("\n=== Testing Embedding Worker Integration ===")
    
    temp_dir, test_files = create_test_documents()
    
    try:
        # Test the worker function directly (not through Celery)
        from app.workers.embedding_worker import (
            calculate_file_hash, 
            store_document_metadata,
            get_embedding_job_status
        )
        
        # Test file hash calculation
        print("Testing file hash calculation...")
        for file_path in test_files:
            file_hash = calculate_file_hash(file_path)
            print(f"  ✓ Hash for {os.path.basename(file_path)}: {file_hash[:16]}...")
        
        # Test document metadata storage
        print("Testing document metadata storage...")
        if test_redis_connectivity():
            doc_id = store_document_metadata(
                user_id="test_user",
                group_id="test_group", 
                file_path=test_files[0],
                page_count=1,
                chunk_count=5
            )
            print(f"  ✓ Stored document metadata with ID: {doc_id}")
            
            # Cleanup
            redis_client.client.delete(f"document:test_user:test_group:{doc_id}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Worker integration error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

def test_vector_database_connectivity():
    """Test LanceDB connectivity and operations."""
    print("\n=== Testing Vector Database Connectivity ===")
    
    try:
        import lancedb
        
        # Test database connection
        db_path = "./test_vector_db.lance"
        db = lancedb.connect(db_path)
        print("  ✓ Successfully connected to LanceDB")
        
        # Test table creation (basic test)
        try:
            # Try to list tables (this will work even if no tables exist)
            tables = db.table_names()
            print(f"  ✓ Successfully accessed database (found {len(tables)} existing tables)")
        except Exception as e:
            print(f"  ⚠ Warning: Could not list tables: {e}")
        
        # Cleanup
        import shutil
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Vector database error: {e}")
        return False

def main():
    """Run all embedding functionality tests."""
    print("🧪 EMBEDDING FUNCTIONALITY TEST SUITE")
    print("=" * 50)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Document Info Extraction", test_document_info),
        ("Document Processor Health", test_document_processor_health),
        ("Redis Connectivity", test_redis_connectivity),
        ("Vector Database Connectivity", test_vector_database_connectivity),
        ("Embedding Worker Integration", test_embedding_worker_integration),
        ("Document Processing & Embedding", test_document_processing),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Embedding functionality is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)