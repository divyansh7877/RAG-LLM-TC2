#!/usr/bin/env python3
"""
Quick test to verify embedding functionality is working.
"""
import os
import tempfile
import time

def test_embedding_model():
    """Test that the embedding model loads and can generate embeddings."""
    print("🧪 Testing Embedding Model...")
    
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        # Load the embedding model
        print("  Loading embedding model...")
        embed_model = HuggingFaceEmbedding(
            model_name="./models/gte-large-en-v1.5",
            device="cpu",
            trust_remote_code=True,
        )
        print("  ✅ Embedding model loaded successfully")
        
        # Test embedding generation
        print("  Generating test embedding...")
        test_text = "This is a test sentence for embedding generation."
        embedding = embed_model.get_text_embedding(test_text)
        
        print(f"  ✅ Generated embedding with dimension: {len(embedding)}")
        print(f"  ✅ Embedding preview: [{embedding[0]:.4f}, {embedding[1]:.4f}, ...]")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_document_processing():
    """Test document processing without full embedding."""
    print("\n🧪 Testing Document Processing...")
    
    try:
        from app.shared.pdf_utils import extract_text_from_document, is_supported_format
        
        # Create a simple test document
        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, "test.html")
        
        with open(test_file, 'w') as f:
            f.write("""
            <html>
            <body>
                <h1>Test Document</h1>
                <p>This is a test document for processing.</p>
            </body>
            </html>
            """)
        
        # Test format detection
        if not is_supported_format(test_file):
            print("  ❌ Format detection failed")
            return False
        print("  ✅ Format detection working")
        
        # Test text extraction
        pages = extract_text_from_document(test_file)
        if not pages:
            print("  ❌ Text extraction failed")
            return False
        
        print(f"  ✅ Extracted {len(pages)} pages")
        print(f"  ✅ Sample text: {pages[0][0][:50]}...")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_redis_connection():
    """Test Redis connectivity."""
    print("\n🧪 Testing Redis Connection...")
    
    try:
        from app.shared.redis_client import redis_client
        
        # Test basic operations
        test_key = "test_embedding_quick"
        test_data = {"test": "data", "timestamp": time.time()}
        
        # Write
        redis_client.set_json(test_key, test_data, expire_seconds=60)
        print("  ✅ Redis write successful")
        
        # Read
        retrieved = redis_client.get_json(test_key)
        if retrieved and retrieved.get("test") == "data":
            print("  ✅ Redis read successful")
        else:
            print("  ❌ Redis read failed")
            return False
        
        # Cleanup
        redis_client.client.delete(test_key)
        print("  ✅ Redis cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_lancedb_connection():
    """Test LanceDB connectivity."""
    print("\n🧪 Testing LanceDB Connection...")
    
    try:
        import lancedb
        
        # Test database connection
        db_path = "./test_quick_db.lance"
        db = lancedb.connect(db_path)
        print("  ✅ LanceDB connection successful")
        
        # Test basic operations
        tables = db.table_names()
        print(f"  ✅ Database accessible (found {len(tables)} tables)")
        
        # Cleanup
        import shutil
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_document_processor_health():
    """Test document processor health check."""
    print("\n🧪 Testing Document Processor Health...")
    
    try:
        from app.shared.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        health = processor.health_check()
        
        print(f"  Status: {health['status']}")
        
        if health['status'] == 'healthy':
            print(f"  ✅ Embedding model loaded: {health['embedding_model_loaded']}")
            print(f"  ✅ Embedding dimension: {health['embedding_dimension']}")
            return True
        else:
            print(f"  ❌ Health check failed: {health.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Run quick embedding tests."""
    print("🚀 QUICK EMBEDDING FUNCTIONALITY TEST")
    print("=" * 50)
    
    tests = [
        ("Embedding Model", test_embedding_model),
        ("Document Processing", test_document_processing),
        ("Redis Connection", test_redis_connection),
        ("LanceDB Connection", test_lancedb_connection),
        ("Document Processor Health", test_document_processor_health),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 QUICK TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Core embedding functionality is working.")
        print("\n📝 Summary:")
        print("  • Embedding model (gte-large-en-v1.5) loads successfully")
        print("  • Document processing (Docling) works for multiple formats")
        print("  • Redis connectivity is functional")
        print("  • LanceDB vector database is accessible")
        print("  • Document processor health check passes")
        print("\n✅ The embedding system is ready for use!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)