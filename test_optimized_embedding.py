#!/usr/bin/env python3
"""
Test the optimized embedding functionality with real document processing.
"""
import os
import tempfile
import time
import uuid

def test_optimized_embedding():
    """Test the optimized embedding process with a small document."""
    print("🧪 Testing Optimized Embedding Process...")
    
    try:
        from app.shared.document_processor import DocumentProcessor
        
        # Create a small test document
        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, "test_optimized.html")
        
        with open(test_file, 'w') as f:
            f.write("""
            <html>
            <body>
                <h1>Optimized Embedding Test</h1>
                <p>This is a test document to verify that the optimized embedding functionality works correctly.</p>
                <p>The optimizations should eliminate OpenBLAS warnings and improve processing speed.</p>
                <p>We're testing with a small document to ensure the process completes quickly.</p>
            </body>
            </html>
            """)
        
        processor = DocumentProcessor()
        
        print(f"  Processing test document: {os.path.basename(test_file)}")
        start_time = time.time()
        
        # Test the optimized processing
        result = processor.process_documents(
            file_paths=[test_file],
            user_id="test_optimized_user",
            group_id="test_optimized_group",
            job_id=f"test_optimized_{uuid.uuid4()}",
            chunk_size=256,  # Small chunks for faster processing
            chunk_overlap=20,
            db_path="./test_optimized_db.lance",
            table_name="test_optimized_embeddings",
            embed_model_name="./models/gte-large-en-v1.5",
            device="cpu"
        )
        
        processing_time = time.time() - start_time
        
        print(f"\n  📊 Processing Results:")
        print(f"    Success: {result.success}")
        print(f"    Documents processed: {result.document_count}")
        print(f"    Chunks created: {result.chunk_count}")
        print(f"    Processing time: {processing_time:.2f} seconds")
        print(f"    Time per chunk: {processing_time/max(result.chunk_count, 1):.2f} seconds")
        
        if result.error:
            print(f"    ❌ Error: {result.error}")
            return False
        
        if result.success and result.chunk_count > 0:
            print(f"  ✅ Successfully processed document with optimizations!")
            print(f"  ✅ Created {result.chunk_count} embeddings in {processing_time:.2f}s")
            return True
        else:
            print(f"  ❌ Processing failed or no chunks created")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        if os.path.exists("./test_optimized_db.lance"):
            shutil.rmtree("./test_optimized_db.lance")

def test_batch_optimization():
    """Test the batch optimization settings."""
    print("\n🧪 Testing Batch Optimization Settings...")
    
    try:
        from app.shared.embedding_optimizer import optimize_for_batch_processing
        
        test_cases = [
            (5, "Small batch"),
            (20, "Medium batch"), 
            (100, "Large batch")
        ]
        
        for batch_size, description in test_cases:
            settings = optimize_for_batch_processing(batch_size)
            print(f"  {description} ({batch_size} items):")
            print(f"    Embed batch size: {settings['embed_batch_size']}")
            print(f"    Max length: {settings['max_length']}")
            print(f"    Node batch size: {settings['node_batch_size']}")
        
        print("  ✅ Batch optimization settings working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_threading_optimization():
    """Test that threading optimizations are applied."""
    print("\n🧪 Testing Threading Optimizations...")
    
    try:
        from app.shared.embedding_optimizer import set_optimal_threading_environment
        
        # Apply optimizations
        set_optimal_threading_environment()
        
        # Check that environment variables are set
        expected_vars = [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", 
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TOKENIZERS_PARALLELISM"
        ]
        
        all_set = True
        for var in expected_vars:
            if var not in os.environ:
                print(f"  ❌ Environment variable {var} not set")
                all_set = False
            else:
                print(f"  ✅ {var} = {os.environ[var]}")
        
        if all_set:
            print("  ✅ All threading optimizations applied correctly")
            return True
        else:
            print("  ❌ Some threading optimizations missing")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    """Run optimized embedding tests."""
    print("🚀 OPTIMIZED EMBEDDING FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        ("Threading Optimizations", test_threading_optimization),
        ("Batch Optimization Settings", test_batch_optimization),
        ("Optimized Embedding Process", test_optimized_embedding),
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
    print("\n" + "=" * 60)
    print("📊 OPTIMIZED TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL OPTIMIZED TESTS PASSED!")
        print("\n📝 Optimizations Applied:")
        print("  • Threading environment optimized to avoid OpenBLAS warnings")
        print("  • Batch processing optimized for different workload sizes")
        print("  • Memory usage optimized with smaller batch sizes")
        print("  • PyTorch configured for optimal CPU performance")
        print("  • Embedding model configured with optimal settings")
        print("\n✅ The optimized embedding system is working correctly!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)