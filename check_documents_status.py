#!/usr/bin/env python3
"""
Script to check if documents have been uploaded to the LanceDB table.
This helps diagnose the "Table not initialized" error.
"""

import os
import sys
import lancedb

def check_lancedb_status():
    """Check the status of LanceDB tables and documents."""
    
    db_path = "./multi_user_db.lance"
    table_name = os.getenv("LANCEDB_TABLE_NAME", "document_embeddings_v2")
    
    print(f"Checking LanceDB at: {os.path.abspath(db_path)}")
    print(f"Looking for table: {table_name}")
    print("-" * 50)
    
    try:
        # Connect to the database
        db = lancedb.connect(db_path)
        
        # List all tables
        tables = db.table_names()
        print(f"Available tables: {tables}")
        
        if not tables:
            print("\n⚠️  No tables found in the database!")
            print("This means no documents have been uploaded yet.")
            print("\nSolution: Upload documents first using the /api/documents/upload endpoint")
            return False
        
        if table_name not in tables:
            print(f"\n⚠️  Table '{table_name}' not found!")
            print(f"Available tables: {tables}")
            
            # Check for the fallback table
            fallback_table = "document_embeddings"
            if fallback_table in tables:
                print(f"\n✓ Fallback table '{fallback_table}' exists")
                table = db.open_table(fallback_table)
                count = table.count_rows()
                print(f"  - Contains {count} document chunks")
                
                if count > 0:
                    # Sample a few entries
                    df = table.to_pandas()[:5]
                    print(f"  - Sample metadata columns: {df.columns.tolist()}")
                    if 'user_id' in df.columns and 'group_id' in df.columns:
                        print(f"  - Sample user_ids: {df['user_id'].unique()[:3].tolist()}")
                        print(f"  - Sample group_ids: {df['group_id'].unique()[:3].tolist()}")
            return False
        
        # Open the table and check contents
        table = db.open_table(table_name)
        count = table.count_rows()
        
        print(f"\n✓ Table '{table_name}' exists!")
        print(f"  - Contains {count} document chunks")
        
        if count == 0:
            print("\n⚠️  Table exists but is empty!")
            print("Documents may have been uploaded but processing failed.")
            return False
        
        # Get some stats about the documents
        df = table.to_pandas()
        
        if 'user_id' in df.columns:
            unique_users = df['user_id'].nunique()
            print(f"  - Unique users: {unique_users}")
            print(f"  - Sample user_ids: {df['user_id'].unique()[:3].tolist()}")
        
        if 'group_id' in df.columns:
            unique_groups = df['group_id'].nunique()
            print(f"  - Unique groups: {unique_groups}")
            print(f"  - Sample group_ids: {df['group_id'].unique()[:3].tolist()}")
        
        if 'document_name' in df.columns:
            unique_docs = df['document_name'].nunique()
            print(f"  - Unique documents: {unique_docs}")
            print(f"  - Sample documents: {df['document_name'].unique()[:3].tolist()}")
        
        print("\n✓ Documents are available for querying!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error checking database: {e}")
        print("\nThis could mean:")
        print("1. The database doesn't exist yet (no uploads have been attempted)")
        print("2. Permission issues with the database file")
        print("3. LanceDB is not properly installed")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("LanceDB Document Status Check")
    print("=" * 50)
    
    success = check_lancedb_status()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ System is ready for queries!")
    else:
        print("⚠️  System needs documents to be uploaded first!")
        print("\nTo fix this:")
        print("1. Start all services (Redis, FastAPI, Celery workers)")
        print("2. Authenticate and get a token")
        print("3. Upload documents using the /api/documents/upload endpoint")
        print("4. Wait for processing to complete")
        print("5. Then try querying again")
    
    sys.exit(0 if success else 1)