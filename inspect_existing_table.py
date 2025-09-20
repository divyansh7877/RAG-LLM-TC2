#!/usr/bin/env python3
"""
Inspect the existing document_embeddings table to understand its structure.
"""

import os
import json
import lancedb
import pandas as pd

def inspect_table():
    """Inspect the existing table structure and data."""
    
    db_path = "./multi_user_db.lance"
    
    try:
        db = lancedb.connect(db_path)
        
        # Open the existing table
        table = db.open_table("document_embeddings")
        
        print("=" * 60)
        print("Table: document_embeddings")
        print("=" * 60)
        
        # Get row count
        count = table.count_rows()
        print(f"\nTotal rows: {count}")
        
        # Get a sample of the data
        df = table.to_pandas()
        
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nData types:")
        for col in df.columns:
            if col != 'vector':  # Skip vector column as it's large
                print(f"  - {col}: {df[col].dtype}")
        
        # Check metadata structure
        if 'metadata' in df.columns:
            print(f"\nMetadata structure (first non-null entry):")
            for idx, meta in enumerate(df['metadata']):
                if meta:
                    try:
                        if isinstance(meta, str):
                            meta_dict = json.loads(meta)
                        else:
                            meta_dict = meta
                        print(f"  Keys: {list(meta_dict.keys())}")
                        print(f"  Sample: {json.dumps(meta_dict, indent=2)[:500]}...")
                        break
                    except:
                        print(f"  Raw value: {str(meta)[:200]}...")
                        break
        
        # Show sample texts
        print(f"\nSample text entries (first 3):")
        for i, text in enumerate(df['text'].head(3)):
            print(f"\n  [{i+1}] {text[:150]}...")
        
        # Check for user/group info in metadata
        print("\nChecking for user/group information in metadata...")
        has_user_info = False
        if 'metadata' in df.columns:
            for meta in df['metadata']:
                if meta:
                    try:
                        if isinstance(meta, str):
                            meta_dict = json.loads(meta)
                        else:
                            meta_dict = meta
                        if 'user_id' in meta_dict or 'group_id' in meta_dict:
                            has_user_info = True
                            print(f"  Found user_id: {meta_dict.get('user_id', 'N/A')}")
                            print(f"  Found group_id: {meta_dict.get('group_id', 'N/A')}")
                            break
                    except:
                        pass
        
        if not has_user_info:
            print("  ⚠️  No user_id or group_id found in metadata!")
            print("  These documents were likely uploaded before the multi-user system was implemented.")
        
        return True
        
    except Exception as e:
        print(f"Error inspecting table: {e}")
        return False

if __name__ == "__main__":
    inspect_table()