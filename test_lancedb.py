#!/usr/bin/env python3
"""
Quick LanceDB connectivity test.

Usage:
  python scripts/test_lancedb_connectivity.py

This uses the same config path as the app and performs:
  - Connect to LanceDB at `config.LANCEDB_PATH`
  - List tables
  - Create or open a small test table
  - Insert a tiny row and read it back
"""
import os
import sys
import time


def main() -> int:
    start = time.time()

    db_path = os.path.abspath("./multi_user_db.lance")
    print(f"LanceDB path: {db_path}")

    # Ensure directory exists
    os.makedirs(db_path, exist_ok=True)

    try:
        import lancedb
        import pyarrow as pa
        from lancedb import vector
        __version = getattr(lancedb, "__version__", "unknown")
        print(f"lancedb version: {__version}")
    except Exception as exc:
        print(f"Failed to import lancedb dependencies: {exc}")
        return 2

    print("Connecting ...", flush=True)
    connect_start = time.time()
    db = lancedb.connect(db_path)
    print(f"Connected in {time.time() - connect_start:.3f}s")

    try:
        table_names = [t.name for t in db.table_names()]
    except Exception:
        # Older lancedb may return a list of strings
        table_names = list(db.table_names())
    print(f"Tables: {table_names}")

    # Prepare minimal schema for a simple insert
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", vector(2)),
    ])

    table_name = "connectivity_test"
    try:
        tbl = db.open_table(table_name)
        print("Opened existing table 'connectivity_test'")
    except Exception:
        tbl = db.create_table(table_name, schema=schema)
        print("Created table 'connectivity_test'")

    # Insert a tiny row
    row = {"id": "1", "text": "hello", "vector": [0.1, 0.2]}
    tbl.add([row])
    count = tbl.count_rows()
    print(f"Inserted row. Count now: {count}")

    # Read back
    df = tbl.search([0.1, 0.2]).limit(1).to_pandas()
    print("Query OK. Top row:")
    print(df.head(1))

    print(f"All good. Total time: {time.time() - start:.2f}s ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
