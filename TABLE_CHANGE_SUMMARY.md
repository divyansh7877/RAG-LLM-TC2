# Table Configuration Changed to `document_embeddings`

## ✅ Changes Applied

All files have been updated to use the **`document_embeddings`** table (where your existing data is stored).

### Files Modified:

1. **`app/shared/query_engine_factory.py`** (Line 45)
   ```python
   # BEFORE
   TABLE_NAME = os.getenv("LANCEDB_TABLE_NAME", "document_embeddings_v2")
   
   # AFTER
   TABLE_NAME = os.getenv("LANCEDB_TABLE_NAME", "document_embeddings")
   ```

2. **`app/shared/document_processor.py`** (Line 43)
   ```python
   # BEFORE
   def __init__(self, ..., table_name: str = "document_embeddings_v2", ...):
   
   # AFTER
   def __init__(self, ..., table_name: str = "document_embeddings", ...):
   ```

3. **`app/workers/embedding_worker.py`** (Line 63)
   ```python
   # BEFORE
   table_name = os.getenv("LANCEDB_TABLE_NAME", "document_embeddings_v2")
   
   # AFTER
   table_name = os.getenv("LANCEDB_TABLE_NAME", "document_embeddings")
   ```

4. **`app/workers/query_worker.py`** (Line 533)
   ```python
   # BEFORE
   if "Table document_embeddings_v2 is not initialized" in error_str:
   
   # AFTER
   if "Table document_embeddings is not initialized" in error_str:
   ```

## Current Database State

```
multi_user_db.lance/
├── connectivity_test.lance       (old test data - can be deleted)
└── document_embeddings.lance     (YOUR DATA - NOW ACTIVE! ✓)
```

## What This Means

✅ **Your existing documents** in `document_embeddings` will now be accessible  
✅ **New uploads** will go to `document_embeddings`  
✅ **Queries** will search in `document_embeddings`  
✅ **Everything uses ONE table** - consistent and simple!

## Next Steps

### 1. Restart Celery Workers

**IMPORTANT:** You must restart the workers to apply these changes!

```bash
cd /home/divyansh/Downloads/CatCapInterview/LLM\ -\ Techincal\ Case\ 2

# Stop all workers
pkill -f "celery.*worker"
pkill -f "celery.*beat"

# Wait a moment
sleep 2

# Restart all workers
./start_workers.sh
```

Or restart manually:
```bash
# Query worker
pkill -f "query_worker"
celery -A app.workers.celery_app worker --hostname=query_worker@%h --queues=query --concurrency=1 --pool=solo --loglevel=info --logfile=logs/workers/query_worker.log &

# Embedding worker
pkill -f "embedding_worker"
celery -A app.workers.celery_app worker --hostname=embedding_worker@%h --queues=embedding --concurrency=1 --pool=threads --loglevel=info --logfile=logs/workers/embedding_worker.log &

# Maintenance worker
pkill -f "maintenance_worker"
celery -A app.workers.celery_app worker --hostname=maintenance_worker@%h --queues=maintenance --concurrency=1 --loglevel=info --logfile=logs/workers/maintenance_worker.log &

# Beat scheduler
pkill -f "celery.*beat"
celery -A app.workers.celery_app beat --loglevel=info --logfile=logs/workers/beat.log &
```

### 2. Test Query Functionality

After restarting workers:

1. **Navigate to Query page:**
   ```
   http://localhost:3000/query
   ```

2. **Enter a question** about your documents

3. **Submit query**

4. ✅ **Should now work!** Your existing documents will be searched

### 3. Verify Documents Are Accessible

Check how many documents are in the table:
```bash
cd /home/divyansh/Downloads/CatCapInterview/LLM\ -\ Techincal\ Case\ 2

# Activate conda environment and check
conda activate llm
python3 << 'EOF'
import lancedb
db = lancedb.connect("./multi_user_db.lance")
if "document_embeddings" in db.table_names():
    table = db.open_table("document_embeddings")
    count = len(table.to_pandas())
    print(f"✓ document_embeddings has {count} documents")
else:
    print("✗ Table not found")
EOF
```

## Optional: Clean Up Old Test Table

You can delete the unused test table:
```bash
conda activate llm
python3 << 'EOF'
import lancedb
db = lancedb.connect("./multi_user_db.lance")
try:
    db.drop_table("connectivity_test")
    print("✓ Dropped connectivity_test table")
except:
    print("Table not found or already dropped")
print("Remaining tables:", db.table_names())
EOF
```

## Troubleshooting

### Query Still Fails?

1. **Check worker logs:**
   ```bash
   tail -f logs/workers/query_worker.log
   ```

2. **Verify workers restarted:**
   ```bash
   ps aux | grep celery
   ```

3. **Check table exists:**
   ```bash
   ls -la multi_user_db.lance/document_embeddings.lance/
   ```

### Upload Fails?

1. **Check embedding worker logs:**
   ```bash
   tail -f logs/workers/embedding_worker.log
   ```

2. **Verify table is writable:**
   ```bash
   ls -la multi_user_db.lance/
   ```

## Summary

✅ **Configuration**: All files now use `document_embeddings`  
✅ **Your data**: Preserved and will be accessible  
✅ **Consistency**: One table for everything  
⚠️ **Action required**: Restart Celery workers  
🎯 **Expected result**: Queries should work with your existing documents

---

**Status**: Configuration updated, workers need restart to apply changes.
