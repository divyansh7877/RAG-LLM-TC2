# Query Debugging Enhancement

## Issue
After uploading documents successfully, queries were failing with "No documents have been uploaded yet" error. Need better visibility into what's happening.

## Changes Made

### File Modified: `/app/workers/query_worker.py`

#### 1. Added Document Count Check (Lines 514-528)

**Before query execution, check how many documents exist:**

```python
# Verify documents exist in the vector store
try:
    from ..shared.lancedb_client import get_db_connection
    db = get_db_connection()
    table_name = os.getenv("LANCEDB_TABLE_NAME", "document_embeddings")
    if table_name in db.table_names():
        table = db.open_table(table_name)
        doc_count = len(table.to_pandas())
        logger.info(f"Vector store contains {doc_count} total document chunks in table '{table_name}'")
        if doc_count == 0:
            logger.warning(f"Table '{table_name}' exists but is empty!")
    else:
        logger.warning(f"Table '{table_name}' does not exist in database")
except Exception as check_error:
    logger.warning(f"Could not verify document count: {check_error}")
```

**Purpose:**
- Verify table exists
- Count total documents in vector store
- Log warnings if table is empty or missing

####  (Lines 531-536)

**After query execution, log retrieval results:**

```python
# Log retrieval results for debugging
if hasattr(response, 'source_nodes'):
    retrieved_count = len(response.source_nodes)
    logger.info(f"Query {query_id} retrieved {retrieved_count} document chunks")
else:
    logger.warning(f"Query {query_id} response has no source_nodes attribute")
```

**Purpose:**
- Show how many documents were actually retrieved
- Detect if query returned empty results
- Distinguish between "no documents exist" vs "no relevant documents found"

## What You'll See in Logs

### Successful Query (Expected Output)
```
[INFO] Vector store contains 45 total document chunks in table 'document_embeddings'
[INFO] Query 54315eed... retrieved 6 document chunks
[INFO] Query 54315eed... completed in 2.5s
```

### Empty Table (Problem Detection)
```
[WARNING] Table 'document_embeddings' exists but is empty!
[ERROR] Query 54315eed... failed: No documents have been uploaded yet
```

### Table Not Found (Problem Detection)
```
[WARNING] Table 'document_embeddings' does not exist in database
[ERROR] Query 54315eed... failed: No documents have been uploaded yet
```

### No Relevant Documents Found (Different Issue)
```
[INFO] Vector store contains 45 total document chunks in table 'document_embeddings'
[INFO] Query 54315eed... retrieved 0 document chunks
[INFO] Response: I couldn't find relevant information...
```

## Testing the Enhancement

### 1. Restart Query Worker

```bash
cd /home/divyansh/Downloads/CatCapInterview/LLM\ -\ Techincal\ Case\ 2

# Restart query worker
pkill -f "query_worker"
celery -A app.workers.celery_app worker --hostname=query_worker@%h --queues=query --concurrency=1 --pool=solo --loglevel=info --logfile=logs/workers/query_worker.log &
```

### 2. Watch Worker Logs

```bash
tail -f logs/workers/query_worker.log
```

### 3. Try a Query

Navigate to `http://localhost:3000/query` and submit a query.

### 4. Analyze the Output

Look for these key log lines:
1. **"Vector store contains X total document chunks"** - Shows total documents
2. **"Query ... retrieved Y document chunks"** - Shows what was retrieved
3. Any **warnings** about empty or missing tables

## Diagnostic Scenarios

### Scenario 1: Documents Exist But Query Fails

**Symptoms:**
```
[INFO] Vector store contains 45 total document chunks
[ERROR] Query failed: No documents have been uploaded yet
```

**Diagnosis:** Vector store cache issue or stale connection
**Solution:** Invalidate vector store cache

### Scenario 2: Table Exists But Empty

**Symptoms:**
```
[WARNING] Table 'document_embeddings' exists but is empty!
```

**Diagnosis:** Documents not properly embedded or wrong table
**Solution:** Check embedding worker logs, verify upload succeeded

### Scenario 3: Documents Exist But Zero Retrieved

**Symptoms:**
```
[INFO] Vector store contains 45 total document chunks
[INFO] Query retrieved 0 document chunks
```

**Diagnosis:** 
- User doesn't have access to documents (security filters)
- Query not semantically similar to any documents
**Solution:** Check user_id and group_id filters

## Additional Debugging Commands

### Check Documents in Table Directly
```bash
cd /home/divyansh/Downloads/CatCapInterview/LLM\ -\ Techincal\ Case\ 2
conda activate llm

python3 << 'EOF'
import lancedb
db = lancedb.connect("./multi_user_db.lance")

# List tables
print("Tables:", db.table_names())

# Check document_embeddings table
if "document_embeddings" in db.table_names():
    table = db.open_table("document_embeddings")
    df = table.to_pandas()
    print(f"\nTotal chunks: {len(df)}")
    
    if len(df) > 0:
        print(f"Columns: {list(df.columns)}")
        if 'user_id' in df.columns:
            print(f"Unique users: {df['user_id'].nunique()}")
            print(f"Users: {df['user_id'].unique()}")
        if 'group_id' in df.columns:
            print(f"Unique groups: {df['group_id'].nunique()}")
            print(f"Groups: {df['group_id'].unique()}")
        if 'doc_id' in df.columns:
            print(f"Unique documents: {df['doc_id'].nunique()}")
EOF
```

### Monitor Embedding Worker
```bash
tail -f logs/workers/embedding_worker.log | grep -E "(Successfully|Error|Failed|completed)"
```

### Check Vector Store Invalidation
```bash
grep -i "invalidat" logs/workers/embedding_worker.log
```

## Expected Fix

After restarting the query worker with these changes, you should see:

1. ✅ **Document count logged** before each query
2. ✅ **Retrieved chunk count** after each query
3. ✅ **Clear indication** if table is empty or missing
4. ✅ **Better error messages** with context

This will help identify whether the problem is:
- No documents in database (upload issue)
- Documents exist but not retrieved (security filter issue)
- Documents exist and retrieved but query failed (LLM issue)

## Next Steps

After reviewing the enhanced logs:

1. Check if documents actually exist in the table
2. Verify user_id and group_id match uploaded documents
3. Confirm vector store invalidation is working
4. Check for any security filter mismatches

---

**Status:** Debugging enhancement applied. Restart query worker and run a test query to see detailed logging.
