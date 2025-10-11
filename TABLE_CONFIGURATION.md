# LanceDB Table Configuration

## Current Status: ✅ ALL FILES CONFIGURED CORRECTLY

All files in the system are already configured to use **`document_embeddings_v2`** table.

## File Configuration

### 1. Query Engine (`app/shared/query_engine_factory.py`)
**Line 45:**
```python
TABLE_NAME = os.getenv("LANCEDB_TABLE_NAME", "document_embeddings_v2")
```
✅ Uses `document_embeddings_v2`

### 2. Document Processor (`app/shared/document_processor.py`)
**Line 43:**
```python
def __init__(self, db_path: str = "./multi_user_db.lance", table_name: str = "document_embeddings_v2", ...):
```
✅ Default is `document_embeddings_v2`

### 3. Embedding Worker (`app/workers/embedding_worker.py`)
**Line 63:**
```python
table_name = os.getenv("LANCEDB_TABLE_NAME", "document_embeddings_v2")
```
✅ Uses `document_embeddings_v2`

### 4. Query Worker (`app/workers/query_worker.py`)
**Line 533:**
```python
if "Table document_embeddings_v2 is not initialized" in error_str:
```
✅ References `document_embeddings_v2` in error handling

## Environment Variable (Optional)

You can override the table name with an environment variable if needed:

```bash
export LANCEDB_TABLE_NAME="document_embeddings_v2"
```

But this is **NOT required** since all defaults are already set to `document_embeddings_v2`.

## Current Database State

```
multi_user_db.lance/
├── connectivity_test.lance         (old test data)
├── document_embeddings.lance       (old documents - NOT USED)
└── document_embeddings_v2/         (will be created on first upload)
```

## What This Means

### ✅ Upload Behavior
- Any new document uploaded will be stored in `document_embeddings_v2`
- Old documents in `document_embeddings` will NOT be visible

### ✅ Query Behavior  
- Queries will search in `document_embeddings_v2`
- Old documents in `document_embeddings` will NOT be found

### ⚠️ Old Documents
- Documents you uploaded before are in `document_embeddings` (old table)
- These will NOT be accessible until you upload them again
- The system will show them in "My Documents" (Redis metadata) but queries won't find them

## Solution: Upload Fresh Documents

Since you don't want to migrate old data, simply:

1. ✅ **Configuration is already correct** - no changes needed
2. 📤 **Upload new documents** through the application
3. 🔍 **Query will work** once new documents are uploaded to `document_embeddings_v2`

## Steps to Test

### 1. Restart Workers (to apply the query worker fix)
```bash
cd /home/divyansh/Downloads/CatCapInterview/LLM\ -\ Techincal\ Case\ 2

# Stop all workers
pkill -f "celery.*worker"
pkill -f "celery.*beat"

# Wait
sleep 2

# Restart
./start_workers.sh
```

### 2. Upload a Test Document
```bash
# Via frontend
http://localhost:3000/upload
```

1. Select a group/dataset
2. Choose a file (PDF, DOCX, etc.)
3. Click Upload
4. Wait for processing to complete (check Jobs page)

### 3. Verify Table Created
```bash
ls -la multi_user_db.lance/
# Should now show: document_embeddings_v2/
```

### 4. Test Query
```bash
# Via frontend
http://localhost:3000/query
```

1. Enter a question about your document
2. Submit
3. ✅ Should work without "no documents" error

## Optional: Clean Up Old Tables

If you want to remove old unused tables to avoid confusion:

```bash
python3 << 'EOF'
import lancedb
db = lancedb.connect("./multi_user_db.lance")
print("Current tables:", db.table_names())

# Optional: Drop old tables
try:
    db.drop_table("document_embeddings")
    print("✓ Dropped document_embeddings")
except:
    print("✗ document_embeddings not found or already dropped")

try:
    db.drop_table("connectivity_test")
    print("✓ Dropped connectivity_test")
except:
    print("✗ connectivity_test not found or already dropped")

print("Remaining tables:", db.table_names())
EOF
```

## Verification Commands

### Check Current Tables
```bash
python3 -c "import lancedb; db=lancedb.connect('./multi_user_db.lance'); print('Tables:', db.table_names())"
```

### Check Document Count in New Table
```bash
python3 << 'EOF'
import lancedb
db = lancedb.connect("./multi_user_db.lance")
tables = db.table_names()

if "document_embeddings_v2" in tables:
    table = db.open_table("document_embeddings_v2")
    count = len(table.to_pandas())
    print(f"✓ document_embeddings_v2 has {count} documents")
else:
    print("⚠ document_embeddings_v2 not created yet (upload a document first)")
EOF
```

## Summary

✅ **All code is already configured to use `document_embeddings_v2`**  
✅ **No code changes needed**  
⚠️ **Old documents won't be visible until re-uploaded**  
📤 **Upload new documents to populate `document_embeddings_v2`**  
🔍 **Queries will work once new documents are uploaded**

---

**Current Status**: Ready to use - just restart workers and upload documents!
