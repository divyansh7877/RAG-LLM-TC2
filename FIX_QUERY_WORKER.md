# Query Worker Fix - AttributeError

## Issue Summary

When trying to submit a query, the Celery query worker was failing with:
```
AttributeError: 'JobManager' object has no attribute 'fail_job'
```

Additionally, the query was failing because:
1. No documents have been uploaded yet (expected behavior)
2. The error handling was trying to call a non-existent method

## Root Cause

The query worker (`app/workers/query_worker.py`) was calling `job_manager.fail_job()` which doesn't exist. The correct method is `job_manager.update_job_status()` with `JobStatus.FAILED`.

## Fix Applied

### File Modified: `app/workers/query_worker.py`

**Line 523** - OpenAI quota check:
```python
# BEFORE
job_manager.fail_job(job_id, error_msg)

# AFTER
job_manager.update_job_status(job_id, JobStatus.FAILED, error=error_msg)
```

**Line 536** - No documents check:
```python
# BEFORE
job_manager.fail_job(job_id, error_msg)

# AFTER
job_manager.update_job_status(job_id, JobStatus.FAILED, error=error_msg)
```

**Line 576** - Insufficient quota check:
```python
# BEFORE
job_manager.fail_job(job_id, error_msg)

# AFTER
job_manager.update_job_status(job_id, JobStatus.FAILED, error=error_msg)
```

## How to Apply the Fix

### Option 1: Restart All Workers (Recommended)

From your project root directory:

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

### Option 2: Restart Individual Workers

If you want to restart workers one at a time:

```bash
cd /home/divyansh/Downloads/CatCapInterview/LLM\ -\ Techincal\ Case\ 2

# Restart query worker only (the one that had the error)
pkill -f "query_worker"
celery -A app.workers.celery_app worker --hostname=query_worker@%h --queues=query --concurrency=1 --pool=solo --loglevel=info --logfile=logs/workers/query_worker.log &

# Or restart all workers individually
pkill -f "embedding_worker"
celery -A app.workers.celery_app worker --hostname=embedding_worker@%h --queues=embedding --concurrency=1 --pool=threads --loglevel=info --logfile=logs/workers/embedding_worker.log &

pkill -f "maintenance_worker"
celery -A app.workers.celery_app worker --hostname=maintenance_worker@%h --queues=maintenance --concurrency=1 --loglevel=info --logfile=logs/workers/maintenance_worker.log &

pkill -f "celery.*beat"
celery -A app.workers.celery_app beat --loglevel=info --logfile=logs/workers/beat.log &
```

## Current Worker PIDs

Based on the current running processes:
- **Embedding Worker**: PID 30842
- **Query Worker**: PID 30935 (needs restart)
- **Maintenance Worker**: PID 30956, 32135
- **Beat Scheduler**: PID 30977

## Testing After Fix

### Step 1: Restart Workers

Use Option 1 above to restart all workers.

### Step 2: Upload a Document First

Before querying, you need to upload at least one document:

1. Navigate to `http://localhost:3000/upload`
2. Select a group/dataset
3. Choose a file (PDF, DOCX, etc.)
4. Click Upload
5. Wait for processing to complete (check Jobs page)

### Step 3: Test Query

Once documents are uploaded:

1. Navigate to `http://localhost:3000/query`
2. Enter a question about your documents
3. Submit the query
4. ✅ Should complete successfully without AttributeError

### Step 4: Test Query Without Documents (Optional)

To test the "no documents" error handling:

1. If you haven't uploaded any documents yet, try querying
2. You should get a friendly error: "No documents have been uploaded yet. Please upload some documents first before querying."
3. The job should fail gracefully without crashing

## Expected Behavior After Fix

### Scenario 1: Query Without Documents
**Before Fix:** Worker crashes with AttributeError  
**After Fix:** Graceful failure with helpful error message

**Error Message:**
```json
{
  "error": "No documents have been uploaded yet. Please upload some documents first before querying.",
  "job_id": "...",
  "no_documents": true,
  "help": "Upload documents using the /api/upload endpoint before querying"
}
```

### Scenario 2: Query With Documents
**Before Fix:** Would have worked if not for the AttributeError  
**After Fix:** Works correctly, returns query results

### Scenario 3: OpenAI Quota Exceeded
**Before Fix:** Worker crashes with AttributeError  
**After Fix:** Graceful failure with quota exceeded message

**Error Message:**
```json
{
  "error": "OpenAI quota exceeded. Please check billing and try again later.",
  "job_id": "...",
  "quota_exceeded": true
}
```

## Related Documentation

- **Job Manager API**: See `app/shared/job_manager.py` for available methods
  - `update_job_status(job_id, status, error=None, result=None)`
  - `update_job_progress(job_id, progress, message=None)`
  - `cancel_job(job_id, reason=None)`
  - `delete_job(job_id)`

- **Query Worker**: See `app/workers/query_worker.py` for query processing logic

## Troubleshooting

### Workers Not Restarting

If workers don't restart properly:

```bash
# Check if any workers are still running
ps aux | grep celery

# Force kill all celery processes
pkill -9 -f celery

# Check logs for errors
tail -f logs/workers/query_worker.log
tail -f logs/workers/embedding_worker.log
```

### Still Getting Errors

If you still see the AttributeError after restarting:

1. Make sure you're in the correct directory
2. Verify the fix was applied: `grep -n "fail_job" app/workers/query_worker.py`
   - Should return no results (or only comments)
3. Check that workers picked up the new code:
   ```bash
   tail -f logs/workers/query_worker.log
   ```
   - Look for "Connected to redis" or similar startup messages

### Query Still Failing

If queries fail for other reasons:

1. **Check LanceDB table**: Make sure documents were uploaded successfully
   ```bash
   ls -la multi_user_db.lance/
   ```

2. **Check OpenAI API key**: Verify your OpenAI API key is set correctly
   ```bash
   echo $OPENAI_API_KEY
   ```

3. **Check worker logs**: Look for detailed error messages
   ```bash
   tail -100 logs/workers/query_worker.log
   ```

## Summary

✅ **Fixed**: AttributeError when calling non-existent `fail_job()` method  
✅ **Improved**: Better error handling for queries without documents  
✅ **Maintained**: Proper job status tracking with JobStatus.FAILED  
⚠️  **Action Required**: Restart Celery workers to apply the fix  
📝 **Note**: Upload documents before querying to test successfully

---

**Status**: Fix applied, workers need restart  
**Priority**: High (blocks query functionality)  
**Impact**: Query worker will now handle errors gracefully without crashing
