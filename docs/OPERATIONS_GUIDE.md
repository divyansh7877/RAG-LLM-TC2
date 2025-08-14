# Operations and Setup Guide

## Prerequisites
- Python 3.10+
- CUDA-capable GPU + drivers (for GPU acceleration)
- Redis running locally (`setup_redis.sh`) or via container
- Model files in `./models/gte-large-en-v1.5` (use `download_model.py` if provided)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate rag-llm
```

## Configuration
- Environment variables (examples):
  - `REDIS_URL=redis://localhost:6379/0`
  - `WORKER_CONCURRENCY=2`
- GPU/CPU selection is automatic; can be overridden in code if needed.

## Running Services
- API server (example):
```bash
./start_dev.sh
```
- Celery workers:
```bash
# Embedding / ingestion worker
celery -A app.workers.embedding_worker worker -Q embeddings -n embedding_worker@%h --loglevel=INFO

# Query worker
celery -A app.workers.query_worker worker -Q query -n query_worker@%h --loglevel=INFO

# Maintenance worker (optional)
celery -A app.workers.maintenance_worker worker -Q maintenance -n maintenance_worker@%h --loglevel=INFO
```

## Monitoring
- Logs are in `logs/` (ignored by git).
- Flower (optional):
```bash
flower -A app.workers.embedding_worker --port=5555
```

## Data Locations
- LanceDB dataset: `multi_user_db.lance/` (ignored)
- Uploaded PDFs: `pdfs/` (ignored)
- Temporary uploads: `temp_uploads/` (ignored)

## Common Tasks
- Reset LanceDB index safely by removing specific tables/directories within `multi_user_db.lance/` when the app is stopped.
- Re-run embeddings by submitting documents again; dedup can be keyed by file hash.

## Performance Tuning
- Increase worker concurrency cautiously; watch GPU memory.
- Adjust embedding batch size via `optimize_for_batch_processing` in `document_processor`.
- Ensure model path is local for low latency.

## Troubleshooting
- First-run warning: LanceDB table will be created on first insert (expected).
- Heartbeat drift warnings: Ensure NTP/chrony is running; heavy GPU tasks may cause transient drift.
- Out-of-memory: Reduce batch size; close other GPU apps.
- Docling issues: Keep it on GPU for performance; if stability issues arise, switch to CPU temporarily.

## Backups and Recovery
- Back up `multi_user_db.lance/` periodically.
- Keep `models/` and `requirements.txt` under version control or artifact management (models are ignored by git here).

## Testing
```bash
pytest -q
```
See `docs/TESTING_GUIDE.md` and `docs/TEST_SUITE_SUMMARY.md` for details.
