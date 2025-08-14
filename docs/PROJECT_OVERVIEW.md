# Project Overview

This project implements a production-ready RAG system with GPU-accelerated document ingestion and efficient vector search.

## Architecture
- API and Backend: FastAPI
- Workers: Celery (separate queues for ingestion, query, maintenance)
- Vector Store: LanceDB
- Embeddings: HuggingFace (gte-large-en-v1.5) via LlamaIndex
- Text Extraction: Docling (GPU-enabled)
- Broker/Cache/State: Redis
- Monitoring: Structured logging; optional Flower

## High-level Flow
1. Upload: Files received by the API and placed in `temp_uploads/`.
2. Extraction: Docling converts documents to text/markdown.
3. Chunking: Content split into nodes with metadata (user, group, file hash, page).
4. Embedding: Batches of nodes embedded on GPU; model is a singleton per worker for performance.
5. Storage: Vectors persisted to LanceDB table `document_embeddings`.
6. Query: Retrieval over LanceDB + response synthesis.

## Key Components
- `app/shared/pdf_utils.py`: Docling-based extraction for PDF and other supported formats.
- `app/shared/embedding_optimizer.py`: Embedding model singleton, device auto-detection, CUDA/CPU optimizations.
- `app/shared/document_processor.py`: Orchestrates extract → chunk → embed → store with batching.
- `app/workers/embedding_worker.py`: Celery task with retry-aware error handling and progress updates.

## GPU Usage
- Embeddings: Runs on CUDA when available; falls back to CPU.
- Docling: Uses `cuda:0` when available for faster conversion and OCR.

## Data Model (LanceDB)
- Table: `document_embeddings`
- Fields (typical): `id`, `text`, `embedding`, `user_id`, `group_id`, `file_hash`, `file_name`, `page_number`, timestamps.

## Configuration
- Models directory: `./models/gte-large-en-v1.5` (local path for offline use)
- Device selection: Auto-resolves via config; optional override in `DocumentProcessor`.
- Environment: See `docs/SETUP.md` and `requirements.txt`.

## Performance Considerations
- Embedding singleton avoids repeated model loads.
- Batched insertion prevents OOM and improves throughput.
- TF32 enabled on Ampere+ GPUs when available.
- Use Redis and Celery prefetch tuning for concurrent workloads.

## Reliability and Ops
- First insert auto-creates LanceDB table.
- On task errors, worker differentiates between retry vs. final failure and cleans up accordingly.
- Logs emitted with user/job context for observability.

## Directory Highlights
- `app/`: Source code (API, workers, shared utilities)
- `models/`: Local model files (ignored by git)
- `multi_user_db.lance/`: LanceDB dataset (ignored by git)
- `docs/`: All documentation
