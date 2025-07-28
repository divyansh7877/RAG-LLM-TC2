# Technology Stack

## Core Technologies

### Backend Framework
- **FastAPI**: Modern async web framework for REST API endpoints
- **Gradio**: Web UI framework for the main user interface
- **Celery**: Distributed task queue for background processing
- **Redis**: In-memory store for sessions, caching, and message broker

### AI/ML Stack
- **LlamaIndex**: RAG pipeline framework for document processing and querying
- **LlamaCPP**: Local LLM inference engine for Llama-3.2-3B model
- **Sentence Transformers**: Embedding model (gte-large-en-v1.5) for document vectorization
- **PyTorch**: ML framework backend

### Data Storage
- **LanceDB**: Vector database for document embeddings and metadata
- **File System**: Local storage for uploaded PDFs and model files

### Development Tools
- **Python 3.12**: Primary programming language
- **Conda**: Environment management
- **pytest**: Testing framework
- **uvicorn**: ASGI server for FastAPI

## Common Commands

### Environment Setup
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate expert-rag

# Install dependencies
pip install -r requirements.txt
```

### Development Workflow
```bash
# Setup Redis and infrastructure
./setup_redis.sh
./start_dev.sh

# Run tests
python -m pytest tests/
python test_setup.py  # Infrastructure tests

# Start services (in separate terminals)
python -m app.api.main                    # FastAPI server
python app/main.py                        # Gradio interface
./start_workers.sh                        # Celery workers
```

### Worker Management
```bash
# Start all workers
celery -A app.workers.celery_app worker --loglevel=info

# Start specific queues
celery -A app.workers.celery_app worker --queues=embedding --concurrency=2
celery -A app.workers.celery_app worker --queues=query --concurrency=4
celery -A app.workers.celery_app worker --queues=maintenance --concurrency=1

# Monitor workers
celery -A app.workers.celery_app inspect active
celery -A app.workers.celery_app flower  # Web monitoring
```

### Model Management
```bash
# Download models
python download_model.py

# Model locations
./models/Llama-3.2-3B-Instruct-IQ3_M.gguf     # LLM
./models/gte-large-en-v1.5/                   # Embeddings
```

## Build System

The project uses a simple Python-based build system with:
- **requirements.txt**: Python dependencies
- **environment.yml**: Conda environment specification
- **Shell scripts**: Service management and setup automation
- **No complex build tools**: Direct Python execution for simplicity

## Performance Considerations

- **CPU Optimization**: Models quantized for CPU inference
- **Memory Management**: Configurable worker concurrency limits
- **Caching**: Redis for session and query result caching
- **Resource Monitoring**: Built-in system metrics and alerting