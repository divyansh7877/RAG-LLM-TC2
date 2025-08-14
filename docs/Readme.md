# Project: Concurrent Multi-User Private RAG System

## 1. Objective

This project provides a secure, concurrent, multi-user, and private Retrieval-Augmented Generation (RAG) system. It allows multiple users to simultaneously upload their own private documents in multiple formats (PDF, DOCX, PPTX, XLSX, HTML, MD, CSV) and query them using a local Large Language Model (LLM). The system has been transformed from a single-threaded application into a robust, production-ready system that can handle concurrent operations safely and efficiently.

The key goals are:
- **Data Privacy:** Ensure users can only query their own documents or documents from groups they belong to, with complete isolation between users.
- **Concurrency:** Support multiple users simultaneously without data leakage or performance degradation.
- **Local First:** All components, including the LLM and vector database, run locally to prevent data from leaving the machine.
- **Accurate & Citable Answers:** The LLM is prompted to answer questions based *only* on the provided documents and to cite its sources by document name and page number.
- **Production Ready:** Includes comprehensive error handling, monitoring, job management, and real-time updates.

---

## 2. Features

- **Concurrent User Support:** Multiple users can simultaneously upload documents and query the system without data leakage or performance issues.
- **JWT-based Authentication:** Secure token-based authentication with session management.
- **Multi-Tenant Data Storage:** Documents are associated with a `user_id` and `group_id`, allowing for both personal and shared knowledge bases.
- **Background Job Processing:** Document uploads and queries are processed asynchronously using Celery workers.
- **Real-time Updates:** WebSocket connections provide real-time job status updates and progress tracking.
- **Modern Web Interface:** FastAPI-based REST API with a responsive HTML/CSS/JavaScript frontend.
- **End-to-End RAG Pipeline:**
    - **Ingestion:** Extracts text, splits it into chunks, and generates embeddings using background workers.
    - **Storage:** Stores embeddings and metadata in a LanceDB vector database with proper user isolation.
    - **Retrieval:** Fetches relevant document chunks based on the user's query and access rights with thread-safe filtering.
    - **Reranking:** Refines the retrieved results for better relevance.
    - **Generation:** Uses a local LLM to synthesize an answer from the retrieved context.
- **Job Management:** Track, monitor, and manage document processing and query jobs with detailed status reporting.
- **Resource Management:** Intelligent resource allocation and monitoring to prevent system overload.
- **Comprehensive Monitoring:** System health checks, performance metrics, and error tracking.
- **Multi-Format Document Support:** Advanced document processing using Docling for PDF, DOCX, PPTX, XLSX, HTML, MD, and CSV files with enhanced text extraction and OCR capabilities.

---

## 3. System Architecture

The system has been completely redesigned as a modern, concurrent web application with the following architecture:

### Frontend Layer
- **FastAPI Web Application** (`app/api/main.py`): REST API endpoints with WebSocket support for real-time updates
- **Modern Web Interface** (`app/static/`): Responsive HTML/CSS/JavaScript frontend replacing Gradio
- **Authentication**: JWT-based token authentication with secure session management

### Application Layer
- **Session Manager** (`app/shared/session_manager.py`): Redis-backed distributed session management
- **Job Manager** (`app/shared/job_manager.py`): Comprehensive job lifecycle management and tracking
- **Resource Manager** (`app/shared/resource_manager.py`): Intelligent resource allocation and monitoring
- **Query Engine Factory** (`app/shared/query_engine_factory.py`): Thread-safe query engine with connection pooling

### Task Processing Layer
- **Celery Workers**: Background task processing with proper resource management
  - **Embedding Workers** (`app/workers/embedding_worker.py`): Process document uploads asynchronously
  - **Query Workers** (`app/workers/query_worker.py`): Handle user queries with security isolation
  - **Maintenance Workers** (`app/workers/maintenance_worker.py`): System cleanup and monitoring
- **Redis Queue**: Message broker and result backend for task distribution

### Data Layer
- **LanceDB Vector Store**: Stores document embeddings with user isolation metadata
- **Redis**: Session storage, job tracking, and caching
- **Document Processor** (`app/shared/document_processor.py`): Multi-format document processing and embedding service
- **Document Utils** (`app/shared/pdf_utils.py`): Docling-powered text extraction utilities for multiple document formats

### Key Architectural Improvements

1. **Thread Safety**: All components are designed for concurrent access with proper locking mechanisms
2. **User Isolation**: Complete data separation between users with metadata filtering at every level
3. **Scalability**: Horizontal scaling support through Celery workers and Redis clustering
4. **Monitoring**: Comprehensive health checks, performance metrics, and error tracking
5. **Real-time Updates**: WebSocket connections for live job status and progress updates

---

## 4. Core Components

-   **LLM:** `Llama-3.2-3B-Instruct-IQ3_M.gguf` (a quantized model for efficient local inference).
-   **Embedding Model:** `Alibaba-NLP/gte-large-en-v1.5` (quantized for CPU performance).
-   **Document Processing:** `Docling` (advanced multi-format document parsing with OCR capabilities).
-   **Vector Database:** `LanceDB` (for efficient, file-based vector storage).
-   **Task Queue:** `Redis` + `Celery` (for background job processing and message brokering).
-   **Web Framework:** `FastAPI` (for REST API and WebSocket support).
-   **Core Frameworks:** `LlamaIndex` (for the RAG pipeline), `Redis` (for caching and sessions).

---

## 5. Setup and Usage

### Installation

1.  It is recommended to use a virtual environment (e.g., `conda` or `venv`).
2.  Install the required Python packages using the provided files. For conda environments:
    ```bash
    conda env create -f environment.yml
    conda activate llm_rag
    ```
    Alternatively, using pip:
    ```bash
    pip install -r requirements.txt
    ```
3.  The `run.sh` script can be used to download the necessary models, but ensure the dependencies are installed first.

### Running the Application

The concurrent system requires multiple components to be running:

1. **Start Redis server:**
   ```bash
   ./setup_redis.sh
   ```

2. **Start the FastAPI web application:**
   ```bash
   python -m app.api.main
   ```

3. **Start Celery workers (in separate terminals):**
   ```bash
   # Start all workers
   ./start_workers.sh
   
   # Or start individual worker types
   celery -A app.workers.celery_app worker --queues=embedding --concurrency=2
   celery -A app.workers.celery_app worker --queues=query --concurrency=4
   celery -A app.workers.celery_app worker --queues=maintenance --concurrency=1
   ```

4. **Access the application:**
   Open your browser to `http://localhost:8000` to access the modern web interface.

### How to Use

1.  **Login:** Use the authentication system with JWT tokens.
    -   Default credentials are defined in the authentication manager
2.  **Upload Documents:** 
    - Navigate to the "Upload Documents" tab
    - Select your documents (PDF, DOCX, PPTX, XLSX, HTML, MD, CSV) using drag-and-drop or file browser
    - Choose a destination group
    - Upload files are processed asynchronously with real-time progress updates
3.  **Query Documents:** 
    - Navigate to the "Query Documents" tab
    - Type your question and submit
    - Queries are processed in the background with real-time status updates
4.  **Monitor Jobs:**
    - Navigate to the "Job Status" tab to track all your document processing and query jobs
    - View real-time progress, completion status, and error details

---

## 6. Multi-Format Document Processing

### Docling Integration

The system has been upgraded from using PyMuPDF (fitz) to **Docling**, a state-of-the-art document processing library that provides superior text extraction and multi-format support.

### Supported Document Formats

| Format | Extension | Description | Features |
|--------|-----------|-------------|----------|
| **PDF** | `.pdf` | Portable Document Format | OCR support, table extraction, layout preservation |
| **Word** | `.docx` | Microsoft Word documents | Full text extraction, formatting preservation |
| **PowerPoint** | `.pptx` | Microsoft PowerPoint presentations | Slide content extraction, text and layout |
| **Excel** | `.xlsx`, `.xls` | Microsoft Excel spreadsheets | Cell data extraction, sheet processing |
| **HTML** | `.html` | Web documents | Tag-aware text extraction, structure preservation |
| **Markdown** | `.md` | Markdown documents | Native markdown processing, structure awareness |
| **CSV** | `.csv` | Comma-separated values | Tabular data processing, column extraction |

### Enhanced Features

- **OCR Capabilities**: Automatic text extraction from scanned documents and images
- **Table Structure Recognition**: Intelligent table detection and data extraction
- **Layout Preservation**: Maintains document structure through markdown export
- **Metadata Extraction**: Comprehensive document metadata for all formats
- **Error Handling**: Robust processing with graceful fallbacks for corrupted files
- **Visual File Management**: Format-specific icons and type identification in the UI

### Processing Pipeline

1. **Format Detection**: Automatic file type identification by extension and MIME type
2. **Document Conversion**: Docling processes the document using format-specific parsers
3. **Text Extraction**: Advanced text extraction with structure preservation
4. **Chunking**: Intelligent text splitting with overlap for better retrieval
5. **Embedding**: Vector generation using the embedding model
6. **Storage**: Secure storage in LanceDB with user isolation metadata

---

## 7. Modularity and Customization

The system is designed to be modular and easily customizable:

-   **LLM:** To use a different model, change the `GGUF_MODEL_PATH` in `app/shared/query_engine_factory.py` to point to another GGUF-compatible file.
-   **Retriever:** The retrieval logic in the query engine factory can be modified to use different LlamaIndex retrievers (e.g., hybrid search, different MMR settings).
-   **Document Formats:** The Docling-powered document processor supports multiple formats out of the box. Additional formats can be added by extending the `SUPPORTED_FORMATS` dictionary in `app/shared/pdf_utils.py` as Docling adds support for new formats.
-   **Document Processing:** The text extraction pipeline can be customized by modifying the Docling converter configuration for format-specific optimizations (OCR settings, table extraction, etc.).
-   **Workers:** Additional worker types can be added to handle different processing tasks or integrate with external services.
-   **Authentication:** The authentication system can be extended to integrate with external identity providers (LDAP, OAuth, etc.).
-   **Storage:** The system can be configured to use different vector databases or add additional storage backends.

---

## 8. Production Deployment

### Docker Support
The system includes Docker containerization for easy deployment:

```bash
# Build and run with docker-compose
docker-compose up -d
```

### Monitoring and Observability
- **Health Checks:** Comprehensive health check endpoints for all system components
- **Metrics Collection:** Performance metrics for queries, embeddings, and system resources
- **Error Tracking:** Centralized error handling with detailed logging and alerting
- **Real-time Monitoring:** WebSocket-based real-time system status updates

### Security Considerations
- **User Isolation:** Complete data separation with metadata filtering at every level
- **Session Security:** JWT tokens with configurable expiration and refresh mechanisms
- **Rate Limiting:** Configurable rate limits to prevent abuse
- **Input Validation:** Comprehensive input validation and sanitization
- **Audit Logging:** Detailed audit trails for all user actions and system events

### Performance Optimization
- **Connection Pooling:** Efficient database connection management
- **Query Caching:** Redis-based caching for frequently accessed queries
- **Resource Management:** Intelligent resource allocation and monitoring
- **Batch Processing:** Optimized batch processing for document uploads
- **Horizontal Scaling:** Support for multiple worker instances and Redis clustering