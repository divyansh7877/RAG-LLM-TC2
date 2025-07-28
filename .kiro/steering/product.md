# Product Overview

## Multi-User Private RAG System

A secure, multi-user Retrieval-Augmented Generation (RAG) system that enables users to upload private PDF documents and query them using a local Large Language Model. The system ensures data privacy by keeping all processing local and implementing strict user access controls.

### Key Features

- **Data Privacy**: Users can only access their own documents or shared group documents
- **Local-First Architecture**: All components (LLM, embeddings, vector database) run locally
- **Multi-Tenant Support**: Document isolation by user_id and group_id
- **Accurate Citations**: LLM responses include document name and page number references
- **Concurrent Processing**: Optimized for multi-user concurrent access with task queues

### Core Components

- **Frontend**: Gradio web interface with authentication
- **Backend**: FastAPI REST API with WebSocket support
- **Processing**: Celery workers for embedding and query tasks
- **Storage**: LanceDB vector database with Redis for sessions/caching
- **Models**: Local Llama-3.2-3B LLM and gte-large-en-v1.5 embeddings

### User Workflow

1. Login with credentials
2. Upload PDF documents to personal or group storage
3. Query documents through natural language interface
4. Receive answers with source citations

The system is designed for knowledge workers, research teams, and organizations requiring private document analysis with AI assistance.