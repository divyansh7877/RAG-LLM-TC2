---
inclusion: always
---

# Multi-User Private RAG System

## Product Requirements

### Core Principles
- **Data Privacy First**: All user documents must remain isolated by user_id/group_id
- **Local Processing**: No external API calls - all AI processing happens locally
- **Citation Accuracy**: Every LLM response must include document name and page number
- **Multi-Tenant Security**: Users can only access their own or explicitly shared documents

### Architecture Patterns

#### Data Isolation
- Use metadata filters in LanceDB queries: `{"user_id": user_id}` or `{"group_id": group_id}`
- Never return documents without proper user/group authorization
- Session management through Redis with JWT tokens

#### Resource Management
- Thread-safe singleton pattern for shared models (LLM, embeddings)
- Factory pattern for user-specific query engines
- Celery task queues: `embedding`, `query`, `maintenance`

#### Error Handling
- All errors flow through `app.shared.error_handling`
- User-facing errors should be informative but not expose system internals
- Log all security-related events (failed auth, unauthorized access attempts)

### Code Conventions

#### Response Format
```python
# Always include source citations in LLM responses
{
    "answer": "Your answer here",
    "sources": [
        {"document": "filename.pdf", "page": 5},
        {"document": "filename.pdf", "page": 7}
    ]
}
```

#### Authentication Flow
1. Validate JWT token from request headers
2. Extract user_id from token payload
3. Apply user_id filter to all database operations
4. Never trust client-provided user_id values

#### Document Processing
- PDF uploads go to temp storage first, then processed by embedding worker
- Chunk documents with overlap for better retrieval
- Store metadata: filename, upload_date, user_id, group_id, page_numbers

### Performance Requirements
- Support concurrent users without blocking
- Embedding tasks: max 2 concurrent per worker
- Query tasks: max 4 concurrent per worker
- Response time target: <10 seconds for typical queries