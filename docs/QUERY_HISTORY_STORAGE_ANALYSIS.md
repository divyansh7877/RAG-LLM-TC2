# Query History Storage Architecture Analysis

## Current System Architecture

The RAG system currently uses:
- **Redis**: Session management, job tracking, caching, message brokering
- **LanceDB**: Vector embeddings storage (persistent)
- **File System**: Document storage, model storage
- **No RDBMS**: No PostgreSQL, MySQL, or SQLite

## Storage Options for Query History

### Option 1: Redis (Current Implementation) ✅ **RECOMMENDED**

**Pros:**
- ✅ **Architectural Consistency**: Fits existing Redis-centric architecture
- ✅ **Performance**: Sub-millisecond retrieval times
- ✅ **TTL Support**: Built-in expiration handling (30-day default)
- ✅ **Operational Simplicity**: No additional database to maintain
- ✅ **Native Data Structures**: Sorted sets, hashes, lists for complex queries
- ✅ **Memory Efficiency**: Can implement compression for large responses
- ✅ **Horizontal Scaling**: Redis clustering support
- ✅ **Real-time Access**: Perfect for job status page integration

**Cons:**
- ⚠️ **Memory Usage**: Stores everything in RAM (but with TTL this is manageable)
- ⚠️ **Durability**: Depends on Redis persistence configuration
- ⚠️ **Cost**: More expensive per GB than disk-based storage

**Redis Storage Strategy:**
```
query_history:{query_id} -> Compressed JSON with metadata
user_queries:{user_id} -> Sorted set (timestamp -> query_id)
job_query:{job_id} -> Simple hash linking job to query
query_content:{query_id} -> Large content (compressed if >1KB)
```

**Memory Usage Estimate:**
- Average query: ~2KB (metadata + small response)
- With compression: ~1KB per query
- 1000 queries/user × 100 users = ~100MB
- 30-day retention: ~3GB total (very reasonable)

### Option 2: PostgreSQL/MySQL

**Pros:**
- ✅ **ACID Transactions**: Full consistency guarantees
- ✅ **Rich Queries**: Complex SQL operations
- ✅ **Storage Cost**: Cheaper per GB than Redis
- ✅ **Mature Ecosystem**: Well-understood operational patterns

**Cons:**
- ❌ **Architectural Disruption**: Adds new database dependency
- ❌ **Operational Overhead**: Additional DB to maintain, backup, monitor
- ❌ **Performance**: Disk I/O latency for query retrieval
- ❌ **Complexity**: Connection pooling, migrations, schema management
- ❌ **Development Time**: New ORM, models, migration system needed

### Option 3: SQLite

**Pros:**
- ✅ **Simplicity**: File-based, no server needed
- ✅ **ACID Transactions**: Full consistency
- ✅ **No Network**: Local file access

**Cons:**
- ❌ **Concurrency Issues**: Poor concurrent write performance
- ❌ **Scaling Problems**: Can't distribute across workers
- ❌ **File Locking**: Potential deadlocks with multiple Celery workers
- ❌ **Backup Complexity**: File-level backup coordination needed

### Option 4: Hybrid Approach

**Strategy**: Redis for recent queries (7 days) + PostgreSQL for archives

**Pros:**
- ✅ **Best of Both**: Fast access + long-term storage
- ✅ **Cost Optimization**: Hot data in Redis, cold data on disk

**Cons:**
- ❌ **Complexity**: Dual storage systems to manage
- ❌ **Migration Logic**: Complex data lifecycle management
- ❌ **Operational Overhead**: Two databases to maintain

## **Final Recommendation: Enhanced Redis Implementation**

### Why Redis Wins:

1. **Fits Existing Architecture**: The system is already Redis-native
2. **Performance Requirements**: Job status page needs instant query history access
3. **Operational Simplicity**: No new infrastructure components
4. **Scale Characteristics**: 30-day retention with reasonable user base fits well in memory
5. **Development Speed**: Leverages existing Redis client and patterns

### Enhanced Redis Implementation Features:

```python
# Efficient storage with compression
query_history_manager = QueryHistoryManager()

# Features:
- Automatic compression for responses >1KB
- Sorted sets for time-ordered retrieval
- Efficient pagination
- User-specific indices
- Job-query linking
- Configurable TTL (30 days default)
- Memory monitoring and cleanup
- Optional backup export to JSON files
```

### Scaling Path:

1. **Current**: Single Redis instance (sufficient for MVP)
2. **Growth**: Redis clustering when memory becomes constraint
3. **Future**: Consider PostgreSQL archive tier if long-term retention needed

### Configuration Options:

```bash
# Environment variables for tuning
QUERY_HISTORY_RETENTION_DAYS=30        # TTL for queries
MAX_QUERIES_PER_USER=1000              # Per-user limits
REDIS_MAXMEMORY_POLICY=allkeys-lru     # Memory management
ENABLE_QUERY_COMPRESSION=true          # Compress large responses
QUERY_BACKUP_ENABLED=false             # Optional JSON export
```

### Memory Management Strategy:

1. **TTL-based expiration**: Automatic cleanup after 30 days
2. **Compression**: Large responses compressed with gzip
3. **Efficient indexing**: Sorted sets instead of large lists
4. **Monitoring**: Track memory usage and query patterns
5. **Fallback**: LRU eviction if memory pressure occurs

## Implementation Status

✅ **Basic Redis storage** - Implemented
✅ **Job-query linking** - Implemented  
✅ **User query indexing** - Implemented
✅ **TTL configuration** - Implemented
🔄 **Compression optimization** - In progress
🔄 **API endpoints** - Planned
🔄 **Frontend integration** - Planned

The Redis-based approach provides the best balance of performance, simplicity, and architectural consistency for this use case.
