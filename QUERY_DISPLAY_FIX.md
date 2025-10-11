# Query Display Fix - Frontend/Backend Mismatch

## Issue
Query engine was generating correct results in the backend logs, but the frontend was showing "No relevant documents found". This was caused by a mismatch between the backend response format and frontend expectations.

## Root Cause

### Backend Response Format
```json
{
  "answer": "Actual LLM-generated answer with citations",
  "sources": [
    {
      "document_name": "doc.pdf",
      "page_number": 3,
      "score": 0.92,
      "content_snippet": "relevant text..."
    }
  ],
  "processing_time": 2.5,
  "status": "completed"
}
```

### Frontend Expected Format (Before Fix)
```json
{
  "results": [
    {
      "id": "result_1",
      "score": 0.8,
      "text": "source text",
      "source": "source name"
    }
  ],
  "took_ms": 2500
}
```

## Changes Made

### 1. Fixed API Client (`/frontend/lib/api.ts`)

#### Added Polling Logic (Lines 424-449)
```typescript
async pollForQueryResult(queryId: string, jobId: string, maxAttempts: number = 60): Promise<QueryResult> {
  // Poll every second for up to 60 seconds
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    try {
      const result = await this.getQuery(queryId);
      
      // Check if query is complete
      if (result.status === 'completed') {
        return result;
      } else if (result.status === 'failed') {
        throw new Error('Query processing failed');
      }
      // Continue polling if status is 'pending' or 'processing'
    } catch (error) {
      // If we get a 404, the query might not be ready yet, continue polling
      if (attempt < maxAttempts - 1) {
        continue;
      }
      throw error;
    }
  }
  
  throw new Error('Query processing timeout');
}
```

**Purpose:**
- Wait for async query processing to complete
- Poll every 1 second for up to 60 seconds
- Handle pending/processing/completed/failed states

#### Updated Response Transformation (Lines 395-420)
```typescript
// Transform sources to results format
const results = (queryResult.sources || []).map((source: any, index: number) => {
  // Handle both string sources and object sources
  if (typeof source === 'string') {
    return {
      id: `result_${index}`,
      score: 0.8,
      text: source,
      source: source,
    };
  } else {
    // Source is an object with document_name, page_number, etc.
    return {
      id: `result_${index}`,
      score: source.score || 0.8,
      text: source.content_snippet || source.text || '',
      source: source.document_name || source.source || 'Unknown',
      doc_id: source.doc_id,
    };
  }
});

return {
  results,
  took_ms: (queryResult.processing_time || 0) * 1000,
  answer: queryResult.answer, // Include the answer from the backend
};
```

**Purpose:**
- Transform backend's `sources` array to frontend's `results` array
- Handle both string and object source formats
- Extract document names, scores, and snippets correctly
- Include the actual LLM-generated answer

### 2. Updated Query Page (`/frontend/app/query/page.tsx`)

#### Use Real Answer (Lines 51-56)
```typescript
// BEFORE
if (data.results?.length > 0) {
  setAnswer(`Based on ${data.results.length} documents, here are the most relevant findings.`)
} else {
  setAnswer('No relevant documents found for your query.')
}

// AFTER
if (data.answer) {
  setAnswer(data.answer)  // Use the actual LLM answer!
} else if (data.results?.length > 0) {
  setAnswer(`Found ${data.results.length} relevant documents.`)
} else {
  setAnswer('No relevant documents found for your query.')
}
```

**Purpose:**
- Display the actual LLM-generated answer with citations
- Fallback to simple message if no answer provided
- Maintain graceful degradation

## Testing

### 1. Restart Frontend

```bash
cd /home/divyansh/Downloads/CatCapInterview/LLM\ -\ Techincal\ Case\ 2/frontend

# Kill existing process
pkill -f "next dev"

# Start fresh
npm run dev
```

### 2. Test Query Flow

1. Navigate to `http://localhost:3000/query`
2. Enter a question about your documents
3. Submit the query
4. **Expected behavior:**
   - Loading spinner appears
   - After 1-5 seconds, results appear
   - **Answer section** shows actual LLM-generated response with citations
   - **Source Documents section** shows retrieved document chunks

### 3. Verify in Browser Console

Open browser DevTools (F12) and check Console tab:
- Should see API calls to `/api/query` (submit)
- Should see polling calls to `/api/query/{query_id}` (status check)
- Should see final result with answer and sources

## Expected Output

### Successful Query
**Answer Section:**
```
Based on the provided documents, [actual detailed answer with inline citations].

(Source: document.pdf, Page: 3)
(Source: another-doc.pdf, Page: 7)

Sources Referenced:
- document.pdf
- another-doc.pdf
```

**Source Documents Section:**
```
Source Documents (3)
├─ document.pdf - Relevance: 92%
│  └─ "...relevant text excerpt from document..."
├─ another-doc.pdf - Relevance: 85%
│  └─ "...another relevant excerpt..."
└─ third-doc.pdf - Relevance: 78%
   └─ "...yet another excerpt..."
```

### No Documents Found
```
No relevant documents found for your query.
```

## Troubleshooting

### Query Timeout
**Symptom:** "Query processing timeout" error after 60 seconds

**Solutions:**
1. Check backend worker logs for errors
2. Verify documents exist in vector store
3. Check OpenAI API key is valid
4. Increase `maxAttempts` in `pollForQueryResult()` if needed

### Empty Results
**Symptom:** Query completes but shows "No relevant documents found"

**Solutions:**
1. Check user has access to documents (group_id matches)
2. Verify documents were uploaded successfully
3. Try a different query that's more specific
4. Check backend logs for retrieval count

### Answer Shows But No Sources
**Symptom:** Answer displays but "Source Documents" section is empty

**Check:**
- Backend might be returning `answer` but empty `sources` array
- This is valid if LLM couldn't find relevant chunks but still generated a response
- Look for "retrieved 0 document chunks" in backend logs

## Summary

### Before Fix
- ❌ Frontend showed "No relevant documents found"
- ❌ Didn't wait for async query processing
- ❌ Didn't transform sources correctly
- ❌ Generated fake answers instead of using LLM output

### After Fix  
- ✅ Frontend displays actual LLM-generated answers
- ✅ Polls for query completion (up to 60 seconds)
- ✅ Correctly transforms backend sources to frontend results
- ✅ Shows real document excerpts and metadata
- ✅ Displays relevance scores and source information

---

**Status:** Fixed! Restart frontend and test queries to see real results.
