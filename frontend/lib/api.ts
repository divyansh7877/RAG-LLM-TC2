/**
 * API client for the RAG system backend
 * Integrates with FastAPI endpoints
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
const USE_MOCK_API = process.env.NEXT_PUBLIC_USE_MOCK_API === 'true';

// Types based on backend models
export interface User {
  id: string;
  username?: string;
  email?: string;
  groups: string[];
  roles: string[];
}

export interface Document {
  document_id: string;
  user_id: string;
  group_id: string;
  filename: string;
  file_size: number;
  upload_date: string;
  processing_status: 'pending' | 'processing' | 'completed' | 'failed';
  page_count?: number;
  chunk_count?: number;
  file_hash?: string;
  content_type?: string;
}

export interface Job {
  job_id: string;
  user_id: string;
  job_type: 'embedding' | 'query';
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  created_at: string;
  started_at?: string;
  completed_at?: string;
  progress: number;
  result?: any;
  error?: string;
  metadata: Record<string, any>;
}

export interface QueryResult {
  query_id: string;
  query_text: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  answer?: string;
  sources?: string[];
  result_count?: number;
  processing_time?: number;
}

export interface QueryHistoryEntry {
  query_id: string;
  user_id: string;
  query_text: string;
  created_at: string;
  processing_time?: number;
  result_count?: number;
  status: string;
  response?: string;
  sources: string[];
}

// HTTP client with retry and abort support
class ApiClient {
  private baseUrl: string;
  private defaultHeaders: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  private requestCache: Map<string, Promise<any>> = new Map();

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const cacheKey = `${options.method || 'GET'}:${url}:${JSON.stringify(options.body || '')}`;
    
    // For GET requests, check if we have a pending request to avoid duplicates
    if (!options.method || options.method === 'GET') {
      if (this.requestCache.has(cacheKey)) {
        return this.requestCache.get(cacheKey);
      }
    }
    
    const requestPromise = this.performRequest<T>(url, options);
    
    // Cache GET requests only
    if (!options.method || options.method === 'GET') {
      this.requestCache.set(cacheKey, requestPromise);
      
      // Clean up cache after request completes
      requestPromise.finally(() => {
        setTimeout(() => this.requestCache.delete(cacheKey), 1000);
      });
    }
    
    return requestPromise;
  }

  private async performRequest<T>(url: string, options: RequestInit): Promise<T> {
    // Add default headers
    const headers: HeadersInit = {
      ...this.defaultHeaders,
      ...options.headers,
    };

    // Add auth header if available
    const token = this.getAuthToken();
    if (token) {
      (headers as Record<string, string>)['Authorization'] = `Bearer ${token}`;
    }

    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const errorText = await response.text();
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
      
      try {
        const errorJson = JSON.parse(errorText);
        if (errorJson.error?.message) {
          errorMessage = errorJson.error.message;
        } else if (errorJson.detail) {
          errorMessage = errorJson.detail;
        }
      } catch {
        // Use default error message
      }
      
      throw new Error(errorMessage);
    }

    return response.json();
  }

  private async requestWithRetry<T>(
    endpoint: string,
    options: RequestInit = {},
    maxRetries: number = 2
  ): Promise<T> {
    let lastError: Error | null = null;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await this.request<T>(endpoint, options);
      } catch (error) {
        lastError = error as Error;
        
        // Don't retry POST/PUT/DELETE requests
        if (options.method && options.method !== 'GET') {
          throw error;
        }
        
        // Don't retry on last attempt
        if (attempt === maxRetries) {
          throw error;
        }
        
        // Wait before retry
        await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
      }
    }
    
    throw lastError;
  }

  private getAuthToken(): string | null {
    // This will be set by the auth context
    return this.authToken;
  }

  private authToken: string | null = null;

  setAuthToken(token: string | null) {
    this.authToken = token;
  }

  // Document endpoints
  async getDocuments(params: {
    group_id?: string;
    status?: string;
    limit?: number;
    q?: string;
    offset?: number;
  } = {}): Promise<{ items: Document[]; total: number }> {
    const searchParams = new URLSearchParams();
    if (params.group_id) searchParams.set('group_id', params.group_id);
    if (params.status) searchParams.set('status', params.status);
    if (params.limit) searchParams.set('limit', params.limit.toString());
    if (params.q) searchParams.set('q', params.q);
    if (params.offset) searchParams.set('offset', params.offset.toString());

    const result = await this.requestWithRetry<{
      documents: Document[];
      total_count: number;
    }>(`/api/documents?${searchParams.toString()}`);

    return {
      items: result.documents,
      total: result.total_count,
    };
  }

  async getDocument(id: string): Promise<Document> {
    return this.requestWithRetry<Document>(`/api/documents/${id}`);
  }

  async deleteDocument(id: string): Promise<{ ok: boolean }> {
    await this.request(`/api/documents/${id}`, { method: 'DELETE' });
    return { ok: true };
  }

  async uploadDocuments(
    files: File[],
    groupId: string
  ): Promise<{ task_id: string; status: string; job_id: string }> {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    formData.append('group_id', groupId);

    const response = await this.request<{
      job_id: string;
      message: string;
      status: string;
    }>('/api/documents/upload', {
      method: 'POST',
      headers: {
        // Remove Content-Type to let browser set it with boundary
        ...Object.fromEntries(
          Object.entries(this.defaultHeaders).filter(([key]) => key !== 'Content-Type')
        ),
      },
      body: formData,
    });

    return {
      task_id: response.job_id,
      status: response.status,
      job_id: response.job_id,
    };
  }

  async ingestFromUrl(payload: {
    source_type: 'url';
    payload: string;
    dataset: string;
  }): Promise<{ task_id: string; status: string }> {
    // Your backend doesn't have this endpoint, so we'll implement it as a mock or map it
    throw new Error('URL ingestion not implemented in backend');
  }

  async ingestFromText(payload: {
    source_type: 'text';
    payload: string;
    dataset: string;
  }): Promise<{ task_id: string; status: string }> {
    // Your backend doesn't have this endpoint, so we'll implement it as a mock or map it  
    throw new Error('Text ingestion not implemented in backend');
  }

  async getIngestStatus(taskId: string): Promise<{
    status: 'queued' | 'running' | 'done' | 'error';
    message?: string;
    progress?: number;
    job_id?: string;
  }> {
    // Map to job status endpoint
    const job = await this.requestWithRetry<Job>(`/api/jobs/${taskId}`);
    
    const statusMap = {
      'pending': 'queued' as const,
      'processing': 'running' as const,
      'completed': 'done' as const,
      'failed': 'error' as const,
      'cancelled': 'error' as const,
    };

    return {
      status: statusMap[job.status],
      message: job.error || 'Processing...',
      progress: job.progress,
      job_id: job.job_id,
    };
  }

  // Job endpoints
  async getJobs(params: {
    limit?: number;
    offset?: number;
  } = {}): Promise<{ items: Job[]; total: number }> {
    const searchParams = new URLSearchParams();
    if (params.limit) searchParams.set('limit', params.limit.toString());
    if (params.offset) searchParams.set('offset', params.offset.toString());

    const result = await this.requestWithRetry<{
      jobs: Job[];
      total_count: number;
    }>(`/api/jobs?${searchParams.toString()}`);

    return {
      items: result.jobs,
      total: result.total_count,
    };
  }

  async getJob(id: string): Promise<Job> {
    return this.requestWithRetry<Job>(`/api/jobs/${id}`);
  }

  async cancelJob(id: string): Promise<{ ok: boolean }> {
    await this.request(`/api/jobs/${id}/cancel`, { method: 'POST' });
    return { ok: true };
  }

  // Query endpoints
  async submitQuery(payload: {
    query: string;
    dataset: string;
    top_k?: number;
  }): Promise<{
    results: Array<{
      id: string;
      score: number;
      text: string;
      source: string;
      doc_id?: string;
    }>;
    took_ms: number;
  }> {
    // Map to your backend's query endpoint structure
    const response = await this.request<{
      query_id: string;
      job_id: string;
      message: string;
      status: string;
    }>('/api/query', {
      method: 'POST',
      body: JSON.stringify({
        query_text: payload.query,
      }),
    });

    // Poll for results - simplified for now
    const queryResult = await this.getQuery(response.query_id);
    
    return {
      results: queryResult.sources?.map((source, index) => ({
        id: `result_${index}`,
        score: 0.8,
        text: source,
        source: source,
      })) || [],
      took_ms: (queryResult.processing_time || 0) * 1000,
    };
  }

  async getQuery(queryId: string): Promise<QueryResult> {
    return this.requestWithRetry<QueryResult>(`/api/query/${queryId}`);
  }

  // Query history endpoints
  async getQueryHistory(params: {
    limit?: number;
    offset?: number;
  } = {}): Promise<{ items: QueryHistoryEntry[]; total: number }> {
    const searchParams = new URLSearchParams();
    if (params.limit) searchParams.set('limit', params.limit.toString());
    if (params.offset) searchParams.set('offset', params.offset.toString());

    const result = await this.requestWithRetry<{
      queries: QueryHistoryEntry[];
      pagination: {
        count: number;
      };
    }>(`/api/query-history/user?${searchParams.toString()}`);

    return {
      items: result.queries,
      total: result.pagination.count,
    };
  }

  // Mock datasets for now (your backend doesn't have this endpoint)
  async getDatasets(): Promise<{ datasets: string[] }> {
    // This would map to actual groups or be implemented as a new endpoint
    return { datasets: ['default', 'common_rules', 'assistance'] };
  }
}

// Mock data for offline development
const mockApi = {
  setAuthToken(token: string | null) {
    // Mock implementation - do nothing
  },

  async getDocuments() {
    // Add delay to simulate network
    await new Promise(resolve => setTimeout(resolve, 300));
    return {
      items: [
        {
          doc_id: '1',
          filename: 'document1.pdf',
          original_name: 'My Important Document.pdf',
          file_type: 'application/pdf',
          file_size: 1024000,
          uploaded_at: new Date().toISOString(),
          status: 'completed' as const,
          chunk_count: 50,
        },
        {
          doc_id: '2',
          filename: 'document2.docx',
          original_name: 'Research Paper.docx',
          file_type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
          file_size: 512000,
          uploaded_at: new Date(Date.now() - 86400000).toISOString(),
          status: 'processing' as const,
          chunk_count: 25,
        },
      ],
      total: 2,
    };
  },

  async getDocument(id: string) {
    return {
      document_id: id,
      user_id: 'user1',
      group_id: 'default',
      filename: 'document1.pdf',
      file_size: 1024000,
      upload_date: new Date().toISOString(),
      processing_status: 'completed' as const,
      page_count: 10,
      chunk_count: 50,
    };
  },

  async deleteDocument(id: string) {
    return { ok: true };
  },

  async uploadDocuments(files: File[], groupId: string) {
    return {
      task_id: 'mock-task-' + Date.now(),
      status: 'queued' as const,
      job_id: 'mock-job-' + Date.now(),
    };
  },

  async getIngestStatus(taskId: string) {
    return {
      status: 'done' as const,
      message: 'Processing complete',
      progress: 1.0,
      job_id: taskId,
    };
  },

  async getJobs() {
    // Add delay to simulate network
    await new Promise(resolve => setTimeout(resolve, 200));
    return {
      items: [
        {
          job_id: '1',
          user_id: 'user1',
          job_type: 'embedding' as const,
          status: 'completed' as const,
          created_at: new Date(Date.now() - 300000).toISOString(),
          started_at: new Date(Date.now() - 250000).toISOString(),
          completed_at: new Date(Date.now() - 200000).toISOString(),
          progress: 1.0,
          metadata: {
            filename: 'document1.pdf',
            original_name: 'My Important Document.pdf',
            total_chunks: 50,
            processed_chunks: 50,
          },
        },
        {
          job_id: '2',
          user_id: 'user1',
          job_type: 'embedding' as const,
          status: 'processing' as const,
          created_at: new Date(Date.now() - 60000).toISOString(),
          started_at: new Date(Date.now() - 30000).toISOString(),
          progress: 0.6,
          metadata: {
            filename: 'document2.docx',
            original_name: 'Research Paper.docx',
            total_chunks: 25,
            processed_chunks: 15,
          },
        },
        {
          job_id: '3',
          user_id: 'user1',
          job_type: 'query' as const,
          status: 'failed' as const,
          created_at: new Date(Date.now() - 120000).toISOString(),
          started_at: new Date(Date.now() - 100000).toISOString(),
          completed_at: new Date(Date.now() - 90000).toISOString(),
          progress: 0.0,
          error: 'Query processing failed due to insufficient context',
          metadata: {},
        },
      ],
      total: 3,
    };
  },

  async getJob(id: string) {
    return {
      job_id: id,
      user_id: 'user1',
      job_type: 'embedding' as const,
      status: 'completed' as const,
      created_at: new Date().toISOString(),
      progress: 1.0,
      metadata: {},
    };
  },

  async cancelJob(id: string) {
    return { ok: true };
  },

  async submitQuery(payload: any) {
    return {
      results: [],
      took_ms: 100,
    };
  },

  async getQuery(queryId: string) {
    return {
      query_id: queryId,
      query_text: 'Sample query',
      status: 'completed' as const,
      created_at: new Date().toISOString(),
      answer: 'Sample answer',
      sources: ['Source 1', 'Source 2'],
      result_count: 2,
      processing_time: 1.5,
    };
  },

  async getQueryHistory() {
    // Add delay to simulate network
    await new Promise(resolve => setTimeout(resolve, 250));
    return {
      items: [
        {
          query_id: '1',
          query_text: 'What is the main topic of the research paper?',
          dataset: 'default',
          top_k: 5,
          created_at: new Date(Date.now() - 3600000).toISOString(),
          took_ms: 1200,
          result_count: 3,
          results: [
            {
              id: 'result_1',
              score: 0.92,
              text: 'The research paper focuses on artificial intelligence and machine learning applications in healthcare.',
              source: 'Research Paper.docx',
              doc_id: '2',
            },
            {
              id: 'result_2', 
              score: 0.85,
              text: 'Healthcare AI systems require careful validation and regulatory compliance.',
              source: 'Research Paper.docx',
              doc_id: '2',
            },
          ],
          answer: 'Based on the documents, the research paper discusses artificial intelligence applications in healthcare, focusing on machine learning systems and their regulatory requirements.',
        },
        {
          query_id: '2',
          query_text: 'How do I configure the system?',
          dataset: 'default',
          top_k: 3,
          created_at: new Date(Date.now() - 1800000).toISOString(),
          took_ms: 800,
          result_count: 2,
          results: [
            {
              id: 'result_3',
              score: 0.78,
              text: 'System configuration requires setting environment variables and database connections.',
              source: 'My Important Document.pdf',
              doc_id: '1',
            },
          ],
          answer: 'To configure the system, you need to set up environment variables and establish database connections as described in the documentation.',
        },
      ],
      total: 2,
    };
  },

  async getDatasets() {
    return { datasets: ['default', 'common_rules', 'assistance'] };
  },
};

// Export the API client instance
export const api = new ApiClient(API_BASE_URL);

// For development, you can swap to mock API
export default USE_MOCK_API ? mockApi : api;
