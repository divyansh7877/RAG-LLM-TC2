'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { 
  Search, 
  ArrowLeft,
  Clock,
  FileText,
  Filter,
  RotateCcw,
  Copy,
  ExternalLink,
  Calendar
} from 'lucide-react'
import Link from 'next/link'
import { format, formatDistanceToNow } from 'date-fns'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useApi } from '@/lib/authenticated-api'
import { AuthenticatedLayout } from '@/components/AuthenticatedLayout'
import { MarkdownRenderer } from '@/lib/markdown'
import { formatDuration, copyToClipboard, cn, timeAgo } from '@/lib/utils'

interface QueryHistoryItem {
  query_id: string
  user_id: string
  query_text: string
  created_at: string
  processing_time?: number
  result_count?: number
  status: string
  response?: string
  sources: string[]
  // Extended fields for display
  took_ms?: number
  results?: Array<{
    id: string
    score: number
    text: string
    source: string
    doc_id?: string
  }>
  answer?: string
}

export default function QueryHistoryPage() {
  const [selectedDataset, setSelectedDataset] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState('')
  const [expandedQuery, setExpandedQuery] = useState<string | null>(null)
  
  const api = useApi()

  // Fetch query history
  const { data: queryHistoryResponse, isLoading, refetch } = useQuery({
    queryKey: ['query-history'],
    queryFn: () => api.getQueryHistory({ limit: 50 }),
    staleTime: 30000, // 30 seconds
  })

  const allQueries = queryHistoryResponse?.items || []

  // Fetch available datasets for filter
  const { data: datasets } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => api.getDatasets(),
  })

  // Filter queries by search only (dataset filtering not supported by backend)
  const filteredQueries = allQueries.filter(query => {
    const matchesSearch = query.query_text.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesSearch
  })

  const handleCopyQuery = async (text: string) => {
    await copyToClipboard(text)
    toast.success('Query copied to clipboard')
  }

  const handleCopyResults = async (results: QueryHistoryItem['results']) => {
    if (!results || results.length === 0) return
    const text = results.map(result => `${result.source}:\n${result.text}`).join('\n\n')
    await copyToClipboard(text)
    toast.success('Results copied to clipboard')
  }

  const handleRerunQuery = (query: QueryHistoryItem) => {
    const params = new URLSearchParams({
      q: query.query_text
    })
    window.open(`/query?${params}`, '_blank')
  }

  const toggleExpanded = (queryId: string) => {
    setExpandedQuery(expandedQuery === queryId ? null : queryId)
  }

  return (
    <AuthenticatedLayout>
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <div className="bg-white border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between py-4">
              <div className="flex items-center space-x-4">
                <Button variant="ghost" size="sm" asChild>
                  <Link href="/">
                    <ArrowLeft className="h-4 w-4 mr-2" />
                    Back
                  </Link>
                </Button>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">Query History</h1>
                  <p className="text-gray-600">View your past queries and results</p>
                </div>
              </div>
              <Button onClick={() => refetch()} variant="outline" size="sm">
                <RotateCcw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
            </div>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Filters */}
          <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="flex-1">
                <div className="relative">
                  <Search className="h-4 w-4 absolute left-3 top-3 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search query history..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
                  />
                </div>
              </div>

            </div>
          </div>

          {/* Query History List */}
          <div className="space-y-4">
            {isLoading ? (
              <div className="bg-white rounded-lg border border-gray-200 p-12">
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                  <span className="ml-3 text-gray-600">Loading query history...</span>
                </div>
              </div>
            ) : filteredQueries.length === 0 ? (
              <div className="bg-white rounded-lg border border-gray-200 p-12">
                <div className="text-center">
                  <Clock className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    {allQueries.length === 0 ? 'No Query History Yet' : 'No Queries Found'}
                  </h3>
                  <p className="text-gray-600 mb-4">
                    {allQueries.length === 0
                      ? 'Your query history will appear here after you submit your first query.'
                      : 'No queries match your current search and filters.'
                    }
                  </p>
                  {allQueries.length === 0 && (
                    <Button asChild>
                      <Link href="/query">
                        <Search className="h-4 w-4 mr-2" />
                        Start Querying
                      </Link>
                    </Button>
                  )}
                </div>
              </div>
            ) : (
              filteredQueries.map((query, index) => (
                <motion.div
                  key={query.query_id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="bg-white rounded-lg border border-gray-200 overflow-hidden"
                >
                  <div className="p-6">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1 min-w-0">
                          <div className="flex items-center space-x-3 mb-2">
                            <h3 className="text-lg font-medium text-gray-900 line-clamp-2">
                              {query.query_text}
                            </h3>
                          </div>
                        
                        <div className="flex items-center space-x-4 text-sm text-gray-500 mb-3">
                          <span className="flex items-center">
                            <Calendar className="h-4 w-4 mr-1" />
                            {timeAgo(query.created_at)}
                          </span>
                          {query.processing_time && (
                            <span className="flex items-center">
                              <Clock className="h-4 w-4 mr-1" />
                              {formatDuration(query.processing_time)}
                            </span>
                          )}
                          {query.result_count !== undefined && (
                            <span className="flex items-center">
                              <FileText className="h-4 w-4 mr-1" />
                              {query.result_count} results
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center space-x-2 ml-4">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleCopyQuery(query.query_text)}
                        >
                          <Copy className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleRerunQuery(query)}
                        >
                          <RotateCcw className="h-4 w-4 mr-1" />
                          Rerun
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => toggleExpanded(query.query_id)}
                        >
                          {expandedQuery === query.query_id ? 'Collapse' : 'Expand'}
                        </Button>
                      </div>
                    </div>

                    {/* Answer preview (if available) */}
                    {query.response && (
                      <div className="bg-blue-50 border border-blue-200 rounded-md p-4 mb-4">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="text-sm font-medium text-blue-900">Answer</h4>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => copyToClipboard(query.response!)}
                          >
                            <Copy className="h-4 w-4" />
                          </Button>
                        </div>
                        <div className="text-sm text-blue-800 line-clamp-3">
                          <MarkdownRenderer content={query.response} />
                        </div>
                      </div>
                    )}

                    {/* Source documents preview */}
                    {query.sources && query.sources.length > 0 && (
                      <div className="border border-gray-200 rounded-md">
                        <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
                          <div className="flex items-center justify-between">
                            <h4 className="text-sm font-medium text-gray-900">
                              Source Documents ({query.sources.length})
                            </h4>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => copyToClipboard(query.sources.join('\n'))}
                            >
                              <Copy className="h-4 w-4 mr-1" />
                              Copy All
                            </Button>
                          </div>
                        </div>
                        
                        <div className="divide-y divide-gray-200">
                          {query.sources.slice(0, expandedQuery === query.query_id ? undefined : 3).map((source, index) => (
                            <div key={index} className="p-4">
                              <div className="flex items-start justify-between mb-2">
                                <div className="flex items-center space-x-2">
                                  <span className="text-sm font-medium text-gray-900">
                                    {source}
                                  </span>
                                </div>
                                <div className="flex items-center space-x-1">
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => copyToClipboard(source)}
                                  >
                                    <Copy className="h-4 w-4" />
                                  </Button>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>

                        {query.sources && query.sources.length > 3 && expandedQuery !== query.query_id && (
                          <div className="px-4 py-3 border-t border-gray-200 bg-gray-50">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => toggleExpanded(query.query_id)}
                              className="text-primary hover:text-primary-dark"
                            >
                              Show {query.sources.length - 3} more sources
                            </Button>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </motion.div>
              ))
            )}
          </div>

          {/* Load more button (if needed) */}
          {filteredQueries.length >= 50 && (
            <div className="text-center mt-8">
              <Button variant="outline" onClick={() => refetch()}>
                Load More
              </Button>
            </div>
          )}
        </div>
      </div>
    </AuthenticatedLayout>
  )
}