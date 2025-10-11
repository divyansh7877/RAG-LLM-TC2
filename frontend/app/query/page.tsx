'use client'

import { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useQuery, useMutation } from '@tanstack/react-query'
import { 
  Search, 
  ArrowLeft,
  Settings,
  Loader2,
  FileText,
  ExternalLink,
  Copy,
  RotateCcw
} from 'lucide-react'
import Link from 'next/link'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { useApi } from '@/lib/authenticated-api'
import { AuthenticatedLayout } from '@/components/AuthenticatedLayout'
import { MarkdownRenderer } from '@/lib/markdown'
import { formatDuration, copyToClipboard, cn } from '@/lib/utils'

interface QueryResult {
  id: string
  score: number
  text: string
  source: string
  doc_id?: string
}

export default function QueryPage() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [results, setResults] = useState<QueryResult[]>([])
  const [queryTime, setQueryTime] = useState<number | null>(null)
  const [answer, setAnswer] = useState<string>('')
  
  const api = useApi()
  const queryInputRef = useRef<HTMLTextAreaElement>(null)

  // Query mutation
  const queryMutation = useMutation({
    mutationFn: async ({ query, top_k }: { query: string; top_k: number }) => {
      return api.submitQuery({ query, dataset: 'all', top_k })
    },
    onSuccess: (data) => {
      setResults(data.results || [])
      setQueryTime(data.took_ms)
      // Use the actual answer from the backend
      if (data.answer) {
        setAnswer(data.answer)
      } else if (data.results?.length > 0) {
        setAnswer(`Found ${data.results.length} relevant documents.`)
      } else {
        setAnswer('No relevant documents found for your query.')
      }
      toast.success('Query completed', {
        description: `Found ${data.results?.length || 0} results in ${data.took_ms}ms`,
      })
    },
    onError: (error) => {
      toast.error('Query failed', {
        description: error.message,
      })
      setResults([])
      setAnswer('')
      setQueryTime(null)
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) {
      toast.error('Please enter a query')
      return
    }
    
    queryMutation.mutate({
      query: query.trim(),
      top_k: topK,
    })
  }

  const handleCopyResult = async (text: string) => {
    await copyToClipboard(text)
    toast.success('Copied to clipboard')
  }

  const clearResults = () => {
    setResults([])
    setAnswer('')
    setQueryTime(null)
    setQuery('')
    queryInputRef.current?.focus()
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
                  <h1 className="text-2xl font-bold text-gray-900">Query Documents</h1>
                  <p className="text-gray-600">Search and query your document collection</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Query Form */}
            <div className="lg:col-span-1">
              <div className="bg-white rounded-lg border border-gray-200 p-6 sticky top-8">
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="bg-blue-50 border border-blue-200 rounded-md p-3 mb-4">
                    <p className="text-sm text-blue-800">
                      <strong>Note:</strong> Queries search across all your groups and personal documents.
                    </p>
                  </div>

                  <div>
                    <label htmlFor="topK" className="block text-sm font-medium text-gray-700 mb-2">
                      Results Count: {topK}
                    </label>
                    <input
                      type="range"
                      id="topK"
                      min="1"
                      max="20"
                      value={topK}
                      onChange={(e) => setTopK(parseInt(e.target.value))}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>1</span>
                      <span>20</span>
                    </div>
                  </div>

                  <div>
                    <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
                      Query
                    </label>
                    <textarea
                      ref={queryInputRef}
                      id="query"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      rows={4}
                      placeholder="Enter your question about the documents..."
                      className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
                    />
                  </div>

                  <div className="flex space-x-2">
                    <Button 
                      type="submit" 
                      disabled={queryMutation.isPending || !query.trim()}
                      className="flex-1"
                    >
                      {queryMutation.isPending && (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      )}
                      <Search className="h-4 w-4 mr-2" />
                      Search
                    </Button>
                    {(results.length > 0 || answer) && (
                      <Button variant="outline" onClick={clearResults}>
                        <RotateCcw className="h-4 w-4" />
                      </Button>
                    )}
                  </div>
                </form>

                {queryTime !== null && (
                  <div className="mt-4 p-3 bg-gray-50 rounded-md">
                    <div className="text-sm text-gray-600">
                      Query completed in <span className="font-medium">{formatDuration(queryTime / 1000)}</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Results */}
            <div className="lg:col-span-2">
              <AnimatePresence mode="wait">
                {queryMutation.isPending && (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex items-center justify-center py-12"
                  >
                    <div className="text-center">
                      <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
                      <p className="text-gray-600">Searching documents...</p>
                    </div>
                  </motion.div>
                )}

                {!queryMutation.isPending && answer && (
                  <motion.div
                    key="results"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="space-y-6"
                  >
                    {/* Answer */}
                    <div className="bg-white rounded-lg border border-gray-200 p-6">
                      <div className="flex items-center justify-between mb-4">
                        <h2 className="text-lg font-semibold text-gray-900">Answer</h2>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleCopyResult(answer)}
                        >
                          <Copy className="h-4 w-4 mr-1" />
                          Copy
                        </Button>
                      </div>
                      <div className="prose prose-sm max-w-none">
                        <MarkdownRenderer content={answer} />
                      </div>
                    </div>

                    {/* Results List */}
                    {results.length > 0 && (
                      <div className="bg-white rounded-lg border border-gray-200">
                        <div className="p-6 border-b border-gray-200">
                          <h2 className="text-lg font-semibold text-gray-900">
                            Source Documents ({results.length})
                          </h2>
                        </div>
                        <div className="divide-y divide-gray-200">
                          {results.map((result, index) => (
                            <motion.div
                              key={result.id}
                              initial={{ opacity: 0, x: -20 }}
                              animate={{ opacity: 1, x: 0 }}
                              transition={{ delay: index * 0.1 }}
                              className="p-6 hover:bg-gray-50 transition-colors"
                            >
                              <div className="flex items-start justify-between mb-3">
                                <div className="flex items-center space-x-3">
                                  <div className="bg-primary/10 rounded-full p-2">
                                    <FileText className="h-4 w-4 text-primary" />
                                  </div>
                                  <div>
                                    <h3 className="font-medium text-gray-900">
                                      {result.source}
                                    </h3>
                                    <div className="flex items-center space-x-2 text-sm text-gray-500">
                                      <span>Relevance: {Math.round(result.score * 100)}%</span>
                                    </div>
                                  </div>
                                </div>
                                <div className="flex items-center space-x-2">
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => handleCopyResult(result.text)}
                                  >
                                    <Copy className="h-4 w-4" />
                                  </Button>
                                  {result.doc_id && (
                                    <Button variant="ghost" size="sm">
                                      <ExternalLink className="h-4 w-4" />
                                    </Button>
                                  )}
                                </div>
                              </div>
                              <div className="text-gray-700 text-sm leading-relaxed">
                                {result.text}
                              </div>
                            </motion.div>
                          ))}
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}

                {!queryMutation.isPending && !answer && !results.length && (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="text-center py-12"
                  >
                    <Search className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 mb-2">
                      Ready to Search
                    </h3>
                    <p className="text-gray-600 max-w-sm mx-auto">
                      Enter your question in the search box to find relevant information from your documents.
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </AuthenticatedLayout>
  )
}