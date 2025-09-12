'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  FileText, 
  ArrowLeft,
  MoreVertical,
  Download,
  Trash2,
  Eye,
  Search,
  Filter,
  Grid3x3,
  List,
  Upload,
  Calendar,
  FileType
} from 'lucide-react'
import Link from 'next/link'
import { format } from 'date-fns'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useApi } from '@/lib/authenticated-api'
import { AuthenticatedLayout } from '@/components/AuthenticatedLayout'
import { formatBytes, cn } from '@/lib/utils'

interface Document {
  document_id: string
  user_id: string
  group_id: string
  filename: string
  file_size: number
  upload_date: string
  processing_status: 'pending' | 'processing' | 'completed' | 'failed'
  page_count?: number
  chunk_count?: number
  file_hash?: string
  content_type?: string
}

const FileTypeIcon = ({ filename }: { filename: string }) => {
  const ext = filename.toLowerCase().split('.').pop() || ''
  const iconClass = "h-8 w-8"
  
  if (ext === 'pdf') return <FileType className={cn(iconClass, 'text-red-500')} />
  if (['doc', 'docx'].includes(ext)) return <FileType className={cn(iconClass, 'text-blue-500')} />
  if (['xls', 'xlsx'].includes(ext)) return <FileType className={cn(iconClass, 'text-green-500')} />
  if (['ppt', 'pptx'].includes(ext)) return <FileType className={cn(iconClass, 'text-orange-500')} />
  if (['txt', 'md'].includes(ext)) return <FileType className={cn(iconClass, 'text-gray-500')} />
  return <FileText className={cn(iconClass, 'text-gray-400')} />
}

const getStatusColor = (status: Document['processing_status']) => {
  switch (status) {
    case 'completed':
      return 'bg-green-100 text-green-800'
    case 'failed':
      return 'bg-red-100 text-red-800'
    case 'processing':
      return 'bg-blue-100 text-blue-800'
    case 'pending':
      return 'bg-yellow-100 text-yellow-800'
    default:
      return 'bg-gray-100 text-gray-800'
  }
}

export default function MyDocumentsPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedStatus, setSelectedStatus] = useState<string>('all')
  const [selectedType, setSelectedType] = useState<string>('all')
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('list')
  
  const api = useApi()
  const queryClient = useQueryClient()

  // Fetch documents
  const { data: documentsResponse, isLoading, refetch } = useQuery({
    queryKey: ['documents'],
    queryFn: () => api.getDocuments(),
    staleTime: 30000, // Consider data fresh for 30 seconds
    refetchOnWindowFocus: false, // Don't refetch on window focus
    refetchOnMount: true, // Only refetch on component mount
  })

  const documents = documentsResponse?.items || []

  // Delete document mutation
  const deleteMutation = useMutation({
    mutationFn: (docId: string) => api.deleteDocument(docId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      toast.success('Document deleted successfully')
    },
    onError: (error) => {
      toast.error('Failed to delete document', {
        description: error.message,
      })
    },
  })

  // Filter documents
  const filteredDocuments = documents.filter(doc => {
    const matchesSearch = doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
    const matchesStatus = selectedStatus === 'all' || doc.processing_status === selectedStatus
    const ext = doc.filename.toLowerCase().split('.').pop() || ''
    const matchesType = selectedType === 'all' || (
      (selectedType === 'pdf' && ext === 'pdf') ||
      (selectedType === 'doc' && ['doc', 'docx'].includes(ext)) ||
      (selectedType === 'xls' && ['xls', 'xlsx'].includes(ext)) ||
      (selectedType === 'ppt' && ['ppt', 'pptx'].includes(ext)) ||
      (selectedType === 'text' && ['txt', 'md'].includes(ext))
    )
    
    return matchesSearch && matchesStatus && matchesType
  })

  // Get unique file types for filter
  const fileTypes = Array.from(new Set(documents.map(doc => {
    const ext = doc.filename.toLowerCase().split('.').pop() || ''
    if (ext === 'pdf') return 'pdf'
    if (['doc', 'docx'].includes(ext)) return 'doc'
    if (['xls', 'xlsx'].includes(ext)) return 'xls'
    if (['ppt', 'pptx'].includes(ext)) return 'ppt'
    if (['txt', 'md'].includes(ext)) return 'text'
    return 'other'
  }))).filter(type => type !== 'other')

  const handleDelete = async (docId: string, filename: string) => {
    if (window.confirm(`Are you sure you want to delete "${filename}"?`)) {
      deleteMutation.mutate(docId)
    }
  }

  const completedCount = documents.filter(doc => doc.processing_status === 'completed').length
  const processingCount = documents.filter(doc => doc.processing_status === 'processing').length
  const failedCount = documents.filter(doc => doc.processing_status === 'failed').length

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
                  <h1 className="text-2xl font-bold text-gray-900">My Documents</h1>
                  <p className="text-gray-600">Manage your uploaded documents</p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')}
                >
                  {viewMode === 'grid' ? <List className="h-4 w-4" /> : <Grid3x3 className="h-4 w-4" />}
                </Button>
                <Button asChild>
                  <Link href="/upload">
                    <Upload className="h-4 w-4 mr-2" />
                    Upload
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center">
                <div className="bg-blue-100 rounded-full p-3">
                  <FileText className="h-6 w-6 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Total Documents</p>
                  <p className="text-2xl font-bold text-gray-900">{documents.length}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center">
                <div className="bg-green-100 rounded-full p-3">
                  <FileText className="h-6 w-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Processed</p>
                  <p className="text-2xl font-bold text-gray-900">{completedCount}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center">
                <div className="bg-yellow-100 rounded-full p-3">
                  <FileText className="h-6 w-6 text-yellow-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Processing</p>
                  <p className="text-2xl font-bold text-gray-900">{processingCount}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center">
                <div className="bg-red-100 rounded-full p-3">
                  <FileText className="h-6 w-6 text-red-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Failed</p>
                  <p className="text-2xl font-bold text-gray-900">{failedCount}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Search and Filters */}
          <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
            <div className="flex flex-col lg:flex-row gap-4">
              <div className="flex-1">
                <div className="relative">
                  <Search className="h-4 w-4 absolute left-3 top-3 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search documents..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
                  />
                </div>
              </div>

              <div className="flex gap-4">
                <select
                  value={selectedStatus}
                  onChange={(e) => setSelectedStatus(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
                >
                  <option value="all">All Status</option>
                  <option value="completed">Completed</option>
                  <option value="processing">Processing</option>
                  <option value="failed">Failed</option>
                </select>

                <select
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value)}
                  className="px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
                >
                  <option value="all">All Types</option>
                  {fileTypes.map(type => (
                    <option key={type} value={type}>
                      {type.charAt(0).toUpperCase() + type.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Documents List/Grid */}
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                <span className="ml-3 text-gray-600">Loading documents...</span>
              </div>
            ) : filteredDocuments.length === 0 ? (
              <div className="text-center py-12">
                <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  {documents.length === 0 ? 'No Documents Yet' : 'No Documents Found'}
                </h3>
                <p className="text-gray-600 mb-4">
                  {documents.length === 0 
                    ? 'Upload your first document to get started.'
                    : 'No documents match your current filters.'
                  }
                </p>
                {documents.length === 0 && (
                  <Button asChild>
                    <Link href="/upload">
                      <Upload className="h-4 w-4 mr-2" />
                      Upload Documents
                    </Link>
                  </Button>
                )}
              </div>
            ) : viewMode === 'list' ? (
              <div className="divide-y divide-gray-200">
                {filteredDocuments.map((doc, index) => (
                  <motion.div
                    key={doc.document_id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="p-6 hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4 flex-1 min-w-0">
                        <FileTypeIcon filename={doc.filename} />
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center space-x-3 mb-2">
                            <h3 className="text-lg font-medium text-gray-900 truncate">
                              {doc.filename}
                            </h3>
                            <Badge className={cn('text-xs', getStatusColor(doc.processing_status))}>
                              {doc.processing_status}
                            </Badge>
                          </div>
                          <div className="flex items-center space-x-4 text-sm text-gray-500">
                            <span>{formatBytes(doc.file_size)}</span>
                            <span>Uploaded {format(new Date(doc.upload_date), 'MMM d, yyyy')}</span>
                            {doc.chunk_count && (
                              <span>{doc.chunk_count} chunks</span>
                            )}
                          </div>
                        </div>
                      </div>

                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="sm">
                            <MoreVertical className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem>
                            <Eye className="h-4 w-4 mr-2" />
                            View Details
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <Download className="h-4 w-4 mr-2" />
                            Download
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem 
                            className="text-red-600 focus:text-red-600"
                            onClick={() => handleDelete(doc.document_id, doc.filename)}
                            disabled={deleteMutation.isPending}
                          >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 p-6">
                {filteredDocuments.map((doc, index) => (
                  <motion.div
                    key={doc.document_id}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.05 }}
                    className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
                  >
                    <div className="text-center mb-4">
                      <FileTypeIcon filename={doc.filename} />
                    </div>
                    <div className="space-y-2">
                      <h3 className="font-medium text-gray-900 text-sm truncate" title={doc.filename}>
                        {doc.filename}
                      </h3>
                      <Badge className={cn('text-xs w-full justify-center', getStatusColor(doc.processing_status))}>
                        {doc.processing_status}
                      </Badge>
                      <div className="text-xs text-gray-500 space-y-1">
                        <div>{formatBytes(doc.file_size)}</div>
                        <div>{format(new Date(doc.upload_date), 'MMM d, yyyy')}</div>
                        {doc.chunk_count && <div>{doc.chunk_count} chunks</div>}
                      </div>
                    </div>
                    <div className="mt-4">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="outline" size="sm" className="w-full">
                            <MoreVertical className="h-4 w-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem>
                            <Eye className="h-4 w-4 mr-2" />
                            View Details
                          </DropdownMenuItem>
                          <DropdownMenuItem>
                            <Download className="h-4 w-4 mr-2" />
                            Download
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem 
                            className="text-red-600 focus:text-red-600"
                            onClick={() => handleDelete(doc.document_id, doc.filename)}
                            disabled={deleteMutation.isPending}
                          >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </AuthenticatedLayout>
  )
}