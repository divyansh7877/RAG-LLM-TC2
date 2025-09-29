'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { 
  Upload, 
  File, 
  Link as LinkIcon, 
  Type, 
  X, 
  CheckCircle,
  AlertCircle,
  Loader2,
  ArrowLeft,
  Info
} from 'lucide-react'
import Link from 'next/link'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { formatBytes, getFileIcon, cn } from '@/lib/utils'
import { useApi } from '@/lib/authenticated-api'
import { useAuth } from '@/lib/auth'
import { AuthenticatedLayout } from '@/components/AuthenticatedLayout'

type TabType = 'file' | 'url' | 'text'

interface UploadedFile {
  file: File
  id: string
  preview?: string
}

export default function UploadPage() {
  const [activeTab, setActiveTab] = useState<TabType>('file')
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [dragActive, setDragActive] = useState(false)
  const [urlInput, setUrlInput] = useState('')
  const [textInput, setTextInput] = useState('')
  const [selectedDataset, setSelectedDataset] = useState('')
  
  const fileInputRef = useRef<HTMLInputElement>(null)
  const queryClient = useQueryClient()
  const api = useApi()
  const { user } = useAuth()
  
  // Helper to format dataset label (Personal vs group name)
  const getDatasetLabel = (dataset: string) => {
    if (user && dataset === user.id) {
      return 'Personal'
    }
    return dataset
  }

  // Fetch available datasets (user's groups)
  const { data: datasets } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => api.getDatasets(),
  })
  
  // Set first dataset as default when loaded
  useEffect(() => {
    if (datasets?.datasets && datasets.datasets.length > 0 && !selectedDataset) {
      setSelectedDataset(datasets.datasets[0])
    }
  }, [datasets, selectedDataset])

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: async ({ files, dataset }: { files: File[]; dataset: string }) => {
      return api.uploadDocuments(files, dataset)
    },
    onSuccess: (data) => {
      toast.success('Upload started!', {
        description: `Processing job created: ${data.job_id}`,
      })
      setFiles([])
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
    onError: (error) => {
      toast.error('Upload failed', {
        description: error.message,
      })
    },
  })

  // URL ingestion (not implemented in backend, showing as placeholder)
  const urlMutation = useMutation({
    mutationFn: async ({ url, dataset }: { url: string; dataset: string }) => {
      // This would need to be implemented in your backend
      throw new Error('URL ingestion not yet implemented')
    },
    onSuccess: () => {
      toast.success('URL processing started!')
      setUrlInput('')
    },
    onError: (error) => {
      toast.error('URL processing failed', {
        description: error.message,
      })
    },
  })

  // Text ingestion (not implemented in backend, showing as placeholder)
  const textMutation = useMutation({
    mutationFn: async ({ text, dataset }: { text: string; dataset: string }) => {
      // This would need to be implemented in your backend
      throw new Error('Text ingestion not yet implemented')
    },
    onSuccess: () => {
      toast.success('Text processing started!')
      setTextInput('')
    },
    onError: (error) => {
      toast.error('Text processing failed', {
        description: error.message,
      })
    },
  })

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files) {
      const droppedFiles = Array.from(e.dataTransfer.files)
      addFiles(droppedFiles)
    }
  }, [])

  const addFiles = (newFiles: File[]) => {
    const supportedTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/vnd.openxmlformats-officedocument.presentationml.presentation', 
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'text/html',
      'text/markdown',
      'text/csv'
    ]

    const validFiles = newFiles.filter(file => {
      const isValidType = supportedTypes.includes(file.type) || 
        /\.(pdf|docx|pptx|xlsx|html|md|csv)$/i.test(file.name)
      
      if (!isValidType) {
        toast.error(`File type not supported: ${file.name}`)
        return false
      }
      
      if (file.size > 50 * 1024 * 1024) { // 50MB limit
        toast.error(`File too large: ${file.name}`)
        return false
      }
      
      return true
    })

    const uploadedFiles: UploadedFile[] = validFiles.map(file => ({
      file,
      id: Math.random().toString(36).substr(2, 9),
    }))

    setFiles(prev => [...prev, ...uploadedFiles])
  }

  const removeFile = (id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id))
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      addFiles(Array.from(e.target.files))
    }
  }

  const handleUpload = () => {
    if (files.length === 0) {
      toast.error('Please select files to upload')
      return
    }
    
    uploadMutation.mutate({
      files: files.map(f => f.file),
      dataset: selectedDataset,
    })
  }

  const handleUrlSubmit = () => {
    if (!urlInput.trim()) {
      toast.error('Please enter a URL')
      return
    }
    
    urlMutation.mutate({
      url: urlInput,
      dataset: selectedDataset,
    })
  }

  const handleTextSubmit = () => {
    if (!textInput.trim()) {
      toast.error('Please enter text content')
      return
    }
    
    textMutation.mutate({
      text: textInput,
      dataset: selectedDataset,
    })
  }

  const tabs = [
    { id: 'file', label: 'File Upload', icon: File },
    { id: 'url', label: 'URL', icon: LinkIcon },
    { id: 'text', label: 'Text', icon: Type },
  ]

  return (
    <AuthenticatedLayout>
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
                <h1 className="text-2xl font-bold text-gray-900">Upload Documents</h1>
                <p className="text-gray-600">Process files, URLs, or text content</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Dataset Selection */}
        <div className="mb-8">
          <label htmlFor="dataset" className="block text-sm font-medium text-gray-700 mb-2">
            Dataset
          </label>
          <select
            id="dataset"
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
          >
            {datasets?.datasets.map((dataset) => (
              <option key={dataset} value={dataset}>
                {getDatasetLabel(dataset)}
              </option>
            ))}
          </select>
        </div>

        {/* Tabs */}
        <div className="mb-8">
          <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as TabType)}
                className={cn(
                  'flex-1 flex items-center justify-center px-3 py-2 text-sm font-medium rounded-md transition-colors',
                  activeTab === tab.id
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                )}
              >
                <tab.icon className="h-4 w-4 mr-2" />
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          {activeTab === 'file' && (
            <motion.div
              key="file"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Upload Zone */}
              <div
                className={cn(
                  'border-2 border-dashed rounded-lg p-8 text-center transition-colors',
                  dragActive 
                    ? 'border-primary bg-primary/5' 
                    : 'border-gray-300 hover:border-gray-400'
                )}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <Upload className={cn(
                  'h-12 w-12 mx-auto mb-4',
                  dragActive ? 'text-primary' : 'text-gray-400'
                )} />
                <p className="text-lg font-medium text-gray-900 mb-2">
                  {dragActive ? 'Drop files here' : 'Drop files or click to browse'}
                </p>
                <p className="text-gray-500 mb-4">
                  Support PDF, DOCX, PPTX, XLSX, HTML, MD, CSV
                </p>
                <Button variant="outline">
                  Choose Files
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.docx,.pptx,.xlsx,.html,.md,.csv"
                  onChange={handleFileInput}
                  className="hidden"
                />
              </div>

              {/* File List */}
              {files.length > 0 && (
                <div className="space-y-3">
                  <h3 className="text-lg font-medium text-gray-900">
                    Selected Files ({files.length})
                  </h3>
                  <div className="space-y-2">
                    {files.map((fileItem, index) => (
                      <motion.div
                        key={fileItem.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="flex items-center justify-between p-3 bg-white rounded-lg border border-gray-200"
                      >
                        <div className="flex items-center space-x-3">
                          <div className="text-2xl">
                            {getFileIcon(fileItem.file.name)}
                          </div>
                          <div>
                            <p className="font-medium text-gray-900">
                              {fileItem.file.name}
                            </p>
                            <p className="text-sm text-gray-500">
                              {formatBytes(fileItem.file.size)}
                            </p>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeFile(fileItem.id)}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </motion.div>
                    ))}
                  </div>
                  <Button
                    onClick={handleUpload}
                    disabled={uploadMutation.isPending}
                    className="w-full"
                  >
                    {uploadMutation.isPending && (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    )}
                    Upload {files.length} File{files.length !== 1 ? 's' : ''}
                  </Button>
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'url' && (
            <motion.div
              key="url"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <div className="flex items-start space-x-3 mb-4">
                  <Info className="h-5 w-5 text-blue-500 mt-0.5" />
                  <div>
                    <h3 className="font-medium text-gray-900">URL Processing</h3>
                    <p className="text-sm text-gray-600">
                      This feature is not yet implemented in the backend. It would allow you to process content from web pages.
                    </p>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <label htmlFor="url" className="block text-sm font-medium text-gray-700 mb-2">
                      Website URL
                    </label>
                    <input
                      id="url"
                      type="url"
                      value={urlInput}
                      onChange={(e) => setUrlInput(e.target.value)}
                      placeholder="https://example.com/article"
                      className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
                    />
                  </div>
                  <Button
                    onClick={handleUrlSubmit}
                    disabled={urlMutation.isPending || !urlInput.trim()}
                    className="w-full"
                  >
                    {urlMutation.isPending && (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    )}
                    Process URL
                  </Button>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'text' && (
            <motion.div
              key="text"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <div className="flex items-start space-x-3 mb-4">
                  <Info className="h-5 w-5 text-blue-500 mt-0.5" />
                  <div>
                    <h3 className="font-medium text-gray-900">Text Processing</h3>
                    <p className="text-sm text-gray-600">
                      This feature is not yet implemented in the backend. It would allow you to process raw text content.
                    </p>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <label htmlFor="text" className="block text-sm font-medium text-gray-700 mb-2">
                      Text Content
                    </label>
                    <textarea
                      id="text"
                      value={textInput}
                      onChange={(e) => setTextInput(e.target.value)}
                      rows={10}
                      placeholder="Paste your text content here..."
                      className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
                    />
                  </div>
                  <Button
                    onClick={handleTextSubmit}
                    disabled={textMutation.isPending || !textInput.trim()}
                    className="w-full"
                  >
                    {textMutation.isPending && (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    )}
                    Process Text
                  </Button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Help Section */}
        <div className="mt-8 bg-blue-50 rounded-lg p-6">
          <h3 className="font-medium text-blue-900 mb-2">💡 Ingestion Help</h3>
          <div className="text-sm text-blue-800 space-y-2">
            <p>• <strong>File Upload:</strong> Supports PDF, Word, Excel, PowerPoint, HTML, Markdown, and CSV files</p>
            <p>• <strong>File Size Limit:</strong> Maximum 50MB per file</p>
            <p>• <strong>Processing Time:</strong> Typically 30-60 seconds per file depending on size</p>
            <p>• <strong>Concurrent Processing:</strong> Multiple files are processed simultaneously</p>
            <p>• <strong>Status Monitoring:</strong> Check the Jobs tab to monitor processing progress</p>
          </div>
        </div>
      </div>
    </AuthenticatedLayout>
  )
}
