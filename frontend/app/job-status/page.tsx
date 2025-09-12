'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { 
  Clock, 
  CheckCircle, 
  XCircle, 
  Loader2, 
  ArrowLeft,
  RefreshCw,
  FileText,
  Database,
  Settings,
  Trash2,
  AlertTriangle
} from 'lucide-react'
import Link from 'next/link'
import { format } from 'date-fns'

import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useApi } from '@/lib/authenticated-api'
import { AuthenticatedLayout } from '@/components/AuthenticatedLayout'
import { formatDuration, cn } from '@/lib/utils'

interface Job {
  job_id: string
  job_type: string
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled'
  progress: number
  created_at: string
  started_at?: string
  completed_at?: string
  error_message?: string
  result?: any
  metadata?: {
    filename?: string
    group_id?: string
    original_name?: string
    total_chunks?: number
    processed_chunks?: number
  }
}

const JobTypeIcon = ({ type }: { type: string }) => {
  switch (type) {
    case 'embedding':
      return <Database className="h-4 w-4" />
    case 'query':
      return <FileText className="h-4 w-4" />
    case 'maintenance':
      return <Settings className="h-4 w-4" />
    default:
      return <Clock className="h-4 w-4" />
  }
}

const StatusIcon = ({ status }: { status: Job['status'] }) => {
  switch (status) {
    case 'completed':
      return <CheckCircle className="h-4 w-4 text-green-500" />
    case 'failed':
    case 'cancelled':
      return <XCircle className="h-4 w-4 text-red-500" />
    case 'processing':
      return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
    case 'pending':
      return <Clock className="h-4 w-4 text-yellow-500" />
    default:
      return <Clock className="h-4 w-4 text-gray-500" />
  }
}

const getStatusColor = (status: Job['status']) => {
  switch (status) {
    case 'completed':
      return 'bg-green-100 text-green-800'
    case 'failed':
    case 'cancelled':
      return 'bg-red-100 text-red-800'
    case 'processing':
      return 'bg-blue-100 text-blue-800'
    case 'pending':
      return 'bg-yellow-100 text-yellow-800'
    default:
      return 'bg-gray-100 text-gray-800'
  }
}

export default function JobStatusPage() {
  const [selectedStatus, setSelectedStatus] = useState<string>('all')
  const [selectedType, setSelectedType] = useState<string>('all')
  
  const api = useApi()

  // Fetch jobs with manual refresh only
  const { data: jobsResponse, refetch, isLoading } = useQuery({
    queryKey: ['jobs'],
    queryFn: () => api.getJobs({ limit: 100 }),
    staleTime: 30000, // Consider data fresh for 30 seconds
    refetchOnWindowFocus: false, // Don't refetch on window focus
    refetchOnMount: true, // Only refetch on component mount
  })

  const jobs = jobsResponse?.items || []

  // Filter jobs by status and type
  const filteredJobs = jobs.filter(job => {
    const matchesStatus = selectedStatus === 'all' || job.status === selectedStatus
    const matchesType = selectedType === 'all' || job.job_type === selectedType
    return matchesStatus && matchesType
  })

  const activeJobs = filteredJobs.filter(job => 
    job.status === 'pending' || job.status === 'processing'
  ).length

  const completedJobs = filteredJobs.filter(job => 
    job.status === 'completed'
  ).length

  const failedJobs = filteredJobs.filter(job => 
    job.status === 'failed'
  ).length

  const getJobDuration = (job: Job) => {
    const start = job.started_at ? new Date(job.started_at) : new Date(job.created_at)
    const end = job.completed_at ? new Date(job.completed_at) : new Date()
    return Math.floor((end.getTime() - start.getTime()) / 1000)
  }

  const getProgressText = (job: Job) => {
    if (job.metadata?.total_chunks && job.metadata?.processed_chunks) {
      return `${job.metadata.processed_chunks}/${job.metadata.total_chunks} chunks`
    }
    return `${Math.round(job.progress * 100)}%`
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
                  <h1 className="text-2xl font-bold text-gray-900">Job Status</h1>
                  <p className="text-gray-600">Monitor your background processing jobs</p>
                </div>
              </div>
              <Button onClick={() => refetch()} variant="outline" size="sm">
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
            </div>
          </div>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center">
                <div className="bg-blue-100 rounded-full p-3">
                  <Loader2 className="h-6 w-6 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Active Jobs</p>
                  <p className="text-2xl font-bold text-gray-900">{activeJobs}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center">
                <div className="bg-green-100 rounded-full p-3">
                  <CheckCircle className="h-6 w-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Completed</p>
                  <p className="text-2xl font-bold text-gray-900">{completedJobs}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center">
                <div className="bg-red-100 rounded-full p-3">
                  <XCircle className="h-6 w-6 text-red-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Failed</p>
                  <p className="text-2xl font-bold text-gray-900">{failedJobs}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center">
                <div className="bg-gray-100 rounded-full p-3">
                  <Clock className="h-6 w-6 text-gray-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Total Jobs</p>
                  <p className="text-2xl font-bold text-gray-900">{filteredJobs.length}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Filters */}
          <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
            <div className="flex flex-col sm:flex-row gap-4">
              <div className="flex-1">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Filter by Status
                </label>
                <select
                  value={selectedStatus}
                  onChange={(e) => setSelectedStatus(e.target.value)}
                  className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
                >
                  <option value="all">All Statuses</option>
                  <option value="pending">Pending</option>
                  <option value="processing">Processing</option>
                  <option value="completed">Completed</option>
                  <option value="failed">Failed</option>
                  <option value="cancelled">Cancelled</option>
                </select>
              </div>

              <div className="flex-1">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Filter by Type
                </label>
                <select
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value)}
                  className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-primary focus:border-primary"
                >
                  <option value="all">All Types</option>
                  <option value="embedding">Document Processing</option>
                  <option value="query">Query Processing</option>
                  <option value="maintenance">Maintenance</option>
                </select>
              </div>
            </div>
          </div>

          {/* Jobs List */}
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
                <span className="ml-3 text-gray-600">Loading jobs...</span>
              </div>
            ) : filteredJobs.length === 0 ? (
              <div className="text-center py-12">
                <Clock className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Jobs Found</h3>
                <p className="text-gray-600">No jobs match your current filters.</p>
              </div>
            ) : (
              <div className="divide-y divide-gray-200">
                {filteredJobs.map((job, index) => (
                  <motion.div
                    key={job.job_id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="p-6 hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-4 flex-1">
                        <div className="bg-gray-100 rounded-full p-2">
                          <JobTypeIcon type={job.job_type} />
                        </div>
                        
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center space-x-3 mb-2">
                            <h3 className="text-lg font-medium text-gray-900 truncate">
                              {job.metadata?.original_name || job.metadata?.filename || `${job.job_type} Job`}
                            </h3>
                            <Badge className={cn('text-xs', getStatusColor(job.status))}>
                              <StatusIcon status={job.status} />
                              <span className="ml-1 capitalize">{job.status}</span>
                            </Badge>
                          </div>
                          
                          <div className="flex items-center space-x-4 text-sm text-gray-500 mb-3">
                            <span>Job ID: {job.job_id}</span>
                            <span>Type: {job.job_type}</span>
                            <span>Created: {format(new Date(job.created_at), 'MMM d, HH:mm:ss')}</span>
                            {job.status !== 'pending' && (
                              <span>Duration: {formatDuration(getJobDuration(job))}</span>
                            )}
                          </div>

                          {/* Progress bar for active jobs */}
                          {(job.status === 'processing' || job.status === 'pending') && (
                            <div className="mb-3">
                              <div className="flex items-center justify-between text-sm mb-1">
                                <span className="text-gray-600">Progress</span>
                                <span className="text-gray-600">{getProgressText(job)}</span>
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div 
                                  className="bg-primary h-2 rounded-full transition-all duration-300"
                                  style={{ width: `${job.progress * 100}%` }}
                                />
                              </div>
                            </div>
                          )}

                          {/* Error message */}
                          {job.status === 'failed' && job.error && (
                            <div className="bg-red-50 border border-red-200 rounded-md p-3 mb-3">
                              <div className="flex items-start">
                                <AlertTriangle className="h-5 w-5 text-red-400 mt-0.5 mr-2 flex-shrink-0" />
                                <div>
                                  <p className="text-sm font-medium text-red-800">Error</p>
                                  <p className="text-sm text-red-700 mt-1">{job.error}</p>
                                </div>
                              </div>
                            </div>
                          )}

                          {/* Job result summary */}
                          {job.status === 'completed' && job.result && (
                            <div className="bg-green-50 border border-green-200 rounded-md p-3">
                              <p className="text-sm text-green-800">
                                {job.job_type === 'embedding' && job.result.chunks_processed && (
                                  `Processed ${job.result.chunks_processed} chunks successfully`
                                )}
                                {job.job_type === 'query' && job.result.results_count !== undefined && (
                                  `Found ${job.result.results_count} relevant documents`
                                )}
                              </p>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center space-x-2 ml-4">
                        {job.status === 'failed' && (
                          <Button variant="outline" size="sm">
                            <RefreshCw className="h-4 w-4 mr-1" />
                            Retry
                          </Button>
                        )}
                        {(job.status === 'completed' || job.status === 'failed') && (
                          <Button variant="ghost" size="sm">
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
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