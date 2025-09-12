'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Upload, 
  Search, 
  FileText, 
  Activity, 
  History, 
  Brain,
  ArrowRight,
  Zap,
  Shield,
  Users
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { AuthenticatedLayout } from '@/components/AuthenticatedLayout'

const navigation = [
  { name: 'Upload', href: '/upload', icon: Upload, description: 'Upload and process documents' },
  { name: 'Query', href: '/query', icon: Search, description: 'Search and query your documents' },
  { name: 'Documents', href: '/my-documents', icon: FileText, description: 'Manage your document library' },
  { name: 'Jobs', href: '/job-status', icon: Activity, description: 'Monitor processing status' },
  { name: 'History', href: '/query-history', icon: History, description: 'View query history' },
]

const features = [
  {
    icon: Zap,
    title: 'Lightning Fast',
    description: 'Optimized for high-performance document processing and querying',
  },
  {
    icon: Shield,
    title: 'Secure & Isolated',
    description: 'Complete data isolation and security for multi-user environments',
  },
  {
    icon: Users,
    title: 'Multi-User Ready',
    description: 'Concurrent processing with user and group-based access controls',
  },
]

export default function HomePage() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100" />
  }

  return (
    <AuthenticatedLayout>
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-end mb-8">
            <Button variant="outline" asChild>
              <Link href="/upload">Get Started</Link>
            </Button>
          </div>

      {/* Hero Section */}
      <section className="pt-16 pb-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center"
          >
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 mb-6">
              Intelligent Document
              <span className="bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">
                {' '}Processing
              </span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Upload, process, and query your documents with AI-powered search. 
              Built for teams with enterprise-grade security and concurrent processing.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" asChild>
                <Link href="/upload">
                  Start Processing Documents
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button variant="outline" size="lg" asChild>
                <Link href="/query">
                  Query Existing Documents
                </Link>
              </Button>
            </div>
          </motion.div>

          {/* Features Grid */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="mt-20 grid md:grid-cols-3 gap-8"
          >
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + index * 0.1 }}
                className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-md transition-shadow"
              >
                <div className="bg-primary/10 rounded-lg p-3 w-fit mb-4">
                  <feature.icon className="h-6 w-6 text-primary" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-600">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Navigation Cards */}
      <section className="pb-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Everything You Need
            </h2>
            <p className="text-lg text-gray-600">
              Complete document processing workflow from upload to insights
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <AnimatePresence>
              {navigation.map((item, index) => (
                <motion.div
                  key={item.name}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.5 + index * 0.1 }}
                  whileHover={{ y: -4 }}
                  className="group"
                >
                  <Link href={item.href} className="block">
                    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 hover:shadow-lg transition-all duration-200 hover:border-primary/20">
                      <div className="flex items-center mb-4">
                        <div className="bg-primary/10 rounded-lg p-3 group-hover:bg-primary/20 transition-colors">
                          <item.icon className="h-6 w-6 text-primary" />
                        </div>
                        <div className="ml-4">
                          <h3 className="text-lg font-semibold text-gray-900 group-hover:text-primary transition-colors">
                            {item.name}
                          </h3>
                        </div>
                        <ArrowRight className="ml-auto h-4 w-4 text-gray-400 group-hover:text-primary transition-colors" />
                      </div>
                      <p className="text-gray-600 text-sm">
                        {item.description}
                      </p>
                    </div>
                  </Link>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-200 bg-white/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex flex-col sm:flex-row justify-between items-center">
            <div className="flex items-center space-x-3 mb-4 sm:mb-0">
              <div className="bg-primary rounded-lg p-2">
                <Brain className="h-5 w-5 text-primary-foreground" />
              </div>
              <span className="text-sm text-gray-600">
                RAG System - Built with Next.js and FastAPI
              </span>
            </div>
            <p className="text-sm text-gray-500">
              Production-ready document intelligence platform
            </p>
          </div>
        </div>
      </footer>
        </div>
      </div>
    </AuthenticatedLayout>
  )
}
