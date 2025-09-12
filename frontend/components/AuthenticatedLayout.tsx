'use client'

import { useState } from 'react'
import { useAuth } from '@/lib/auth'
import { LoginScreen } from './LoginScreen'
import { Button } from './ui/button'
import { 
  Brain, 
  LogOut, 
  Search, 
  Upload, 
  FileText, 
  Clock,
  Activity,
  Menu,
  X
} from 'lucide-react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { cn } from '@/lib/utils'

interface AuthenticatedLayoutProps {
  children: React.ReactNode
}

const navigationItems = [
  {
    name: 'Home',
    href: '/',
    icon: Brain,
    description: 'Dashboard and overview'
  },
  {
    name: 'Upload Documents',
    href: '/upload',
    icon: Upload,
    description: 'Upload and process documents'
  },
  {
    name: 'Query Documents',
    href: '/query',
    icon: Search,
    description: 'Search and query your documents'
  },
  {
    name: 'My Documents',
    href: '/my-documents',
    icon: FileText,
    description: 'Manage uploaded documents'
  },
  {
    name: 'Query History',
    href: '/query-history',
    icon: Clock,
    description: 'View past queries and results'
  },
  {
    name: 'Job Status',
    href: '/job-status',
    icon: Activity,
    description: 'Monitor background jobs'
  },
]

export function AuthenticatedLayout({ children }: AuthenticatedLayoutProps) {
  const { isAuthenticated, isLoading, user, logout } = useAuth()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const pathname = usePathname()
  
  // Development mode bypass
  const isDevMode = process.env.NEXT_PUBLIC_DEV_MODE === 'true'
  const shouldBypassAuth = isDevMode && process.env.NODE_ENV === 'development'

  if (isLoading && !shouldBypassAuth) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    )
  }

  if (!isAuthenticated && !shouldBypassAuth) {
    return <LoginScreen />
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <Link href="/" className="flex items-center space-x-3">
              <div className="bg-primary rounded-lg p-2">
                <Brain className="h-6 w-6 text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">RAG System</h1>
                <p className="text-sm text-gray-600">Document Intelligence Platform</p>
              </div>
            </Link>
            
            {/* Desktop Navigation */}
            <nav className="hidden lg:flex items-center space-x-1">
              {navigationItems.slice(1).map((item) => {
                const Icon = item.icon
                const isActive = pathname === item.href
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={cn(
                      'flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors',
                      isActive
                        ? 'bg-primary/10 text-primary'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                    )}
                    title={item.description}
                  >
                    <Icon className="h-4 w-4 mr-2" />
                    {item.name}
                  </Link>
                )
              })}
            </nav>
            
            <div className="flex items-center space-x-4">
              {(user || shouldBypassAuth) && (
                <div className="text-sm text-gray-600 hidden md:block">
                  Welcome, <span className="font-medium">
                    {shouldBypassAuth ? 'Developer' : (user?.preferred_username || user?.username)}
                  </span>
                  {shouldBypassAuth && <span className="ml-1 text-orange-600">(Dev Mode)</span>}
                </div>
              )}
              {!shouldBypassAuth ? (
                <Button variant="outline" onClick={logout} size="sm">
                  <LogOut className="h-4 w-4 mr-2" />
                  <span className="hidden sm:inline">Logout</span>
                </Button>
              ) : (
                <div className="text-xs text-orange-600 bg-orange-50 px-2 py-1 rounded">
                  Dev Mode Active
                </div>
              )}
              
              {/* Mobile menu button */}
              <Button
                variant="ghost"
                size="sm"
                className="lg:hidden"
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              >
                {mobileMenuOpen ? (
                  <X className="h-5 w-5" />
                ) : (
                  <Menu className="h-5 w-5" />
                )}
              </Button>
            </div>
          </div>
          
          {/* Mobile Navigation */}
          {mobileMenuOpen && (
            <div className="lg:hidden border-t border-gray-200 py-4">
              <nav className="space-y-1">
                {navigationItems.map((item) => {
                  const Icon = item.icon
                  const isActive = pathname === item.href
                  return (
                    <Link
                      key={item.name}
                      href={item.href}
                      className={cn(
                        'flex items-center px-3 py-3 text-sm font-medium rounded-md transition-colors',
                        isActive
                          ? 'bg-primary/10 text-primary border-l-2 border-primary'
                          : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                      )}
                      onClick={() => setMobileMenuOpen(false)}
                    >
                      <Icon className="h-4 w-4 mr-3" />
                      <div>
                        <div>{item.name}</div>
                        <div className="text-xs text-gray-500 mt-0.5">{item.description}</div>
                      </div>
                    </Link>
                  )
                })}
              </nav>
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main>
        {children}
      </main>
    </div>
  )
}