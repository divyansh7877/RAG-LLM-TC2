'use client'

import { useAuth } from '@/lib/auth'
import { Button } from '@/components/ui/button'
import { Brain, Lock, LogIn } from 'lucide-react'

export function LoginScreen() {
  const { login, isLoading } = useAuth()

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center">
      <div className="max-w-md w-full space-y-8 p-8">
        <div className="text-center">
          <div className="mx-auto bg-primary rounded-full p-4 w-16 h-16 flex items-center justify-center mb-6">
            <Brain className="h-8 w-8 text-primary-foreground" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">RAG System</h1>
          <p className="text-gray-600 mb-8">Document Intelligence Platform</p>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-8 space-y-6">
          <div className="text-center">
            <div className="bg-red-50 rounded-full p-3 w-12 h-12 flex items-center justify-center mx-auto mb-4">
              <Lock className="h-6 w-6 text-red-600" />
            </div>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">Login Required</h2>
            <p className="text-gray-600 mb-6">
              Please log in to access the application.
            </p>
          </div>

          <Button 
            onClick={login}
            className="w-full"
            size="lg"
          >
            <LogIn className="mr-2 h-5 w-5" />
            Login with Keycloak
          </Button>
        </div>

        <div className="text-center text-sm text-gray-500">
          Secure authentication powered by Keycloak
        </div>
      </div>
    </div>
  )
}