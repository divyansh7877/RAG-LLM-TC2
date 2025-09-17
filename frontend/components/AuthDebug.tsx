'use client'

import { useAuth } from '@/lib/auth'
import { useEffect, useState } from 'react'

export function AuthDebug() {
  const { keycloak, user, isAuthenticated, isLoading, token } = useAuth()
  const [sessionStorageKeys, setSessionStorageKeys] = useState<string[]>([])
  const [localStorageKeys, setLocalStorageKeys] = useState<string[]>([])

  useEffect(() => {
    // Get storage keys
    try {
      const ssKeys = Object.keys(sessionStorage).filter(k => 
        k.startsWith('kc-') || k.startsWith('oidc.') || k.includes('__KC')
      )
      setSessionStorageKeys(ssKeys)

      const lsKeys = Object.keys(localStorage).filter(k => 
        k.startsWith('kc-') || k.startsWith('oidc.')
      )
      setLocalStorageKeys(lsKeys)
    } catch (e) {
      console.error('Error reading storage:', e)
    }
  }, [isAuthenticated])

  const clearAllAuthState = () => {
    try {
      // Clear session storage
      Object.keys(sessionStorage).forEach(key => {
        if (key.startsWith('kc-') || key.startsWith('oidc.') || key.includes('__KC')) {
          sessionStorage.removeItem(key)
        }
      })
      // Clear local storage
      Object.keys(localStorage).forEach(key => {
        if (key.startsWith('kc-') || key.startsWith('oidc.')) {
          localStorage.removeItem(key)
        }
      })
      alert('Auth state cleared. Please refresh the page.')
    } catch (e) {
      console.error('Error clearing storage:', e)
    }
  }

  if (process.env.NODE_ENV !== 'development') {
    return null
  }

  return (
    <div className="fixed bottom-4 right-4 p-4 bg-gray-900 text-white rounded-lg shadow-lg max-w-md text-xs font-mono z-50">
      <div className="mb-2 font-bold text-yellow-400">Auth Debug Panel</div>
      
      <div className="space-y-1">
        <div>Loading: {isLoading ? '✓' : '✗'}</div>
        <div>Authenticated: {isAuthenticated ? '✓' : '✗'}</div>
        <div>Keycloak Instance: {keycloak ? '✓' : '✗'}</div>
        <div>Token: {token ? `${token.substring(0, 20)}...` : 'none'}</div>
        <div>User: {user?.username || 'none'}</div>
        
        <div className="mt-2 pt-2 border-t border-gray-700">
          <div className="text-yellow-400">Session Storage Keys:</div>
          {sessionStorageKeys.length > 0 ? (
            sessionStorageKeys.map(key => (
              <div key={key} className="pl-2 text-xs">{key}</div>
            ))
          ) : (
            <div className="pl-2 text-gray-500">None</div>
          )}
        </div>

        <div className="mt-2 pt-2 border-t border-gray-700">
          <div className="text-yellow-400">Local Storage Keys:</div>
          {localStorageKeys.length > 0 ? (
            localStorageKeys.map(key => (
              <div key={key} className="pl-2 text-xs">{key}</div>
            ))
          ) : (
            <div className="pl-2 text-gray-500">None</div>
          )}
        </div>

        <div className="mt-2 pt-2 border-t border-gray-700">
          <button
            onClick={clearAllAuthState}
            className="bg-red-600 hover:bg-red-700 px-2 py-1 rounded text-xs"
          >
            Clear All Auth State
          </button>
        </div>
      </div>
    </div>
  )
}