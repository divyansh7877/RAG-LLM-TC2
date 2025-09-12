'use client'

import { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import Keycloak from 'keycloak-js'

interface User {
  id: string
  username?: string
  email?: string
  groups: string[]
  roles: string[]
  preferred_username?: string
  sub?: string
}

interface AuthContextType {
  keycloak: Keycloak | null
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  login: () => void
  logout: () => void
  token: string | null
}

const AuthContext = createContext<AuthContextType | null>(null)

interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [keycloak, setKeycloak] = useState<Keycloak | null>(null)
  const [user, setUser] = useState<User | null>(null)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [token, setToken] = useState<string | null>(null)
  const [initialized, setInitialized] = useState(false)
  
  // Check if we should bypass auth in development
  const shouldBypassAuth = process.env.NODE_ENV === 'development' && process.env.NEXT_PUBLIC_DEV_MODE === 'true'

  useEffect(() => {
    // Prevent multiple initializations
    if (initialized) return
    
    console.log('[Auth] Initializing auth - bypass mode:', shouldBypassAuth)
    
    if (shouldBypassAuth) {
      // Development mode - skip Keycloak
      setUser({
        id: 'dev-user',
        username: 'developer',
        email: 'dev@localhost',
        groups: ['default'],
        roles: ['user'],
        preferred_username: 'developer'
      })
      setIsAuthenticated(true)
      setToken('dev-token')
      setIsLoading(false)
      setInitialized(true)
      return
    }
    
    // Production mode - initialize Keycloak
    const initKeycloak = async () => {
      try {
        const keycloakInstance = new Keycloak({
          url: 'http://localhost:8080/',
          realm: 'rag_app',
          clientId: 'fastapi-client'
        })

        const authenticated = await keycloakInstance.init({ 
          onLoad: 'check-sso',
          silentCheckSsoRedirectUri: window.location.origin + '/silent-check-sso.html',
          checkLoginIframe: false, // Disable iframe to prevent refresh loops
          pkceMethod: 'S256'
        })

        setKeycloak(keycloakInstance)
        
        if (authenticated) {
          const tokenParsed = keycloakInstance.tokenParsed || {}
          const userData: User = {
            id: tokenParsed.sub || '',
            username: tokenParsed.preferred_username,
            email: tokenParsed.email,
            groups: tokenParsed.groups || [],
            roles: tokenParsed.realm_access?.roles || [],
            preferred_username: tokenParsed.preferred_username
          }

          setUser(userData)
          setIsAuthenticated(true)
          setToken(keycloakInstance.token || null)
          
          // Set up token refresh - but only once
          keycloakInstance.onTokenExpired = () => {
            keycloakInstance.updateToken(30).catch(() => {
              console.error('Token refresh failed')
              keycloakInstance.logout()
            })
          }
        } else {
          setIsAuthenticated(false)
          setUser(null)
          setToken(null)
        }
      } catch (error) {
        console.error('Keycloak failed, using dev mode:', error)
        // Fallback to dev mode if Keycloak fails
        setUser({
          id: 'dev-user-fallback',
          username: 'developer',
          email: 'dev@localhost',
          groups: ['default'],
          roles: ['user'],
          preferred_username: 'developer (fallback)'
        })
        setIsAuthenticated(true)
        setToken('dev-token-fallback')
      } finally {
        setIsLoading(false)
        setInitialized(true)
      }
    }

    initKeycloak()
  }, []) // Empty dependency array to run only once

  const login = () => {
    keycloak?.login()
  }

  const logout = () => {
    keycloak?.logout()
  }

  const value: AuthContextType = {
    keycloak,
    user,
    isAuthenticated,
    isLoading,
    login,
    logout,
    token
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

// Helper hook for protecting routes
export function useRequireAuth() {
  const { isAuthenticated, isLoading } = useAuth()
  
  return { isAuthenticated, isLoading }
}

// API helper for making authenticated requests (matches your apiFetch method)
export async function makeAuthenticatedRequest(
  keycloak: Keycloak, 
  url: string, 
  options: RequestInit = {}
) {
  // Ensure token is fresh before making the request
  if (!keycloak.authenticated) {
    throw new Error('User not authenticated')
  }

  try {
    await keycloak.updateToken(30)
  } catch (e) {
    console.error('Token refresh failed:', e)
    keycloak.logout()
    throw new Error('User not authenticated')
  }

  const makeRequest = async () => {
    const headers = {
      ...options.headers,
      'Authorization': `Bearer ${keycloak.token}`
    }
    return fetch(url, { ...options, headers })
  }

  let response = await makeRequest()

  // If unauthorized, try one refresh + retry
  if (response.status === 401) {
    try {
      await keycloak.updateToken(30)
      response = await makeRequest()
    } catch (_) {
      // fallthrough to error handling
    }
  }

  if (!response.ok) {
    let error
    try { 
      error = await response.json() 
    } catch { 
      error = {} 
    }
    throw new Error(error.detail || 'API request failed')
  }

  return response.json()
}