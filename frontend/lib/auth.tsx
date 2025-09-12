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

  useEffect(() => {
    const initKeycloak = async () => {
      try {
        // Match your exact Keycloak configuration
        const keycloakInstance = new Keycloak({
          url: 'http://192.168.1.117:8080/',
          realm: 'rag_app',
          clientId: 'fastapi-client'
        })

        const authenticated = await keycloakInstance.init({ 
          onLoad: 'check-sso'
        })

        setKeycloak(keycloakInstance)

        if (authenticated) {
          console.log('User is authenticated')
          
          // Merge id token and access token claims for completeness (exactly like your original)
          const idClaims = keycloakInstance.idTokenParsed || {}
          const accessClaims = keycloakInstance.tokenParsed || {}
          const mergedClaims = { ...idClaims, ...accessClaims }

          // Normalize groups to an array of strings
          const rawGroups = mergedClaims.groups ?? mergedClaims.group ?? null
          const groups = Array.isArray(rawGroups)
            ? rawGroups
            : (rawGroups ? [rawGroups] : [])

          // Normalize roles to an array of strings  
          const rawRoles = mergedClaims.realm_access?.roles ?? mergedClaims.roles ?? []
          const roles = Array.isArray(rawRoles) ? rawRoles : [rawRoles]

          const userData: User = {
            id: mergedClaims.sub || '',
            username: mergedClaims.preferred_username,
            email: mergedClaims.email,
            groups,
            roles,
            preferred_username: mergedClaims.preferred_username,
            sub: mergedClaims.sub
          }

          setUser(userData)
          setIsAuthenticated(true)
          setToken(keycloakInstance.token || null)

          // Set up token refresh (exactly like your original)
          keycloakInstance.onTokenExpired = () => {
            keycloakInstance.updateToken(30).catch(() => {
              console.error('Failed to refresh token')
              keycloakInstance.logout()
            })
          }

          // Update token when refreshed - but only if it actually changed
          keycloakInstance.onAuthRefreshSuccess = () => {
            const newToken = keycloakInstance.token || null
            if (newToken && newToken !== token) {
              setToken(newToken)
            }
          }
        } else {
          console.log('User is not authenticated')
          setIsAuthenticated(false)
          setUser(null)
          setToken(null)
        }
      } catch (error) {
        console.error('Keycloak initialization failed:', error)
        setIsAuthenticated(false)
        setUser(null)
        setToken(null)
      } finally {
        setIsLoading(false)
      }
    }

    initKeycloak()
  }, [])

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