'use client'

import { createContext, useContext, useEffect, useRef, ReactNode } from 'react'
import { api } from './api'
import { useAuth } from './auth'

const AuthenticatedApiContext = createContext(api)

interface AuthenticatedApiProviderProps {
  children: ReactNode
}

export function AuthenticatedApiProvider({ children }: AuthenticatedApiProviderProps) {
  const { token, keycloak } = useAuth()
  const lastTokenRef = useRef<string | null>(null)

  useEffect(() => {
    // Only update the API client if the token actually changed
    if (lastTokenRef.current !== token) {
      console.log('[API] Token changed, updating API client')
      lastTokenRef.current = token
      api.setAuthToken(token)
    }
  }, [token])

  return (
    <AuthenticatedApiContext.Provider value={api}>
      {children}
    </AuthenticatedApiContext.Provider>
  )
}

export function useApi() {
  return useContext(AuthenticatedApiContext)
}