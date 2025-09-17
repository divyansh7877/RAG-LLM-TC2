'use client'

import { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import Keycloak from 'keycloak-js'

declare global {
  interface Window {
    __KC_INIT_STARTED__?: boolean
  }
}

let keycloakSingleton: Keycloak | null = null

function getKeycloakInstance() {
  if (!keycloakSingleton) {
    // Clear any stale nonce/state from ALL storage locations
    if (typeof window !== 'undefined') {
      try {
        // Clear from sessionStorage
        const ssKeys = Object.keys(sessionStorage);
        ssKeys.forEach(key => {
          if (key.startsWith('kc-callback-') || key.startsWith('oidc.') || key === '__KC_RETRY_ONCE__') {
            sessionStorage.removeItem(key);
          }
        });
        
        // IMPORTANT: Also clear from localStorage as Keycloak might use either
        const lsKeys = Object.keys(localStorage);
        lsKeys.forEach(key => {
          if (key.startsWith('kc-callback-') || key.startsWith('oidc.')) {
            localStorage.removeItem(key);
          }
        });
      } catch (e) {
        console.warn('[Auth] Could not clear old storage data:', e);
      }
    }
    
    keycloakSingleton = new Keycloak({
      url: 'http://192.168.1.117:8080/',
      realm: 'rag_app',
      clientId: 'fastapi-client'
    })
  }
  return keycloakSingleton
}

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
  const [initializingRef, setInitializingRef] = useState(false)
  
  // Check if we should bypass auth in development
  const shouldBypassAuth = process.env.NODE_ENV === 'development' && process.env.NEXT_PUBLIC_DEV_MODE === 'true'

  useEffect(() => {
    // Prevent multiple initializations - strict guard and global guard for StrictMode/HMR
    if (typeof window !== 'undefined') {
      if ((window as any).__KC_INIT_STARTED__) {
        console.log('[Auth] Skipping init - already started (global)')
        return
      }
      ;(window as any).__KC_INIT_STARTED__ = true
    }

    if (initialized || initializingRef) {
      console.log('[Auth] Skipping init - already initialized or in progress:', { initialized, initializingRef })
      return
    }
    
    setInitializingRef(true)
    console.log('[Auth] Starting auth initialization - bypass mode:', shouldBypassAuth)
    
    if (shouldBypassAuth) {
      // Development mode - skip Keycloak
      console.log('[Auth] Using dev mode bypass')
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
      setInitializingRef(false)
      return
    }
    
    // Production mode - initialize Keycloak
    const initKeycloak = async () => {
      // Helpers
      const cleanUrlParams = () => {
        const url = new URL(window.location.href)
        ;['code', 'state', 'session_state', 'iss', 'error', 'error_description'].forEach((p) => url.searchParams.delete(p))
        const newUrl = url.pathname + (url.searchParams.toString() ? '?' + url.searchParams.toString() : '') + url.hash
        window.history.replaceState({}, document.title, newUrl)
      }
      const clearKcCallbackState = () => {
        try {
          // Clear from both sessionStorage and localStorage
          for (let i = 0; i < sessionStorage.length; i++) {
            const key = sessionStorage.key(i)
            if (key && key.startsWith('kc-callback-')) {
              sessionStorage.removeItem(key)
            }
          }
          for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i)
            if (key && key.startsWith('kc-callback-')) {
              localStorage.removeItem(key)
            }
          }
        } catch (e) {
          console.warn('[Auth] Unable to clear KC callback state', e)
        }
      }

      let observedAuthCodeAtStart = false
      try {
        console.log('[Auth] Current URL:', window.location.href)
        console.log('[Auth] URL fragment:', window.location.hash)
        console.log('[Auth] URL search params:', window.location.search)
        
        // Check if we're returning from Keycloak with an auth code or error
        const urlParams = new URLSearchParams(window.location.search)
        const hasAuthCode = urlParams.has('code') && urlParams.has('state')
        const hasError = urlParams.has('error')
        const errorType = urlParams.get('error')
        const urlState = urlParams.get('state') || ''
        
        console.log('[Auth] Has auth code in URL:', hasAuthCode)
        console.log('[Auth] Has error in URL:', hasError, 'Type:', errorType)
        observedAuthCodeAtStart = hasAuthCode
        
        // If we have a login_required error, clean it up before init
        if (hasError && errorType === 'login_required') {
          console.log('[Auth] Cleaning login_required error from URL')
          cleanUrlParams()
          // Clear any associated state
          if (urlState) {
            const callbackKey = `kc-callback-${urlState}`
            sessionStorage.removeItem(callbackKey)
            localStorage.removeItem(callbackKey)
          }
        }

        // Determine response mode from env once
        const responseModeEnv = process.env.NEXT_PUBLIC_KEYCLOAK_RESPONSE_MODE
        const responseMode = responseModeEnv === 'fragment' ? 'fragment' : 'query'

        // Debug: inspect callback storage for the state
        try {
          const cbKey = urlState ? `kc-callback-${urlState}` : null
          if (cbKey) {
            const ss = sessionStorage.getItem(cbKey)
            const ls = localStorage.getItem(cbKey)
            const keys = Array.from({ length: sessionStorage.length }, (_, i) => sessionStorage.key(i)).filter(Boolean)
            const kcKeys = keys.filter((k) => k && k.startsWith('kc-callback-'))
            console.log('[Auth][Debug] Callback key:', cbKey, 'existsInSessionStorage:', !!ss, 'existsInLocalStorage:', !!ls)
            console.log('[Auth][Debug] SessionStorage kc-callback-* keys:', kcKeys)
            if (ss) {
              try { console.log('[Auth][Debug] SessionStorage callback entry snippet:', ss.slice(0, 200)) } catch {}
            }
            if (ls) {
              try { console.log('[Auth][Debug] LocalStorage callback entry snippet:', ls.slice(0, 200)) } catch {}
            }
          }
        } catch (e) {
          console.warn('[Auth][Debug] Error inspecting storage', e)
        }
        
        // Clear any error fragments from previous attempts
        if (window.location.hash.includes('error=')) {
          console.log('[Auth] Clearing error fragment from URL')
          window.history.replaceState({}, document.title, window.location.pathname + window.location.search)
        }
        
        console.log('[Auth] Getting Keycloak singleton instance')
        const keycloakInstance = getKeycloakInstance()
        // Ensure we always have a KC instance available for manual login even if init fails
        setKeycloak(keycloakInstance)

        console.log('[Auth] Calling keycloak.init()')
        
        // Init configuration with fixes for cookie and storage issues
        const usePkce = process.env.NEXT_PUBLIC_KEYCLOAK_USE_PKCE === 'true'
        
        // Use 'login-required' instead of 'check-sso' to avoid cookie issues
        // This will redirect to login immediately if not authenticated
        const onLoadAction = hasAuthCode ? 'check-sso' : 'check-sso'
        
        const initConfig: any = {
          onLoad: onLoadAction,
          checkLoginIframe: false, // Disable iframe completely
          enableLogging: true, // Enable keycloak logging
          flow: 'standard', // Authorization code flow
          responseMode, // Use query by default
          silentCheckSsoFallback: false, // Don't fallback when SSO check fails
          silentCheckSsoRedirectUri: window.location.origin + '/',
          redirectUri: window.location.origin + '/',
          // Add message and time skew tolerance
          messageReceiveTimeout: 10000,
          timeSkew: 5,
          // Use default adapter
          adapter: 'default'
        }
        if (usePkce) {
          initConfig.pkceMethod = 'S256'
        }
        
        console.log('[Auth] Init config:', initConfig)
        const authenticated = await keycloakInstance.init(initConfig)

        console.log('[Auth] Keycloak init completed. Authenticated:', authenticated)
        
        if (authenticated) {
          console.log('[Auth] User is authenticated, setting up user data')
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
          
          // Clean up URL if we have auth params
          if (hasAuthCode) {
            console.log('[Auth] Cleaning up auth parameters from URL')
            cleanUrlParams()
          }
          
          // Set up token event handlers - wire once per singleton
          if (!(keycloakInstance as any).__EVENTS_WIRED__) {
            ;(keycloakInstance as any).__EVENTS_WIRED__ = true
            keycloakInstance.onAuthSuccess = () => {
              console.log('[Auth] onAuthSuccess')
            }
            keycloakInstance.onAuthError = (err) => {
              console.error('[Auth] onAuthError', err)
            }
            keycloakInstance.onTokenExpired = () => {
              console.log('[Auth] Token expired, attempting refresh')
              keycloakInstance
                .updateToken(30)
                .then((refreshed) => {
                  if (refreshed) {
                    setToken(keycloakInstance.token || null)
                  }
                })
                .catch((error) => {
                  console.error('[Auth] Token refresh failed:', error)
                  keycloakInstance.logout()
                })
            }
            keycloakInstance.onAuthRefreshSuccess = () => {
              console.log('[Auth] Token refresh success')
              setToken(keycloakInstance.token || null)
            }
          }
        } else {
          console.log('[Auth] User is not authenticated')
          setIsAuthenticated(false)
          setUser(null)
          setToken(null)
        }
      } catch (error) {
        console.error('[Auth] Keycloak initialization failed:', error)

        // Attempt one clean retry if we detect callback params (likely nonce/state mismatch)
        try {
          const alreadyRetried = sessionStorage.getItem('__KC_RETRY_ONCE__') === '1'
          const hasAuthCodeNow = new URLSearchParams(window.location.search).has('code')
          if (!alreadyRetried && (hasAuthCodeNow || observedAuthCodeAtStart)) {
            console.warn('[Auth] Detected init failure with auth code present. Cleaning and retrying login once...')
            sessionStorage.setItem('__KC_RETRY_ONCE__', '1')
            clearKcCallbackState()
            cleanUrlParams()
            const kc = getKeycloakInstance()
            setTimeout(() => kc.login({ redirectUri: window.location.origin + '/' }), 0)
            return
          }
        } catch (e) {
          console.warn('[Auth] Retry path encountered an error', e)
        }
        
        // If it's a configuration error, show a helpful message
        if (error && typeof error === 'object' && 'error' in error) {
          if ((error as any).error === 'unauthorized_client') {
            console.error(`
[Auth] KEYCLOAK CONFIGURATION ERROR:
The client 'fastapi-client' needs to be configured properly in Keycloak.

Required settings (in Client Details):
- Client authentication: OFF (makes it public)
- Authorization: OFF
- Standard flow: ON
- Implicit flow: OFF
- Direct access grants: ON
- Valid redirect URIs: http://localhost:3000/*
- Web origins: http://localhost:3000
            `)
          }
        }
        
        setIsAuthenticated(false)
        setUser(null)
        setToken(null)
      } finally {
        console.log('[Auth] Auth initialization complete')
        setIsLoading(false)
        setInitialized(true)
        setInitializingRef(false)
      }
    }

    initKeycloak()
    
    // Cleanup function to prevent issues on unmount
    return () => {
      console.log('[Auth] Auth provider cleanup')
    }
  }, []) // Empty dependency array to run only once

  const login = () => {
    if (!keycloak) {
      console.error('[Auth] Keycloak not initialized')
      return
    }
    
    // Clear any stale authentication state from BOTH storage locations before login
    try {
      // Clear sessionStorage
      const ssKeys = Object.keys(sessionStorage);
      ssKeys.forEach(key => {
        if (key.startsWith('kc-callback-') || key.startsWith('oidc.')) {
          sessionStorage.removeItem(key);
        }
      });
      sessionStorage.removeItem('__KC_RETRY_ONCE__');
      
      // Clear localStorage too
      const lsKeys = Object.keys(localStorage);
      lsKeys.forEach(key => {
        if (key.startsWith('kc-callback-') || key.startsWith('oidc.')) {
          localStorage.removeItem(key);
        }
      });
    } catch (e) {
      console.warn('[Auth] Could not clear storage before login:', e);
    }
    
    console.log('[Auth] Initiating login')
    keycloak.login({
      redirectUri: window.location.origin + '/',
      // Don't force login prompt if user might be logged in
      prompt: undefined
    })
  }

  const logout = () => {
    if (!keycloak) return
    const redirectUri = window.location.origin + '/'
    keycloak.logout({ redirectUri })
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