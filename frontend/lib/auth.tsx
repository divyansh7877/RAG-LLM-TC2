'use client'

import { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import Keycloak from 'keycloak-js'

declare global {
  interface Window {
    __KC_INIT_STARTED__?: boolean
    __KEYCLOAK_INSTANCE__?: Keycloak | null
  }
}

let keycloakSingleton: Keycloak | null = null

function getKeycloakInstance() {
  // Try to get instance from window first (persists across page reloads)
  if (typeof window !== 'undefined' && window.__KEYCLOAK_INSTANCE__) {
    console.log('[Auth] Using existing Keycloak instance from window')
    keycloakSingleton = window.__KEYCLOAK_INSTANCE__
    return keycloakSingleton
  }
  
  if (!keycloakSingleton) {
    // Only clear stale state if we're NOT returning from Keycloak
    // (Don't clear callback state when we have an auth code in URL)
    if (typeof window !== 'undefined') {
      const urlParams = new URLSearchParams(window.location.search);
      const hasAuthCode = urlParams.has('code') && urlParams.has('state');
      
      if (!hasAuthCode) {
        console.log('[Auth] No auth code - clearing old storage data');
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
      } else {
        console.log('[Auth] Auth code present - preserving callback state in storage');
      }
    }
    
    // Get Keycloak configuration from environment variables
    const keycloakUrl = process.env.NEXT_PUBLIC_KEYCLOAK_URL || 'http://192.168.1.117:8080';
    const keycloakRealm = process.env.NEXT_PUBLIC_KEYCLOAK_REALM || 'rag_app';
    const keycloakClientId = process.env.NEXT_PUBLIC_KEYCLOAK_CLIENT_ID || 'fastapi-client';
    
    console.log('[Auth] Initializing Keycloak with config:', {
      url: keycloakUrl,
      realm: keycloakRealm,
      clientId: keycloakClientId
    });
    
    keycloakSingleton = new Keycloak({
      url: keycloakUrl,
      realm: keycloakRealm,
      clientId: keycloakClientId
    })
    
    // Store in window to persist across page reloads
    if (typeof window !== 'undefined') {
      window.__KEYCLOAK_INSTANCE__ = keycloakSingleton
    }
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
        
        // If we have an auth code, we need to handle this manually due to page reload
        // The Keycloak instance that initiated login is gone, so we'll exchange the code ourselves
        if (hasAuthCode && urlState) {
          console.log('[Auth] Auth code detected - handling OAuth callback manually')
          
          try {
            // Get the callback data from storage
            const callbackKey = `kc-callback-${urlState}`
            let callbackData = sessionStorage.getItem(callbackKey) || localStorage.getItem(callbackKey)
            
            if (!callbackData) {
              console.error('[Auth] No callback data found for state:', urlState)
              // Clean up and trigger fresh login
              cleanUrlParams()
              const kc = getKeycloakInstance()
              setTimeout(() => kc.login({ redirectUri: window.location.origin + '/' }), 0)
              return
            }
            
            const { nonce, redirectUri } = JSON.parse(callbackData)
            const code = urlParams.get('code')
            
            console.log('[Auth] Exchanging authorization code for tokens')
            
            // Exchange code for tokens using fetch
            const tokenEndpoint = `${process.env.NEXT_PUBLIC_KEYCLOAK_URL || 'http://192.168.1.117:8080'}/realms/${process.env.NEXT_PUBLIC_KEYCLOAK_REALM || 'rag_app'}/protocol/openid-connect/token`
            
            const tokenResponse = await fetch(tokenEndpoint, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
              },
              body: new URLSearchParams({
                grant_type: 'authorization_code',
                code: code!,
                redirect_uri: decodeURIComponent(redirectUri),
                client_id: process.env.NEXT_PUBLIC_KEYCLOAK_CLIENT_ID || 'fastapi-client',
              }),
            })
            
            if (!tokenResponse.ok) {
              throw new Error(`Token exchange failed: ${tokenResponse.status} ${tokenResponse.statusText}`)
            }
            
            const tokens = await tokenResponse.json()
            console.log('[Auth] Successfully exchanged code for tokens')
            
            // Parse the ID token to verify nonce
            const idTokenParts = tokens.id_token.split('.')
            const idTokenPayload = JSON.parse(atob(idTokenParts[1]))
            
            if (idTokenPayload.nonce !== nonce) {
              console.error('[Auth] Nonce mismatch! Expected:', nonce, 'Got:', idTokenPayload.nonce)
              throw new Error('Nonce mismatch')
            }
            
            // Parse access token for user data
            const accessTokenParts = tokens.access_token.split('.')
            const accessTokenPayload = JSON.parse(atob(accessTokenParts[1]))
            
            // Set up user data
            const userData: User = {
              id: accessTokenPayload.sub || '',
              username: accessTokenPayload.preferred_username,
              email: accessTokenPayload.email,
              groups: accessTokenPayload.groups || [],
              roles: accessTokenPayload.realm_access?.roles || [],
              preferred_username: accessTokenPayload.preferred_username
            }
            
            // Get Keycloak instance and manually set tokens
            const keycloakInstance = getKeycloakInstance()
            keycloakInstance.token = tokens.access_token
            keycloakInstance.refreshToken = tokens.refresh_token
            keycloakInstance.idToken = tokens.id_token
            keycloakInstance.tokenParsed = accessTokenPayload
            keycloakInstance.authenticated = true
            
            setKeycloak(keycloakInstance)
            setUser(userData)
            setIsAuthenticated(true)
            setToken(tokens.access_token)
            
            // Clean up
            cleanUrlParams()
            sessionStorage.removeItem(callbackKey)
            localStorage.removeItem(callbackKey)
            sessionStorage.removeItem('__KC_RETRY_ONCE__')
            
            // Setup token refresh
            if (!(keycloakInstance as any).__EVENTS_WIRED__) {
              ;(keycloakInstance as any).__EVENTS_WIRED__ = true
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
            }
            
            console.log('[Auth] Manual authentication complete')
            setIsLoading(false)
            setInitialized(true)
            setInitializingRef(false)
            return
            
          } catch (error) {
            console.error('[Auth] Manual token exchange failed:', error)
            // Fall through to normal init flow which will trigger fresh login
            cleanUrlParams()
          }
        }
        
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
        
        // When we have an auth code (returning from Keycloak), don't specify onLoad
        // to let Keycloak complete the authentication flow
        // Otherwise, use 'login-required' to force redirect to login page
        const onLoadAction = hasAuthCode ? undefined : 'login-required'
        
        const initConfig: any = {
          checkLoginIframe: false, // Disable iframe completely (doesn't work with 3rd party cookies blocked)
          enableLogging: true, // Enable keycloak logging
          flow: 'standard', // Authorization code flow
          responseMode, // Use query by default
          // Remove silent SSO settings as they don't work with blocked cookies
          redirectUri: window.location.origin + '/',
          // Add message and time skew tolerance
          messageReceiveTimeout: 10000,
          timeSkew: 5
        }
        // Only set onLoad if we have a value (when no auth code present)
        if (onLoadAction) {
          initConfig.onLoad = onLoadAction
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
    console.log('[Auth] Logging out')
    
    // Clear authentication state
    setUser(null)
    setIsAuthenticated(false)
    setToken(null)
    
    // Clear all storage
    try {
      sessionStorage.clear()
      localStorage.clear()
    } catch (e) {
      console.warn('[Auth] Could not clear storage:', e)
    }
    
    // Clear window globals
    if (typeof window !== 'undefined') {
      window.__KC_INIT_STARTED__ = false
      window.__KEYCLOAK_INSTANCE__ = null
    }
    
    // Build Keycloak logout URL
    const keycloakUrl = process.env.NEXT_PUBLIC_KEYCLOAK_URL || 'http://192.168.1.117:8080'
    const realm = process.env.NEXT_PUBLIC_KEYCLOAK_REALM || 'rag_app'
    const redirectUri = encodeURIComponent(window.location.origin + '/')
    const logoutUrl = `${keycloakUrl}/realms/${realm}/protocol/openid-connect/logout?post_logout_redirect_uri=${redirectUri}`
    
    // Redirect to Keycloak logout
    window.location.href = logoutUrl
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