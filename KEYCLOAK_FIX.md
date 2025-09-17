# Fix for Keycloak Invalid Nonce Error

## Problem
When logging in through Keycloak from the React frontend, users were redirected properly to Keycloak but upon returning, received an "invalid nonce" error in the console, preventing successful authentication.

## Root Causes
1. **Session Storage Mismatch**: The nonce stored in sessionStorage during the initial redirect didn't match what was returned
2. **Multiple Initialization**: React StrictMode or hot module reloading could cause multiple Keycloak instances
3. **Stale Authentication State**: Previous failed attempts left orphaned session data
4. **Configuration Mismatch**: Redirect URIs and PKCE settings needed alignment

## Solutions Applied

### 1. Code Changes in `frontend/lib/auth.tsx`

#### Clear Stale Session Data on Initialization
- Added automatic cleanup of old `kc-callback-*` and `oidc.*` keys from sessionStorage when creating Keycloak instance
- Prevents nonce conflicts from previous sessions

#### Enhanced Keycloak Configuration
- Added `silentCheckSsoRedirectUri` and explicit `redirectUri` to init config
- Added `messageReceiveTimeout` and `timeSkew` tolerance
- Enabled PKCE (S256) for better security

#### Login Flow Improvements  
- Clear all stale authentication state before initiating login
- Added `prompt: 'login'` to force fresh authentication
- Remove retry markers from sessionStorage

### 2. Environment Configuration Changes

Updated `.env.local`:
```env
NEXT_PUBLIC_KEYCLOAK_USE_PKCE=true  # Changed from false
NEXT_PUBLIC_KEYCLOAK_RESPONSE_MODE=query
```

### 3. Keycloak Client Configuration Requirements

The `fastapi-client` in Keycloak must have:

**Access Settings:**
- Root URL: `http://localhost:3000`
- Valid redirect URIs:
  - `http://localhost:3000/*`
  - `http://192.168.1.117:3000/*`
- Web origins:
  - `http://localhost:3000`
  - `http://192.168.1.117:3000`
  - `+` (for CORS)

**Capability Config:**
- Client authentication: OFF (Public client)
- Standard flow: ON
- Direct access grants: ON
- Implicit flow: OFF

**Advanced Settings:**
- PKCE Code Challenge Method: S256

### 4. Debug Tools Added

- **AuthDebug Component**: Shows real-time auth state, storage keys, and provides manual cleanup
- **verify_keycloak_config.sh**: Script to verify Keycloak settings

## How to Apply the Fix

1. **Stop your frontend server** if running

2. **Clear browser data** for localhost:3000:
   - Open DevTools → Application → Storage → Clear site data

3. **Verify Keycloak configuration**:
   ```bash
   ./verify_keycloak_config.sh
   ```
   Then check settings at http://192.168.1.117:8080/admin

4. **Restart the frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

5. **Test authentication**:
   - Navigate to http://localhost:3000
   - Click login
   - Complete Keycloak authentication
   - Should redirect back successfully

## Troubleshooting

If issues persist:

1. **Use the Auth Debug Panel** (bottom-right in dev mode):
   - Check if session storage has stale `kc-callback-*` keys
   - Use "Clear All Auth State" button and refresh

2. **Check browser console** for specific errors:
   - Look for `[Auth]` prefixed messages
   - Note any CORS or redirect URI mismatches

3. **Verify network conditions**:
   - Keycloak at 192.168.1.117:8080 is accessible
   - No firewall blocking between frontend and Keycloak

4. **Try alternate response mode** if needed:
   - Change `NEXT_PUBLIC_KEYCLOAK_RESPONSE_MODE=fragment` in `.env.local`
   - Clear browser data and retry

## Prevention

- Always clear browser storage when switching between auth configurations
- Keep PKCE enabled for better security
- Ensure redirect URIs in Keycloak match exactly with your access URLs
- Use the debug panel during development to monitor auth state