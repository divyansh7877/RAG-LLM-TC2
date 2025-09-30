# Keycloak Authentication Fix - Step by Step

## Issue Summary
Your React frontend is experiencing authentication failures with Keycloak due to:
1. **Browser blocking 3rd-party cookies** - preventing silent SSO checks
2. **Storage mismatch** - Keycloak storing state in localStorage but looking in sessionStorage
3. **Stale authentication state** - leftover data from failed login attempts

## Immediate Fix Steps

### Step 1: Clear All Browser Storage
Open your browser DevTools (F12) and run this in the console while on http://localhost:3000:
```javascript
// Clear all Keycloak data
Object.keys(sessionStorage).forEach(k => {
  if (k.startsWith('kc-') || k.startsWith('oidc.') || k.includes('__KC')) 
    sessionStorage.removeItem(k);
});
Object.keys(localStorage).forEach(k => {
  if (k.startsWith('kc-') || k.startsWith('oidc.')) 
    localStorage.removeItem(k);
});
console.log('Storage cleared!');
```

Or use the helper page:
```bash
# Open this in your browser
http://localhost:3000/clear_auth_storage.html
```

### Step 2: Test Basic Keycloak Connectivity
Before testing the full app, verify Keycloak works:
```bash
# Open the test page
http://localhost:3000/test-keycloak.html
```
1. Click "Initialize Keycloak"
2. Click "Login" 
3. Complete authentication
4. Verify you see user info after redirect

### Step 3: Verify Keycloak Client Configuration
Login to Keycloak Admin Console at http://192.168.1.117:8080/admin

Navigate to: **Realm: rag_app → Clients → fastapi-client**

Ensure these settings:
```
Access Settings:
- Root URL: http://localhost:3000
- Valid redirect URIs: 
  * http://localhost:3000/*
  * http://192.168.1.117:3000/*
- Valid post logout redirect URIs:
  * http://localhost:3000/*
  * http://192.168.1.117:3000/*
- Web origins: 
  * http://localhost:3000
  * http://192.168.1.117:3000
  * +

Capability Config:
- Client authentication: OFF
- Standard flow: ON
- Direct access grants: ON
```

### Step 4: Update Frontend Configuration
The code has been updated with:
- Clearing both localStorage and sessionStorage on init
- Handling login_required errors properly
- Disabling iframe checks that require 3rd-party cookies

### Step 5: Restart the Frontend
```bash
cd frontend
# Kill any running dev server (Ctrl+C)
npm run dev
```

### Step 6: Test Authentication
1. Open http://localhost:3000 in a **new incognito/private window**
2. Open DevTools Console to see auth logs
3. Click Login
4. Complete Keycloak authentication
5. You should be redirected back and logged in

## If Issues Persist

### Option A: Enable Development Mode (Bypass Auth)
Temporarily bypass Keycloak for development:
```bash
# Edit frontend/.env.local
NEXT_PUBLIC_DEV_MODE=true
```

### Option B: Use HTTPS for Cookie Support
3rd-party cookies work better over HTTPS:
```bash
# Generate self-signed cert
cd frontend
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Update package.json dev script
"dev": "next dev --experimental-https"
```

### Option C: Browser Settings
Some browsers block 3rd-party cookies by default. Try:
- Chrome: Settings → Privacy → Cookies → Allow all cookies (temporarily)
- Firefox: Settings → Privacy → Custom → Uncheck "Cookies"
- Or use a different browser for testing

## Debug Tools Available

1. **Auth Debug Panel** - Bottom-right corner in dev mode shows:
   - Current auth state
   - Storage keys
   - Clear storage button

2. **Console Logs** - Look for `[Auth]` prefixed messages:
   ```
   [Auth] Starting auth initialization
   [Auth] Keycloak init completed
   [Auth] User is authenticated
   ```

3. **Test Page** - http://localhost:3000/test-keycloak.html
   - Direct Keycloak testing without React
   - Shows detailed error messages

## Common Error Messages and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "Invalid nonce" | Stale state in storage | Clear all storage and retry |
| "login_required" | SSO check failed due to cookies | Normal - will be cleaned up automatically |
| "requestStorageAccess: May not be used in an insecure context" | HTTP instead of HTTPS | Can be ignored or use HTTPS |
| "Your browser is blocking access to 3rd-party cookies" | Browser security | Expected - auth will still work |

## Files Modified
- `frontend/lib/auth.tsx` - Enhanced storage cleanup and error handling
- `frontend/.env.local` - Disabled PKCE temporarily
- `frontend/components/AuthDebug.tsx` - Debug panel component
- `frontend/app/layout.tsx` - Added debug panel
- `frontend/public/test-keycloak.html` - Standalone test page
- `frontend/clear_auth_storage.html` - Storage cleanup utility

## Next Steps After Fix
Once authentication is working:
1. Re-enable PKCE for production: `NEXT_PUBLIC_KEYCLOAK_USE_PKCE=true`
2. Remove debug panel from production builds
3. Configure proper HTTPS certificates
4. Set up proper CORS headers in Keycloak for production domains