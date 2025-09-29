# Keycloak Integration Fix for React Frontend

## Current Setup
- **Keycloak URL**: http://192.168.1.117:8080
- **Realm**: rag_app
- **Client ID**: fastapi-client
- **Frontend**: Next.js (React) running on http://localhost:3000

## Issues Identified

1. The frontend is hardcoded to connect to `192.168.1.117:8080`
2. Valid redirect URIs may not be configured for `localhost:3000`
3. Web origins not configured properly
4. Possible CORS issues between localhost and 192.168.1.117

## Required Keycloak Configuration

### Step 1: Configure the Client in Keycloak

1. **Access Keycloak Admin Console**
   ```
   URL: http://192.168.1.117:8080
   Username: admin
   Password: <your-admin-password>
   ```

2. **Navigate to Client Configuration**
   - Go to: Realms → `rag_app` → Clients → `fastapi-client`

3. **Settings Tab - Configure these values:**

   ```
   Client ID: fastapi-client
   Name: FastAPI Client
   Description: Client for RAG application
   Enabled: ON
   
   Client authentication: OFF (Public client)
   Authorization: OFF
   
   Authentication flow:
   ☑ Standard flow
   ☐ Implicit flow  
   ☑ Direct access grants
   ☐ Service accounts roles
   
   Valid redirect URIs:
   http://localhost:3000/*
   http://192.168.1.117:3000/*
   http://localhost/*
   
   Valid post logout redirect URIs:
   http://localhost:3000/*
   http://192.168.1.117:3000/*
   
   Web origins:
   http://localhost:3000
   http://192.168.1.117:3000
   http://localhost
   +
   
   Admin URL: (leave empty)
   
   Front channel logout: ON
   ```

4. **Advanced Settings Tab:**
   ```
   Proof Key for Code Exchange Code Challenge Method: S256 (if using PKCE)
   Or: (empty) if PKCE is disabled
   
   Access Token Lifespan: 5 Minutes (or your preference)
   Client Session Idle: 30 Minutes
   Client Session Max: 10 Hours
   ```

### Step 2: Configure Realm Settings

1. **Go to**: Realms → `rag_app` → Realm Settings → Login

2. **Ensure these are enabled:**
   ```
   User registration: OFF (unless you want public registration)
   Forgot password: ON
   Remember me: ON
   ```

3. **Go to**: Realms → `rag_app` → Realm Settings → Tokens

   ```
   Default Signature Algorithm: RS256
   Revoke Refresh Token: OFF
   Access Token Lifespan: 5 Minutes
   SSO Session Idle: 30 Minutes
   SSO Session Max: 10 Hours
   ```

### Step 3: Create Test User (if needed)

1. **Go to**: Realms → `rag_app` → Users → Add User

   ```
   Username: testuser
   Email: testuser@example.com
   Email Verified: ON
   Enabled: ON
   ```

2. **Set Password**:
   - Go to user → Credentials tab
   - Set password: `testpassword` (or your choice)
   - Temporary: OFF

3. **Assign Groups** (if using groups):
   - Go to user → Groups tab
   - Join groups: `default`, `assistants`, etc.

## Frontend Configuration Fixes

### Fix 1: Update Environment Variables

Edit `frontend/.env.local`:

```bash
# API Configuration - Use 192.168.1.117 to match Keycloak
NEXT_PUBLIC_API_BASE_URL=http://192.168.1.117:8000

# Keycloak Configuration
NEXT_PUBLIC_KEYCLOAK_URL=http://192.168.1.117:8080
NEXT_PUBLIC_KEYCLOAK_REALM=rag_app
NEXT_PUBLIC_KEYCLOAK_CLIENT_ID=fastapi-client

# Authentication Settings
NEXT_PUBLIC_KEYCLOAK_USE_PKCE=false
NEXT_PUBLIC_KEYCLOAK_RESPONSE_MODE=query

# Development mode (set to false for production)
NEXT_PUBLIC_DEV_MODE=false
NEXT_PUBLIC_USE_MOCK_API=false
```

### Fix 2: Update auth.tsx to Use Environment Variables

The auth.tsx file is currently hardcoded. We'll update it to use environment variables.

### Fix 3: Handle Network Access

Since Keycloak is on `192.168.1.117`, you need to:

1. **Access the frontend from the same IP**:
   ```bash
   # Instead of http://localhost:3000
   # Access via: http://192.168.1.117:3000
   ```

2. **Or update Next.js to listen on all interfaces**:
   ```bash
   # In package.json, update dev script:
   "dev": "next dev -H 0.0.0.0"
   ```

## Testing the Integration

### Test 1: Check Keycloak Accessibility

```bash
# From your machine
curl http://192.168.1.117:8080/realms/rag_app/.well-known/openid-configuration

# Should return JSON with endpoints
```

### Test 2: Test Login Flow

1. **Start the frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

2. **Open browser**:
   ```
   http://192.168.1.117:3000
   ```

3. **You should be redirected to Keycloak login page**:
   ```
   http://192.168.1.117:8080/realms/rag_app/protocol/openid-connect/auth?...
   ```

4. **After login, you should be redirected back to**:
   ```
   http://192.168.1.117:3000/?code=...&state=...
   ```

### Test 3: Check Browser Console

Open Developer Tools (F12) and look for:
- `[Auth] Starting auth initialization` logs
- No CORS errors
- Successful token exchange

## Common Issues and Solutions

### Issue 1: CORS Errors
```
Access to XMLHttpRequest at 'http://192.168.1.117:8080/...' 
from origin 'http://localhost:3000' has been blocked by CORS policy
```

**Solution**: Add `http://localhost:3000` to Web origins in Keycloak client settings.

### Issue 2: Invalid Redirect URI
```
Error: invalid_request
Description: Invalid redirect_uri
```

**Solution**: Add the exact redirect URI to "Valid redirect URIs" in Keycloak.

### Issue 3: Cannot Access from localhost
```
Frontend works on 192.168.1.117:3000 but not localhost:3000
```

**Solution**: Either:
- Always use `192.168.1.117:3000`
- Or add both URIs to Keycloak configuration
- Or run Keycloak on localhost as well

### Issue 4: Token/State Mismatch
```
Invalid state or nonce
```

**Solution**: Clear browser storage and try again:
```javascript
// In browser console:
sessionStorage.clear()
localStorage.clear()
// Refresh page
```

### Issue 5: Unauthorized Client Error
```
Error: unauthorized_client
```

**Solution**: Check Keycloak client settings:
- Client authentication should be OFF (public client)
- Standard flow should be enabled
- Valid redirect URIs must include your frontend URL

## Network Considerations

### Option A: Access Everything via 192.168.1.117 (Recommended)

```bash
# Keycloak:  http://192.168.1.117:8080
# API:       http://192.168.1.117:8000  
# Frontend:  http://192.168.1.117:3000
```

**Advantages**:
- Consistent network path
- No CORS issues
- Works across machines on network

**Next.js Dev Command**:
```bash
npm run dev -- -H 0.0.0.0 -p 3000
```

### Option B: Use localhost with Keycloak Proxy (Advanced)

Set up nginx or similar to proxy Keycloak to localhost:
```nginx
# Not recommended for development
```

### Option C: Run Keycloak on localhost (If possible)

```bash
# Stop current Keycloak
docker stop keycloak

# Run on localhost
docker run -p 8080:8080 ...
```

## Debugging Checklist

- [ ] Keycloak accessible at http://192.168.1.117:8080
- [ ] Realm `rag_app` exists
- [ ] Client `fastapi-client` configured as public
- [ ] Valid redirect URIs include frontend URL
- [ ] Web origins configured correctly
- [ ] Test user created and password set
- [ ] Frontend .env.local configured
- [ ] Frontend accessible on same network (192.168.1.117:3000)
- [ ] Browser console shows auth initialization
- [ ] No CORS errors in network tab

## Quick Test Command

Run this to verify Keycloak configuration:

```bash
# Test authorization endpoint
curl -v "http://192.168.1.117:8080/realms/rag_app/protocol/openid-connect/auth?client_id=fastapi-client&redirect_uri=http://192.168.1.117:3000/&response_type=code&scope=openid"

# Should redirect to login page (302)
```

## Next Steps After Configuration

1. Clear browser cache and cookies
2. Restart frontend development server
3. Access via http://192.168.1.117:3000
4. Check console for authentication logs
5. Login with test credentials
6. Verify token in console

If issues persist, check the detailed logs in browser console (F12 → Console tab).