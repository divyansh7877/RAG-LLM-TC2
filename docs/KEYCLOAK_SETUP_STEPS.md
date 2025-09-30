# Step-by-Step Keycloak Setup Guide

## ✅ What We've Done

1. Updated `frontend/lib/auth.tsx` to use environment variables
2. Configured `frontend/.env.local` with correct Keycloak settings
3. Updated `frontend/package.json` to listen on all network interfaces
4. Created test script to verify Keycloak configuration

## 🔧 Step 1: Configure Keycloak Client

### Access Keycloak Admin Console

1. Open your browser and go to: **http://192.168.1.117:8080**
2. Click on "Administration Console"
3. Login with admin credentials

### Configure the Client

1. **Select Realm**
   - In the top-left dropdown, select: **rag_app**

2. **Navigate to Clients**
   - Left sidebar → **Clients**
   - Click on **fastapi-client**

3. **Settings Tab - Configure These Values:**

   | Setting | Value |
   |---------|-------|
   | **Client ID** | `fastapi-client` |
   | **Name** | FastAPI Client |
   | **Enabled** | ON |
   | **Client authentication** | OFF (must be public client) |
   | **Authorization** | OFF |

4. **Authentication Flow - Enable These:**
   - ☑ **Standard flow** (MUST be checked)
   - ☐ Implicit flow (leave unchecked)
   - ☑ **Direct access grants** (check this)
   - ☐ Service accounts roles (leave unchecked)

5. **Valid Redirect URIs** (Add these exactly):
   ```
   http://localhost:3000/*
   http://192.168.1.117:3000/*
   http://localhost/*
   ```

6. **Valid Post Logout Redirect URIs**:
   ```
   http://localhost:3000/*
   http://192.168.1.117:3000/*
   ```

7. **Web Origins** (Add these):
   ```
   http://localhost:3000
   http://192.168.1.117:3000
   http://localhost
   +
   ```
   *Note: The `+` is a special value that allows all origins from Valid Redirect URIs*

8. **Click Save** at the bottom

### Advanced Settings (Optional but Recommended)

1. Click on **Advanced** tab
2. Configure:
   ```
   Proof Key for Code Exchange Code Challenge Method: (empty/not set)
   
   Access Token Lifespan: 5 Minutes
   Client Session Idle: 30 Minutes  
   Client Session Max: 10 Hours
   ```

3. **Click Save**

## 👤 Step 2: Create or Verify Test User

### Check if Users Exist

1. Left sidebar → **Users**
2. Click **View all users** button

### Create New User (if needed)

1. Click **Add user** button
2. Fill in:
   ```
   Username: testuser
   Email: testuser@example.com
   Email verified: ON (toggle switch)
   Enabled: ON (toggle switch)
   ```
3. Click **Create**

### Set Password

1. After creating user, click on **Credentials** tab
2. Click **Set password**
3. Enter:
   ```
   Password: testpassword
   Password confirmation: testpassword
   Temporary: OFF (toggle switch)
   ```
4. Click **Save**
5. Confirm by clicking **Save password**

### Assign to Groups (Optional)

1. Click on **Groups** tab
2. Select groups if you have any (like `default`, `assistants`, etc.)
3. Click **Join**

## 🚀 Step 3: Test the Frontend

### Start the Frontend

```bash
cd frontend
npm run dev
```

You should see output like:
```
- ready started server on 0.0.0.0:3000, url: http://localhost:3000
- Local: http://localhost:3000
- Network: http://192.168.1.117:3000
```

### Access the Application

Open your browser and go to: **http://192.168.1.117:3000**

⚠️ **Important**: Use `192.168.1.117:3000` NOT `localhost:3000` for best results

### What Should Happen

1. **Initial Load**:
   - Browser console (F12) shows: `[Auth] Starting auth initialization`
   - Page attempts to check authentication

2. **If Not Authenticated**:
   - You should see a login button
   - OR automatically redirect to Keycloak login page

3. **Keycloak Login Page**:
   - URL will be: `http://192.168.1.117:8080/realms/rag_app/protocol/openid-connect/auth?...`
   - Enter username: `testuser`
   - Enter password: `testpassword`
   - Click **Sign In**

4. **After Successful Login**:
   - Redirect back to: `http://192.168.1.117:3000/?code=...&state=...`
   - Console shows: `[Auth] User is authenticated`
   - You should see the main application interface

## 🐛 Troubleshooting

### Issue 1: "Invalid redirect_uri"

**Error Message**: 
```
We're sorry...
Invalid parameter: redirect_uri
```

**Solution**: 
1. Go back to Keycloak admin console
2. Client → fastapi-client → Settings
3. Check **Valid Redirect URIs** includes: `http://192.168.1.117:3000/*`
4. Save and try again

### Issue 2: CORS Error in Console

**Error Message**:
```
Access to XMLHttpRequest blocked by CORS policy
```

**Solution**:
1. Check **Web origins** in Keycloak client settings
2. Must include: `http://192.168.1.117:3000` or `+`
3. Save and clear browser cache (Ctrl+Shift+Delete)

### Issue 3: Stuck on Login Page

**Symptoms**: Enters credentials but stays on login page

**Solutions**:
1. Clear browser storage:
   - Press F12 → Application tab → Clear storage
   - Or in console: `sessionStorage.clear(); localStorage.clear()`
   
2. Check user is enabled:
   - Keycloak → Users → testuser
   - Enabled should be ON

3. Check password:
   - Go to Credentials tab
   - Reset password if needed

### Issue 4: "Unauthorized client"

**Error Message**:
```
error: unauthorized_client
```

**Solution**:
- Client authentication must be **OFF** (public client)
- Standard flow must be **ON**
- Save and try again

### Issue 5: Works on 192.168.1.117:3000 but not localhost:3000

**This is expected!**

Since Keycloak is on `192.168.1.117:8080`, you should access the frontend the same way.

**Options**:
- **Recommended**: Always use `http://192.168.1.117:3000`
- **Alternative**: Add localhost URLs to Keycloak configuration

## ✅ Verification Checklist

Before testing, ensure:

- [ ] Keycloak admin console accessible at http://192.168.1.117:8080
- [ ] Realm **rag_app** exists and is selected
- [ ] Client **fastapi-client** exists
- [ ] Client authentication is **OFF** (public client)
- [ ] Standard flow is **ENABLED**
- [ ] Valid redirect URIs includes: `http://192.168.1.117:3000/*`
- [ ] Web origins includes: `http://192.168.1.117:3000` or `+`
- [ ] Test user created with password set (temporary = OFF)
- [ ] Frontend `.env.local` configured correctly
- [ ] Frontend started with `npm run dev`
- [ ] Accessing frontend via `http://192.168.1.117:3000`

## 📊 Testing with Browser Console

Open Developer Tools (F12) and check console for these logs:

### Successful Authentication Flow:
```
[Auth] Initializing Keycloak with config: {url: "http://192.168.1.117:8080", ...}
[Auth] Starting auth initialization
[Auth] Current URL: http://192.168.1.117:3000/
[Auth] Getting Keycloak singleton instance
[Auth] Calling keycloak.init()
[Auth] Keycloak init completed. Authenticated: true
[Auth] User is authenticated, setting up user data
```

### Login Redirect Flow:
```
[Auth] User is not authenticated
[Auth] Initiating login
// Browser redirects to Keycloak
// After login, redirects back with code
[Auth] Has auth code in URL: true
[Auth] User is authenticated
```

## 🎯 Quick Test Commands

```bash
# 1. Test Keycloak accessibility
curl http://192.168.1.117:8080/realms/rag_app/.well-known/openid-configuration

# 2. Start frontend
cd frontend && npm run dev

# 3. Open browser
# Go to: http://192.168.1.117:3000

# 4. Check console (F12) for auth logs
```

## 📝 Common Configuration Mistakes

1. ❌ Client authentication is ON → Should be OFF
2. ❌ Standard flow is disabled → Must be ENABLED
3. ❌ Redirect URI missing trailing `/*` → Add `/*`
4. ❌ Using localhost instead of 192.168.1.117 → Use 192.168.1.117
5. ❌ Web origins not configured → Add `+` or specific origin
6. ❌ User password is temporary → Set temporary to OFF

## 🆘 Still Having Issues?

1. **Check logs**:
   ```bash
   # Keycloak logs (if running in Docker)
   docker logs keycloak -f
   ```

2. **Browser Network Tab** (F12 → Network):
   - Look for failed requests to Keycloak
   - Check for 400/401/403 errors
   - Examine request/response headers

3. **Clear Everything**:
   ```bash
   # Stop frontend
   Ctrl+C
   
   # Clear browser completely
   # Press Ctrl+Shift+Delete → Clear all data
   
   # Restart frontend
   cd frontend && npm run dev
   
   # Access again via 192.168.1.117:3000
   ```

4. **Export Keycloak client config** to verify settings:
   - Keycloak → Clients → fastapi-client → Action → Export
   - Check JSON configuration

## 🎉 Success Indicators

When everything works correctly:

- ✅ No CORS errors in console
- ✅ Successful redirect to/from Keycloak
- ✅ Console shows "User is authenticated"
- ✅ User info displayed in UI
- ✅ Can make authenticated API calls
- ✅ Token visible in browser storage (Application tab)